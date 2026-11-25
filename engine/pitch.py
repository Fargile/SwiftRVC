# pitch extractors: bn-folded mlx rmvpe + fft yin + f0 conditioning

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _hz_to_mel_htk(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz_htk(mels: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


# htk + slaney mel basis matching librosa.filters.mel(..., htk=True)
def build_htk_mel_basis(
    sr: int = 16000,
    n_fft: int = 1024,
    n_mels: int = 128,
    fmin: float = 30.0,
    fmax: float = 8000.0,
) -> np.ndarray:
    n_freqs = n_fft // 2 + 1
    fftfreqs = np.linspace(0.0, sr / 2.0, n_freqs, dtype=np.float64)
    min_mel = _hz_to_mel_htk(np.array(fmin, dtype=np.float64))
    max_mel = _hz_to_mel_htk(np.array(fmax, dtype=np.float64))
    mels = np.linspace(min_mel, max_mel, n_mels + 2, dtype=np.float64)
    mel_f = _mel_to_hz_htk(mels)

    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fftfreqs[None, :]
    
    weights = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))

    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, None]
    return weights.astype(np.float32)


def _fold_bn(
    conv_w: mx.array,
    conv_b: Optional[mx.array],
    bn_w: mx.array,
    bn_b: mx.array,
    bn_mean: mx.array,
    bn_var: mx.array,
    eps: float = 1e-5,
) -> Tuple[mx.array, mx.array]:
    w = conv_w.astype(mx.float32)
    b = mx.zeros((w.shape[0],), dtype=mx.float32) if conv_b is None else conv_b.astype(mx.float32)
    inv_std = mx.rsqrt(bn_var.astype(mx.float32) + eps)
    scale = bn_w.astype(mx.float32) * inv_std
    w_fused = w * scale.reshape(-1, 1, 1, 1)
    b_fused = (b - bn_mean.astype(mx.float32)) * scale + bn_b.astype(mx.float32)
    return w_fused, b_fused
    
        
    
    
    
# rmvpe

class _BiGRU(nn.Module):
    def __init__(self, input_size: int = 384, hidden_size: int = 256):
        super().__init__()
        self.gru_forward = nn.GRU(input_size, hidden_size)
        self.gru_backward = nn.GRU(input_size, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        out_f = self.gru_forward(x)
        out_b = self.gru_backward(x[:, ::-1, :])[:, ::-1, :]
        return mx.concatenate([out_f, out_b], axis=-1)


class _FusedConvBlockRes(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True) if in_ch != out_ch else None

    def __call__(self, x: mx.array) -> mx.array:
        h = nn.relu(self.conv1(x))
        h = nn.relu(self.conv2(h))
        if self.shortcut is not None:
            x = self.shortcut(x)
        return h + x


class _ResEncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool, n_blocks: int = 4):
        super().__init__()
        self.pool = nn.AvgPool2d((2, 2)) if pool else None
        self.conv_blocks = [_FusedConvBlockRes(in_ch, out_ch)] + [
            _FusedConvBlockRes(out_ch, out_ch) for _ in range(n_blocks - 1)
        ]

    def __call__(self, x: mx.array):
        for blk in self.conv_blocks:
            x = blk(x)
        if self.pool is not None:
            return x, self.pool(x)
        return x


class _ResDecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_blocks: int = 4):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=3, stride=(2, 2), padding=1, bias=True
        )
        self.conv_blocks = [_FusedConvBlockRes(out_ch * 2, out_ch)] + [
            _FusedConvBlockRes(out_ch, out_ch) for _ in range(n_blocks - 1)
        ]

    def __call__(self, x: mx.array, skip: mx.array) -> mx.array:
        x = nn.relu(self.conv_transpose(x))
        th, tw = skip.shape[1], skip.shape[2]
        ch, cw = x.shape[1], x.shape[2]
        if ch > th:
            x = x[:, :th, :, :]
        elif ch < th:
            x = mx.pad(x, ((0, 0), (0, th - ch), (0, 0), (0, 0)))
        if cw > tw:
            x = x[:, :, :tw, :]
        elif cw < tw:
            x = mx.pad(x, ((0, 0), (0, 0), (0, tw - cw), (0, 0)))
        x = mx.concatenate([x, skip], axis=-1)
        for blk in self.conv_blocks:
            x = blk(x)
        return x


class RMVPEMLX(nn.Module):
    def __init__(self, use_fp16: bool = True, hop_length: int = 160):
        super().__init__()
        self.compute_dtype = mx.float16 if use_fp16 else mx.float32
        self.hop_length = hop_length

        self.in_scale = mx.ones((1,), dtype=mx.float32)
        self.in_shift = mx.zeros((1,), dtype=mx.float32)

        self.enc_layers: List[_ResEncoderBlock] = []
        c_in, c_out = 1, 16
        for _ in range(5):
            self.enc_layers.append(_ResEncoderBlock(c_in, c_out, pool=True, n_blocks=4))
            c_in, c_out = c_out, c_out * 2

        self.inter_layers: List[_ResEncoderBlock] = [
            _ResEncoderBlock(256, 512, pool=False, n_blocks=4)
        ] + [_ResEncoderBlock(512, 512, pool=False, n_blocks=4) for _ in range(3)]

        self.dec_layers: List[_ResDecoderBlock] = []
        dc = 512
        for _ in range(5):
            self.dec_layers.append(_ResDecoderBlock(dc, dc // 2, n_blocks=4))
            dc //= 2

        self.cnn = nn.Conv2d(16, 3, kernel_size=3, padding=1, bias=True)
        self.bigru = _BiGRU(3 * 128, 256)
        self.linear = nn.Linear(512, 360, bias=True)
        
        self._mel_basis = mx.array(build_htk_mel_basis(16000, 1024, 128, 30.0, 8000.0))
        n = np.arange(1024, dtype=np.float32)
        self._hann = mx.array(0.5 - 0.5 * np.cos(2.0 * np.pi * n / 1024.0))
        cents = 20.0 * np.arange(360, dtype=np.float32) + 1997.3794084376191
        self._cents_pad = np.pad(cents, (4, 4))

    @classmethod
    def from_safetensors(cls, path: str, use_fp16: bool = True) -> "RMVPEMLX":
        model = cls(use_fp16=use_fp16)
        dtype = model.compute_dtype
        raw: Dict[str, mx.array] = mx.load(path)

        inv_std = mx.rsqrt(raw["model.unet.encoder.bn.running_var"].astype(mx.float32) + 1e-5)
        scale = raw["model.unet.encoder.bn.weight"].astype(mx.float32) * inv_std
        shift = (
            raw["model.unet.encoder.bn.bias"].astype(mx.float32)
            - raw["model.unet.encoder.bn.running_mean"].astype(mx.float32) * scale
        )
        model.in_scale = scale.astype(dtype)
        model.in_shift = shift.astype(dtype)

        def load_res_block(blk: _FusedConvBlockRes, prefix: str):
            w1, b1 = _fold_bn(
                raw[f"{prefix}.conv1.weight"],
                raw.get(f"{prefix}.conv1.bias"),
                raw[f"{prefix}.bn1.weight"],
                raw[f"{prefix}.bn1.bias"],
                raw[f"{prefix}.bn1.running_mean"],
                raw[f"{prefix}.bn1.running_var"],
            )
            w2, b2 = _fold_bn(
                raw[f"{prefix}.conv2.weight"],
                raw.get(f"{prefix}.conv2.bias"),
                raw[f"{prefix}.bn2.weight"],
                raw[f"{prefix}.bn2.bias"],
                raw[f"{prefix}.bn2.running_mean"],
                raw[f"{prefix}.bn2.running_var"],
            )
            blk.conv1.weight = w1.astype(dtype)
            blk.conv1.bias = b1.astype(dtype)
            blk.conv2.weight = w2.astype(dtype)
            blk.conv2.bias = b2.astype(dtype)
            if blk.shortcut is not None:
                blk.shortcut.weight = raw[f"{prefix}.shortcut.weight"].astype(dtype)
                blk.shortcut.bias = raw[f"{prefix}.shortcut.bias"].astype(dtype)

        for i, layer in enumerate(model.enc_layers):
            for j, blk in enumerate(layer.conv_blocks):
                load_res_block(blk, f"model.unet.encoder.layers.{i}.conv_blocks.{j}")

        for i, layer in enumerate(model.inter_layers):
            for j, blk in enumerate(layer.conv_blocks):
                load_res_block(blk, f"model.unet.intermediate.layers.{i}.conv_blocks.{j}")

        for i, layer in enumerate(model.dec_layers):
            p = f"model.unet.decoder.layers.{i}"
            wt, bt = _fold_bn(
                raw[f"{p}.conv_transpose.weight"],
                raw.get(f"{p}.conv_transpose.bias"),
                raw[f"{p}.bn.weight"],
                raw[f"{p}.bn.bias"],
                raw[f"{p}.bn.running_mean"],
                raw[f"{p}.bn.running_var"],
            )
            layer.conv_transpose.weight = wt.astype(dtype)
            layer.conv_transpose.bias = bt.astype(dtype)
            for j, blk in enumerate(layer.conv_blocks):
                load_res_block(blk, f"{p}.conv_blocks.{j}")

        model.cnn.weight = raw["model.cnn.weight"].astype(dtype)
        model.cnn.bias = raw["model.cnn.bias"].astype(dtype)

        # keep bigru + linear in fp32 for pitch accuracy
        for direction in ("gru_forward", "gru_backward"):
            g = getattr(model.bigru, direction)
            p = f"model.bigru.{direction}"
            g.Wx = raw[f"{p}.Wx"].astype(mx.float32)
            g.Wh = raw[f"{p}.Wh"].astype(mx.float32)
            g.b = raw[f"{p}.b"].astype(mx.float32)
            g.bhn = raw[f"{p}.bhn"].astype(mx.float32)

        model.linear.weight = raw["model.linear.weight"].astype(mx.float32)
        model.linear.bias = raw["model.linear.bias"].astype(mx.float32)

        mx.eval(model.parameters())
        del raw
        mx.clear_cache()
        model.eval()
        return model

    def _compute_log_mel(self, audio_np: np.ndarray) -> mx.array:
        pad = 512
        if audio_np.shape[0] <= pad:
            padded = np.pad(audio_np, (pad, pad), mode="edge")
        else:
            padded = np.pad(audio_np, (pad, pad), mode="reflect")

        n_frames = 1 + (padded.shape[0] - 1024) // self.hop_length
        stride = padded.strides[0]
        frames_np = np.lib.stride_tricks.as_strided(
            padded,
            shape=(n_frames, 1024),
            strides=(self.hop_length * stride, stride),
        )
        frames = mx.array(np.ascontiguousarray(frames_np, dtype=np.float32))
        spec = mx.fft.rfft(frames * self._hann, axis=-1)
        mag = mx.abs(spec)
        mel = mx.matmul(mag, self._mel_basis.T)
        log_mel = mx.log(mx.maximum(mel, 1e-5))
        return log_mel

    def _forward_net(self, mel_4d: mx.array) -> mx.array:
        x = mel_4d.astype(self.compute_dtype) * self.in_scale + self.in_shift
        skips = []
        for layer in self.enc_layers:
            s, x = layer(x)
            skips.append(s)
        for layer in self.inter_layers:
            x = layer(x)
        for i, layer in enumerate(self.dec_layers):
            x = layer(x, skips[-1 - i])
        x = self.cnn(x)
        x = mx.transpose(x, (0, 1, 3, 2)).reshape(1, x.shape[1], 384).astype(mx.float32)
        x = self.bigru(x)
        x = mx.sigmoid(self.linear(x))
        return x

    def infer(self, audio_16k: np.ndarray, threshold: float = 0.045) -> np.ndarray:
        audio_16k = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
        log_mel = self._compute_log_mel(audio_16k)
        n_frames = log_mel.shape[0]
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if n_pad > 0:
            log_mel = mx.pad(log_mel, ((0, n_pad), (0, 0)))
        mel_4d = log_mel[None, :, :, None]
        hidden = self._forward_net(mel_4d)[0, :n_frames, :]
        mx.eval(hidden)
        return self._decode_vectorized(np.asarray(hidden, dtype=np.float32), threshold)

    def _decode_vectorized(self, hidden: np.ndarray, threshold: float = 0.03) -> np.ndarray:
        T = hidden.shape[0]
        center = np.argmax(hidden, axis=1)
        max_val = hidden[np.arange(T), center]

        h_pad = np.pad(hidden, ((0, 0), (4, 4)))
        offsets = np.arange(-4, 5, dtype=np.int64)[None, :]
        idx = (center[:, None] + 4) + offsets

        salience = np.take_along_axis(h_pad, idx, axis=1)
        cents_local = self._cents_pad[idx]
        cents = np.sum(salience * cents_local, axis=1) / (np.sum(salience, axis=1) + 1e-9)
        cents[max_val <= threshold] = 0.0

        f0 = 10.0 * (2.0 ** (cents / 1200.0))
        f0[f0 <= 10.0001] = 0.0
        return f0.astype(np.float32)
    
    
        
    
    

# fast fft yin

class FastYINPitch:
    def __init__(
        self,
        sr: int = 16000,
        hop_length: int = 160,
        frame_length: int = 1024,
        f0_min: float = 55.0,
        f0_max: float = 1050.0,
        threshold: float = 0.16,
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.tau_min = max(2, int(sr / f0_max))
        self.tau_max = min(frame_length // 2, int(sr / f0_min))
        self.threshold = threshold
        self.n_fft = 1 << (frame_length - 1).bit_length() * 1
        if self.n_fft < frame_length * 2:
            self.n_fft = frame_length * 2

    def infer(self, audio_16k: np.ndarray) -> np.ndarray:
        x = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
        pad = self.frame_length // 2
        padded = np.pad(x, (pad, pad), mode="reflect" if len(x) > pad else "edge")
        n_frames = 1 + (len(padded) - self.frame_length) // self.hop_length
        stride = padded.strides[0]
        frames = np.ascontiguousarray(
            np.lib.stride_tricks.as_strided(
                padded,
                shape=(n_frames, self.frame_length),
                strides=(self.hop_length * stride, stride),
            )
        )

        frame_rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
        
        W = self.frame_length // 2
        x_pad = np.zeros((n_frames, self.n_fft), dtype=np.float32)
        x_pad[:, : self.frame_length] = frames
        fx = np.fft.rfft(x_pad, axis=1)

        x_w = np.zeros((n_frames, self.n_fft), dtype=np.float32)
        x_w[:, :W] = frames[:, :W]
        fw = np.fft.rfft(x_w, axis=1)
        acf = np.fft.irfft(np.conj(fw) * fx, n=self.n_fft, axis=1)[:, : self.tau_max + 2]

        sq = frames * frames
        csq = np.concatenate([np.zeros((n_frames, 1), dtype=np.float32), np.cumsum(sq, axis=1)], axis=1)
        e1 = csq[:, W : W + 1]
        taus = np.arange(self.tau_max + 2)
        e2 = csq[:, taus + W] - csq[:, taus]
        diff = np.maximum(e1 + e2 - 2.0 * acf, 0.0)

        cmndf = np.ones_like(diff)
        running_sum = np.cumsum(diff[:, 1:], axis=1)
        lag_idx = np.arange(1, diff.shape[1], dtype=np.float32)[None, :]
        cmndf[:, 1:] = diff[:, 1:] * lag_idx / np.maximum(running_sum, 1e-12)

        sub = cmndf[:, self.tau_min : self.tau_max]
        f0 = np.zeros(n_frames, dtype=np.float32)
        below = sub < self.threshold
        has_dip = np.any(below, axis=1)

        for t in range(n_frames):
            if frame_rms[t] < 0.002:
                continue
            if has_dip[t]:
                k = int(np.argmax(below[t]))
                tau = self.tau_min + k
                while tau + 1 < self.tau_max and cmndf[t, tau + 1] < cmndf[t, tau]:
                    tau += 1
            else:
                k = int(np.argmin(sub[t]))
                tau = self.tau_min + k
                if cmndf[t, tau] > 0.35:
                    continue

            if 1 <= tau < cmndf.shape[1] - 1:
                y0, y1, y2 = cmndf[t, tau - 1], cmndf[t, tau], cmndf[t, tau + 1]
                denom = 2.0 * (y0 - 2.0 * y1 + y2)
                delta = (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
                tau_refined = tau + np.clip(delta, -0.5, 0.5)
            else:
                tau_refined = float(tau)

            if tau_refined > 0:
                f0[t] = self.sr / tau_refined

        if n_frames >= 3:
            out = f0.copy()
            for t in range(1, n_frames - 1):
                if f0[t] > 0 and f0[t - 1] > 0 and f0[t + 1] > 0:
                    out[t] = np.median(f0[t - 1 : t + 2])
            f0 = out

        return f0
    
        
    
    
def condition_f0(
    f0: np.ndarray,
    smooth: bool = True,
    f0_min: float = 55.0,
    f0_max: float = 1050.0,
    pitch_floor_hz: float = 0.0,
) -> np.ndarray:
    out = np.asarray(f0, dtype=np.float32).copy()
    n = len(out)
    if n < 3:
        return out

    if smooth:
        # drop 1-frame voiced spikes
        for t in range(1, n - 1):
            if out[t] > 0.0 and out[t - 1] <= 0.0 and out[t + 1] <= 0.0:
                out[t] = 0.0

        # voiced-only 3-frame median
        med = out.copy()
        for t in range(1, n - 1):
            if out[t] > 0.0 and out[t - 1] > 0.0 and out[t + 1] > 0.0:
                a, b, c = out[t - 1], out[t], out[t + 1]
                if a > b:
                    a, b = b, a
                if b > c:
                    b, c = c, b
                if a > b:
                    a, b = b, a
                med[t] = b
        out = med

        # light smoothing inside contiguous voiced spans
        sm = out.copy()
        for t in range(1, n - 1):
            if out[t] > 0.0 and out[t - 1] > 0.0 and out[t + 1] > 0.0:
                if abs(out[t] - out[t - 1]) < out[t] * 0.25 and abs(out[t] - out[t + 1]) < out[t] * 0.25:
                    sm[t] = 0.2 * out[t - 1] + 0.6 * out[t] + 0.2 * out[t + 1]
        out = sm

    voiced = out > 0.0
    if np.any(voiced):
        v = out[voiced]
        if pitch_floor_hz > 40.0:
            # 4-semitone soft knee above floor
            st = 12.0 * np.log2(np.maximum(v, 20.0) / float(pitch_floor_hz))
            knee = 4.0
            low = st < knee
            if np.any(low):
                below = knee - st[low]
                st[low] = knee - knee * np.tanh(below / knee)
                v[low] = float(pitch_floor_hz) * (2.0 ** (st[low] / 12.0))

        hi_knee = 850.0
        above = v > hi_knee
        if np.any(above):
            v[above] = hi_knee + (f0_max - hi_knee) * np.tanh((v[above] - hi_knee) / (f0_max - hi_knee))
        v = np.clip(v, f0_min, f0_max)
        out[voiced] = v

    return out


def f0_to_coarse(f0: np.ndarray, f0_min: float = 50.0, f0_max: float = 1100.0) -> np.ndarray:
    f0_mel_min = 1127.0 * np.log(1.0 + f0_min / 700.0)
    f0_mel_max = 1127.0 * np.log(1.0 + f0_max / 700.0)
    f0_mel = 1127.0 * np.log(1.0 + np.maximum(f0, 0.0) / 700.0)
    voiced = f0_mel > 0
    f0_mel[voiced] = (f0_mel[voiced] - f0_mel_min) * 254.0 / (f0_mel_max - f0_mel_min) + 1.0
    f0_mel[f0_mel <= 1.0] = 1.0
    f0_mel[f0_mel > 255.0] = 255.0
    return np.rint(f0_mel).astype(np.int32)
