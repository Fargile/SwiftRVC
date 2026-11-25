# mlx rvc v1/v2 synthesizer (text encoder, flow, nsf-hifigan)

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .pth_loader import RVCCheckpoint


def _bake_wn(weights: Dict[str, np.ndarray], prefix: str) -> np.ndarray:
    wk = f"{prefix}.weight"
    if wk in weights:
        return weights[wk].astype(np.float32)
    g = weights[f"{prefix}.weight_g"].astype(np.float32)
    v = weights[f"{prefix}.weight_v"].astype(np.float32)
    norm_axes = tuple(range(1, v.ndim))
    v_norm = np.sqrt(np.sum(v * v, axis=norm_axes, keepdims=True))
    return g * (v / np.maximum(v_norm, 1e-12))


def _pt_conv1d_to_mlx(w_pt: np.ndarray, dtype: mx.Dtype) -> mx.array:
    # (c_out, c_in, k) -> (c_out, k, c_in)
    return mx.array(np.ascontiguousarray(np.transpose(w_pt, (0, 2, 1)))).astype(dtype)


def _pt_convt1d_to_mlx(w_pt: np.ndarray, dtype: mx.Dtype) -> mx.array:
    # (c_in, c_out, k) -> (c_out, k, c_in)
    return mx.array(np.ascontiguousarray(np.transpose(w_pt, (1, 2, 0)))).astype(dtype)


class _RvcLayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = mx.ones((channels,), dtype=mx.float32)
        self.beta = mx.zeros((channels,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        orig = x.dtype
        xf = x.astype(mx.float32)
        mean = mx.mean(xf, axis=-1, keepdims=True)
        var = mx.mean((xf - mean) ** 2, axis=-1, keepdims=True)
        xf = (xf - mean) * mx.rsqrt(var + self.eps) * self.gamma + self.beta
        return xf.astype(orig)
    
        
    
    
class _RelPosMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        n_heads: int = 2,
        window_size: int = 10,
        dtype: mx.Dtype = mx.float16,
    ):
        super().__init__()
        self.channels = channels
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size
        self.scale = self.k_channels ** -0.5

        self.conv_q = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv_k = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv_v = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv_o = nn.Conv1d(channels, channels, kernel_size=1)

        self.emb_rel_k = mx.zeros((1, 2 * window_size + 1, self.k_channels), dtype=dtype)
        self.emb_rel_v = mx.zeros((1, 2 * window_size + 1, self.k_channels), dtype=dtype)

    def _get_rel_emb(self, emb: mx.array, length: int) -> mx.array:
        W = self.window_size
        pad_len = max(length - (W + 1), 0)
        start = max((W + 1) - length, 0)
        end = start + 2 * length - 1
        e = emb[0]
        if pad_len > 0:
            e = mx.pad(e, ((pad_len, pad_len), (0, 0)))
        return e[start:end]

    def __call__(self, x: mx.array, rel_idx: mx.array) -> mx.array:
        B, T, C = x.shape
        q = self.conv_q(x).reshape(B, T, self.n_heads, self.k_channels).transpose(0, 2, 1, 3)
        k = self.conv_k(x).reshape(B, T, self.n_heads, self.k_channels).transpose(0, 2, 1, 3)
        v = self.conv_v(x).reshape(B, T, self.n_heads, self.k_channels).transpose(0, 2, 1, 3)

        qs = q * self.scale
        scores = mx.matmul(qs, k.transpose(0, 1, 3, 2))

        rel_k = self._get_rel_emb(self.emb_rel_k, T)
        rel_logits = mx.matmul(qs, rel_k.T)
        rel_scores = mx.take_along_axis(rel_logits, rel_idx[None, None, :, :], axis=-1)
        scores = scores + rel_scores

        p_attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
        out = mx.matmul(p_attn, v)

        rel_v = self._get_rel_emb(self.emb_rel_v, T)
        rel_v_abs = rel_v[rel_idx]
        out_rel = mx.einsum("bhij,ijd->bhid", p_attn, rel_v_abs)
        out = (out + out_rel).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.conv_o(out)


class _FFN(nn.Module):
    def __init__(self, in_ch: int, filter_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv_1 = nn.Conv1d(in_ch, filter_ch, kernel_size=kernel_size, padding=pad)
        self.conv_2 = nn.Conv1d(filter_ch, in_ch, kernel_size=kernel_size, padding=pad)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv_2(nn.relu(self.conv_1(x)))


class _TextEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        f0: bool = True,
        dtype: mx.Dtype = mx.float16,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.scale = math.sqrt(hidden_channels)
        self.f0 = f0

        self.emb_phone = nn.Linear(in_channels, hidden_channels)
        self.emb_pitch = nn.Embedding(256, hidden_channels) if f0 else None

        self.attn_layers = [
            _RelPosMultiHeadAttention(hidden_channels, n_heads, window_size=10, dtype=dtype)
            for _ in range(n_layers)
        ]
        self.norm_layers_1 = [_RvcLayerNorm(hidden_channels) for _ in range(n_layers)]
        self.ffn_layers = [_FFN(hidden_channels, filter_channels, kernel_size) for _ in range(n_layers)]
        self.norm_layers_2 = [_RvcLayerNorm(hidden_channels) for _ in range(n_layers)]
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, kernel_size=1)

    def __call__(self, phone: mx.array, pitch: Optional[mx.array]) -> Tuple[mx.array, mx.array]:
        x = self.emb_phone(phone)
        if self.f0 and pitch is not None:
            x = x + self.emb_pitch(pitch)
        x = nn.leaky_relu(x * self.scale, negative_slope=0.1)

        T = x.shape[1]
        r = mx.arange(T, dtype=mx.int32)
        rel_idx = r[None, :] - r[:, None] + (T - 1)

        for attn, n1, ffn, n2 in zip(
            self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2
        ):
            x = n1(x + attn(x, rel_idx))
            x = n2(x + ffn(x))

        stats = self.proj(x)
        m, logs = mx.split(stats, 2, axis=-1)
        return m, logs


class _WaveNet(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 192,
        kernel_size: int = 5,
        n_layers: int = 3,
        gin_channels: int = 256,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        pad = (kernel_size - 1) // 2
        self.cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, kernel_size=1)
        self.in_layers = [
            nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size=kernel_size, padding=pad)
            for _ in range(n_layers)
        ]
        self.res_skip_layers = [
            nn.Conv1d(
                hidden_channels,
                2 * hidden_channels if i < n_layers - 1 else hidden_channels,
                kernel_size=1,
            )
            for i in range(n_layers)
        ]

    def __call__(self, x: mx.array, g: mx.array) -> mx.array:
        g_cond = self.cond_layer(g)
        out = mx.zeros_like(x)
        C = self.hidden_channels
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            g_l = g_cond[:, :, i * 2 * C : (i + 1) * 2 * C]
            a = x_in + g_l
            acts = mx.tanh(a[:, :, :C]) * mx.sigmoid(a[:, :, C:])
            rs = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = x + rs[:, :, :C]
                out = out + rs[:, :, C:]
            else:
                out = out + rs
        return out


class _ResidualCouplingLayer(nn.Module):
    def __init__(self, channels: int = 192, hidden_channels: int = 192, gin_channels: int = 256):
        super().__init__()
        half = channels // 2
        self.pre = nn.Conv1d(half, hidden_channels, kernel_size=1)
        self.enc = _WaveNet(hidden_channels, kernel_size=5, n_layers=3, gin_channels=gin_channels)
        self.post = nn.Conv1d(hidden_channels, half, kernel_size=1)

    def reverse(self, x: mx.array, g: mx.array) -> mx.array:
        x0, x1 = mx.split(x, 2, axis=-1)
        h = self.pre(x0)
        h = self.enc(h, g)
        m = self.post(h)
        x1 = x1 - m
        return mx.concatenate([x0, x1], axis=-1)


class _ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: Tuple[int, int, int] = (1, 3, 5)):
        super().__init__()
        self.convs1 = [
            nn.Conv1d(
                channels, channels, kernel_size=kernel_size, dilation=d,
                padding=(kernel_size * d - d) // 2,
            )
            for d in dilations
        ]
        self.convs2 = [
            nn.Conv1d(
                channels, channels, kernel_size=kernel_size, dilation=1,
                padding=(kernel_size - 1) // 2,
            )
            for _ in dilations
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = nn.leaky_relu(x, negative_slope=0.1)
            xt = c1(xt)
            xt = nn.leaky_relu(xt, negative_slope=0.1)
            xt = c2(xt)
            x = x + xt
        return x
    
    
        
    
    

class _GeneratorNSF(nn.Module):
    def __init__(
        self,
        initial_channel: int,
        resblock_kernel_sizes: List[int],
        resblock_dilation_sizes: List[List[int]],
        upsample_rates: List[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: List[int],
        gin_channels: int,
        sr: int,
        dtype: mx.Dtype = mx.float16,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.sr = sr
        self.upp = int(np.prod(upsample_rates))
        self.compute_dtype = dtype

        self.source_linear = nn.Linear(1, 1, bias=True)

        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, kernel_size=7, padding=3)
        self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, kernel_size=1) if gin_channels > 0 else None

        self.ups = []
        self.noise_convs = []
        self.resblocks = []
        
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_in = upsample_initial_channel // (2 ** i)
            c_out = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                nn.ConvTranspose1d(c_in, c_out, kernel_size=k, stride=u, padding=(k - u) // 2)
            )
            if i + 1 < self.num_upsamples:
                stride_f0 = int(np.prod(upsample_rates[i + 1 :]))
                self.noise_convs.append(
                    nn.Conv1d(1, c_out, kernel_size=stride_f0 * 2, stride=stride_f0, padding=stride_f0 // 2)
                )
            else:
                self.noise_convs.append(nn.Conv1d(1, c_out, kernel_size=1, stride=1, padding=0))

            for rk, rd in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(_ResBlock1(c_out, rk, tuple(rd)))

        final_ch = upsample_initial_channel // (2 ** self.num_upsamples)
        self.conv_post = nn.Conv1d(final_ch, 1, kernel_size=7, padding=3, bias=False)

    def _sine_source(self, f0: mx.array, noise: mx.array) -> mx.array:
        # keep phase accum in fp32 so high notes dont drift
        B, T = f0.shape
        f0_up = mx.repeat(f0[:, :, None], self.upp, axis=1)
        rad = (f0_up / float(self.sr)) % 1.0
        phase = (mx.cumsum(rad, axis=1) % 1.0) * (2.0 * math.pi)
        uv = (f0_up > 10.0).astype(mx.float32)
        sine_amp = 0.1
        noise_std = 0.003
        sine_waves = mx.sin(phase) * sine_amp
        noise_amp = uv * noise_std + (1.0 - uv) * (sine_amp / 3.0)
        sine = sine_waves * uv + noise_amp * noise
        har = mx.tanh(self.source_linear(sine))
        return har.astype(self.compute_dtype)

    def __call__(self, z: mx.array, f0: mx.array, g: mx.array, source_noise: mx.array) -> mx.array:
        har_source = self._sine_source(f0, source_noise)
        x = self.conv_pre(z)
        if self.cond is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, negative_slope=0.1)
            x = self.ups[i](x)
            x_src = self.noise_convs[i](har_source)
            x = x + x_src
            xs = self.resblocks[i * self.num_kernels](x)
            for j in range(1, self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs * (1.0 / self.num_kernels)

        x = nn.leaky_relu(x, negative_slope=0.01)
        x = self.conv_post(x)
        return mx.tanh(x.astype(mx.float32))[:, :, 0]
    
        
    
    
    
class RVCSynthesizerMLX(nn.Module):
    def __init__(self, ckpt: RVCCheckpoint, use_fp16: bool = True):
        super().__init__()
        self.sr = ckpt.sample_rate
        self.version = ckpt.version
        self.f0 = bool(ckpt.f0)
        self.compute_dtype = mx.float16 if use_fp16 else mx.float32
        dtype = self.compute_dtype

        cfg = ckpt.config
        inter_ch = int(cfg[2])
        hidden_ch = int(cfg[3])
        filter_ch = int(cfg[4])
        n_heads = int(cfg[5])
        n_layers = int(cfg[6])
        k_sz = int(cfg[7])
        rk_sizes = [int(x) for x in cfg[10]]
        rd_sizes = [[int(y) for y in x] for x in cfg[11]]
        up_rates = [int(x) for x in cfg[12]]
        up_init_ch = int(cfg[13])
        up_k_sizes = [int(x) for x in cfg[14]]
        spk_embed_dim = int(cfg[15])
        gin_ch = int(cfg[16])

        phone_dim = 768 if self.version == "v2" else 256
        self.upp = int(np.prod(up_rates))

        self.enc_p = _TextEncoder(
            in_channels=phone_dim,
            out_channels=inter_ch,
            hidden_channels=hidden_ch,
            filter_channels=filter_ch,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=k_sz,
            f0=self.f0,
            dtype=dtype,
        )
        self.flows = [
            _ResidualCouplingLayer(inter_ch, hidden_ch, gin_ch) for _ in range(4)
        ]
        self.emb_g = nn.Embedding(spk_embed_dim, gin_ch)
        self.dec = _GeneratorNSF(
            initial_channel=inter_ch,
            resblock_kernel_sizes=rk_sizes,
            resblock_dilation_sizes=rd_sizes,
            upsample_rates=up_rates,
            upsample_initial_channel=up_init_ch,
            upsample_kernel_sizes=up_k_sizes,
            gin_channels=gin_ch,
            sr=self.sr,
            dtype=dtype,
        )

        self._load_weights(ckpt.weights)
        mx.eval(self.parameters())
        self.eval()

    def _load_weights(self, w: Dict[str, np.ndarray]) -> None:
        dtype = self.compute_dtype

        # enc_p
        self.enc_p.emb_phone.weight = mx.array(w["enc_p.emb_phone.weight"]).astype(dtype)
        self.enc_p.emb_phone.bias = mx.array(w["enc_p.emb_phone.bias"]).astype(dtype)
        if self.f0 and self.enc_p.emb_pitch is not None:
            self.enc_p.emb_pitch.weight = mx.array(w["enc_p.emb_pitch.weight"]).astype(dtype)

        for i in range(len(self.enc_p.attn_layers)):
            ap = f"enc_p.encoder.attn_layers.{i}"
            attn = self.enc_p.attn_layers[i]
            attn.emb_rel_k = mx.array(w[f"{ap}.emb_rel_k"]).astype(dtype)
            attn.emb_rel_v = mx.array(w[f"{ap}.emb_rel_v"]).astype(dtype)
            for name in ("conv_q", "conv_k", "conv_v", "conv_o"):
                conv = getattr(attn, name)
                conv.weight = _pt_conv1d_to_mlx(w[f"{ap}.{name}.weight"], dtype)
                conv.bias = mx.array(w[f"{ap}.{name}.bias"]).astype(dtype)

            n1 = self.enc_p.norm_layers_1[i]
            n1.gamma = mx.array(w[f"enc_p.encoder.norm_layers_1.{i}.gamma"]).astype(mx.float32)
            n1.beta = mx.array(w[f"enc_p.encoder.norm_layers_1.{i}.beta"]).astype(mx.float32)

            fp = f"enc_p.encoder.ffn_layers.{i}"
            ffn = self.enc_p.ffn_layers[i]
            ffn.conv_1.weight = _pt_conv1d_to_mlx(w[f"{fp}.conv_1.weight"], dtype)
            ffn.conv_1.bias = mx.array(w[f"{fp}.conv_1.bias"]).astype(dtype)
            ffn.conv_2.weight = _pt_conv1d_to_mlx(w[f"{fp}.conv_2.weight"], dtype)
            ffn.conv_2.bias = mx.array(w[f"{fp}.conv_2.bias"]).astype(dtype)

            n2 = self.enc_p.norm_layers_2[i]
            n2.gamma = mx.array(w[f"enc_p.encoder.norm_layers_2.{i}.gamma"]).astype(mx.float32)
            n2.beta = mx.array(w[f"enc_p.encoder.norm_layers_2.{i}.beta"]).astype(mx.float32)

        self.enc_p.proj.weight = _pt_conv1d_to_mlx(w["enc_p.proj.weight"], dtype)
        self.enc_p.proj.bias = mx.array(w["enc_p.proj.bias"]).astype(dtype)

        self.emb_g.weight = mx.array(w["emb_g.weight"]).astype(dtype)

        # flow
        for idx, fl in enumerate(self.flows):
            fp = f"flow.flows.{idx * 2}"
            fl.pre.weight = _pt_conv1d_to_mlx(w[f"{fp}.pre.weight"], dtype)
            fl.pre.bias = mx.array(w[f"{fp}.pre.bias"]).astype(dtype)
            fl.post.weight = _pt_conv1d_to_mlx(w[f"{fp}.post.weight"], dtype)
            fl.post.bias = mx.array(w[f"{fp}.post.bias"]).astype(dtype)

            fl.enc.cond_layer.weight = _pt_conv1d_to_mlx(_bake_wn(w, f"{fp}.enc.cond_layer"), dtype)
            fl.enc.cond_layer.bias = mx.array(w[f"{fp}.enc.cond_layer.bias"]).astype(dtype)
            for j in range(fl.enc.n_layers):
                fl.enc.in_layers[j].weight = _pt_conv1d_to_mlx(
                    _bake_wn(w, f"{fp}.enc.in_layers.{j}"), dtype
                )
                fl.enc.in_layers[j].bias = mx.array(w[f"{fp}.enc.in_layers.{j}.bias"]).astype(dtype)
                fl.enc.res_skip_layers[j].weight = _pt_conv1d_to_mlx(
                    _bake_wn(w, f"{fp}.enc.res_skip_layers.{j}"), dtype
                )
                fl.enc.res_skip_layers[j].bias = mx.array(
                    w[f"{fp}.enc.res_skip_layers.{j}.bias"]
                ).astype(dtype)

        # dec (keep source_linear in fp32)
        self.dec.source_linear.weight = mx.array(w["dec.m_source.l_linear.weight"]).astype(mx.float32)
        self.dec.source_linear.bias = mx.array(w["dec.m_source.l_linear.bias"]).astype(mx.float32)

        self.dec.conv_pre.weight = _pt_conv1d_to_mlx(w["dec.conv_pre.weight"], dtype)
        self.dec.conv_pre.bias = mx.array(w["dec.conv_pre.bias"]).astype(dtype)
        if self.dec.cond is not None and "dec.cond.weight" in w:
            self.dec.cond.weight = _pt_conv1d_to_mlx(w["dec.cond.weight"], dtype)
            self.dec.cond.bias = mx.array(w["dec.cond.bias"]).astype(dtype)

        for i in range(self.dec.num_upsamples):
            self.dec.ups[i].weight = _pt_convt1d_to_mlx(_bake_wn(w, f"dec.ups.{i}"), dtype)
            self.dec.ups[i].bias = mx.array(w[f"dec.ups.{i}.bias"]).astype(dtype)
            self.dec.noise_convs[i].weight = _pt_conv1d_to_mlx(w[f"dec.noise_convs.{i}.weight"], dtype)
            self.dec.noise_convs[i].bias = mx.array(w[f"dec.noise_convs.{i}.bias"]).astype(dtype)

        for r_idx, rb in enumerate(self.dec.resblocks):
            rp = f"dec.resblocks.{r_idx}"
            for k in range(3):
                rb.convs1[k].weight = _pt_conv1d_to_mlx(_bake_wn(w, f"{rp}.convs1.{k}"), dtype)
                rb.convs1[k].bias = mx.array(w[f"{rp}.convs1.{k}.bias"]).astype(dtype)
                rb.convs2[k].weight = _pt_conv1d_to_mlx(_bake_wn(w, f"{rp}.convs2.{k}"), dtype)
                rb.convs2[k].bias = mx.array(w[f"{rp}.convs2.{k}.bias"]).astype(dtype)

        self.dec.conv_post.weight = _pt_conv1d_to_mlx(w["dec.conv_post.weight"], dtype)

    def _enc_and_flow(
        self,
        phone: mx.array,
        pitch: mx.array,
        sid: mx.array,
        z_noise: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        g = self.emb_g(sid)[:, None, :]
        m_p, logs_p = self.enc_p(phone.astype(self.compute_dtype), pitch)
        z_p = m_p + mx.exp(logs_p.astype(mx.float32)).astype(self.compute_dtype) * z_noise

        x = z_p
        for fl in reversed(self.flows):
            x = x[:, :, ::-1]
            x = fl.reverse(x, g)
        return x, g

    def _decode_cropped(
        self,
        z: mx.array,
        f0: mx.array,
        g: mx.array,
        source_noise: mx.array,
    ) -> mx.array:
        return self.dec(z, f0, g, source_noise)

    def infer(
        self,
        phone: mx.array,
        pitch: mx.array,
        pitchf: mx.array,
        sid: int = 0,
        return_length: Optional[int] = None,
        hifi_margin: int = 6,
        noise_scale: float = 0.40,
    ) -> np.ndarray:
        T = phone.shape[1]
        sid_arr = mx.array([sid], dtype=mx.int32)
        z_noise = (mx.random.normal((1, T, self.enc_p.out_channels)) * noise_scale).astype(
            self.compute_dtype
        )
        z, g = self._enc_and_flow(phone, pitch, sid_arr, z_noise)
        
        # tail-crop before hifigan so we only decode the frames we actually emit
        trim_samples = 0
        if return_length is not None and return_length < T:
            margin = min(max(0, hifi_margin), T - return_length)
            dec_frames = return_length + margin
            z = z[:, -dec_frames:, :]
            pitchf = pitchf[:, -dec_frames:]
            trim_samples = margin * self.upp

        n_samples = z.shape[1] * self.upp
        source_noise = mx.random.normal((1, n_samples, 1), dtype=mx.float32)
        audio = self._decode_cropped(z, pitchf, g, source_noise)
        mx.eval(audio)
        wav = np.asarray(audio[0], dtype=np.float32)
        if trim_samples > 0:
            wav = wav[trim_samples:]
        return wav
