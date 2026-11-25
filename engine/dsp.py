# pre/post voice dsp (leveler, eq, soft limiter, formant stretch)

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt, sosfilt_zi


def _peaking_biquad(fs: float, f0: float, q: float, gain_db: float) -> np.ndarray:
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / fs
    cw = math.cos(w0)
    sw = math.sin(w0)
    alpha = sw / (2.0 * q)
    
    b0 = 1.0 + alpha * a
    b1 = -2.0 * cw
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cw
    a2 = 1.0 - alpha / a
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def _high_shelf_biquad(fs: float, f0: float, slope: float, gain_db: float) -> np.ndarray:
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / fs
    cw = math.cos(w0)
    sw = math.sin(w0)
    alpha = (sw / 2.0) * math.sqrt((a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0)
    sq_a = 2.0 * math.sqrt(a) * alpha
    
    b0 = a * ((a + 1.0) + (a - 1.0) * cw + sq_a)
    b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cw)
    b2 = a * ((a + 1.0) + (a - 1.0) * cw - sq_a)
    a0 = (a + 1.0) - (a - 1.0) * cw + sq_a
    a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cw)
    a2 = (a + 1.0) - (a - 1.0) * cw - sq_a
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)
    
    
        
    

class InputVoiceConditioner:
    def __init__(self, sr: int = 16000):
        self.sr = sr
        self._hp_sos = butter(4, 48.0, btype="highpass", fs=sr, output="sos")
        self._hp_zi = sosfilt_zi(self._hp_sos) * 0.0
        self._env_gain = 1.0

    def reset(self) -> None:
        self._hp_zi = sosfilt_zi(self._hp_sos) * 0.0
        self._env_gain = 1.0

    def process(self, x_16k: np.ndarray, polish_enabled: bool = True, gate_db: float = -45.0) -> np.ndarray:
        y, self._hp_zi = sosfilt(self._hp_sos, x_16k, zi=self._hp_zi)
        y = y.astype(np.float32)
        if not polish_enabled:
            return y
        
        rms = float(np.sqrt(np.mean(y * y) + 1e-12))
        db = 20.0 * math.log10(max(rms, 1e-6))

        if db < gate_db + 3.0:
            # dont boost room noise near the gate floor
            target_gain = 1.0
        else:
            target_db = -16.5
            err_db = target_db - db
            if err_db > 0.0:
                adj_db = min(err_db * 0.55, 8.0)
            else:
                adj_db = max(err_db * 0.65, -10.0)
            target_gain = 10.0 ** (adj_db / 20.0)

        prev_g = self._env_gain
        coeff = 0.45 if target_gain < prev_g else 0.25
        next_g = prev_g + coeff * (target_gain - prev_g)
        self._env_gain = next_g

        ramp = np.linspace(prev_g, next_g, len(y), dtype=np.float32)
        y = y * ramp
        return np.tanh(y * 1.05).astype(np.float32) / 1.05


class OutputVoicePolisher:
    def __init__(self, sr: int = 48000):
        self.sr = sr
        hp = butter(2, 85.0, btype="highpass", fs=sr, output="sos")
        mud_cut = _peaking_biquad(sr, f0=310.0, q=1.15, gain_db=-2.0)
        presence = _peaking_biquad(sr, f0=3100.0, q=1.0, gain_db=1.2)
        de_harsh = _high_shelf_biquad(sr, f0=6800.0, slope=0.8, gain_db=-2.6)
        self._sos = np.vstack([hp, mud_cut, presence, de_harsh])
        self._zi = sosfilt_zi(self._sos) * 0.0

    def reset(self) -> None:
        self._zi = sosfilt_zi(self._sos) * 0.0

    def process(self, x_48k: np.ndarray, enabled: bool = True) -> np.ndarray:
        if not enabled:
            return np.clip(x_48k, -0.98, 0.98).astype(np.float32)

        y, self._zi = sosfilt(self._sos, x_48k, zi=self._zi)
        y = y.astype(np.float32)

        drive = 1.08
        y = np.tanh(y * drive) / math.tanh(drive) * 0.96
        return y.astype(np.float32)
    
        
    
    
    
def resample_exact_length(wav: np.ndarray, target_len: int) -> np.ndarray:
    n_in = len(wav)
    if n_in == target_len or n_in == 0 or target_len == 0:
        return wav.astype(np.float32)

    g = math.gcd(target_len, n_in)
    up = target_len // g
    down = n_in // g
    if up <= 64 and down <= 64:
        out = resample_poly(wav, up, down).astype(np.float32)
        if len(out) > target_len:
            return out[:target_len]
        if len(out) < target_len:
            return np.pad(out, (0, target_len - len(out)))
        return out
    
    # fallback to catmull-rom cubic for arbitrary formant ratios
    pos = np.linspace(0.0, float(n_in - 1), target_len, dtype=np.float64)
    i1 = np.floor(pos).astype(np.int64)
    t = (pos - i1).astype(np.float32)
    i0 = np.clip(i1 - 1, 0, n_in - 1)
    i2 = np.clip(i1 + 1, 0, n_in - 1)
    i3 = np.clip(i1 + 2, 0, n_in - 1)

    p0 = wav[i0]
    p1 = wav[i1]
    p2 = wav[i2]
    p3 = wav[i3]

    t2 = t * t
    t3 = t2 * t
    out = 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )
    return out.astype(np.float32)


def stretch_features_for_formant(
    feats_np: np.ndarray,
    f0_np: np.ndarray,
    formant_shift_st: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    # stretch feats/f0 by alpha and divide f0 by alpha, then resample audio back by 1/alpha
    if abs(formant_shift_st) < 0.05:
        return feats_np, f0_np, 1.0

    alpha = 2.0 ** (formant_shift_st / 12.0)
    T = feats_np.shape[0]
    T_new = max(4, int(round(T * alpha)))
    actual_alpha = T_new / float(T)

    idx_f = np.linspace(0.0, float(T - 1), T_new, dtype=np.float32)
    i0 = np.floor(idx_f).astype(np.int64)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (idx_f - i0)[:, None]
    feats_stretched = (feats_np[i0] * (1.0 - w) + feats_np[i1] * w).astype(np.float32)

    i_near = np.rint(idx_f).astype(np.int64)
    f0_stretched = (f0_np[i_near] / actual_alpha).astype(np.float32)
    return feats_stretched, f0_stretched, actual_alpha
