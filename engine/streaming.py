# streaming rvc v2 pipeline on mlx

from __future__ import annotations

import gc
import math
import os
import resource
import time
from dataclasses import dataclass
from typing import Dict, Optional

import mlx.core as mx
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from .contentvec import ContentVecMLX
from .dsp import (
    InputVoiceConditioner,
    OutputVoicePolisher,
    resample_exact_length,
    stretch_features_for_formant,
)
from .faiss_mmap import MMapFaissIndex
from .pitch import FastYINPitch, RMVPEMLX, condition_f0, f0_to_coarse
from .pth_loader import load_rvc_pth
from .rvc_v2 import RVCSynthesizerMLX


IO_SAMPLE_RATE = 48000
INTERNAL_SR = 16000


@dataclass(frozen=True)
class PresetConfig:
    name: str
    block_ms: int
    extra_ms: int
    lookahead_ms: int
    sola_crossfade_ms: int
    sola_search_ms: int
    hifi_margin_frames: int
    f0_method: str
    use_f0_cache: bool
    use_fp16: bool
    quantize_cv_8bit: bool
    allow_index: bool
    default_index_rate: float
    nprobe: int


PRESETS: Dict[str, PresetConfig] = {
    "ultra_performance": PresetConfig(
        name="ultra_performance",
        block_ms=80,
        extra_ms=200,
        lookahead_ms=10,
        sola_crossfade_ms=16,
        sola_search_ms=12,
        hifi_margin_frames=4,
        f0_method="yin",
        use_f0_cache=True,
        use_fp16=True,
        quantize_cv_8bit=True,
        allow_index=False,
        default_index_rate=0.0,
        nprobe=1,
    ),
    "performance": PresetConfig(
        name="performance",
        block_ms=100,
        extra_ms=320,
        lookahead_ms=20,
        sola_crossfade_ms=24,
        sola_search_ms=15,
        hifi_margin_frames=6,
        f0_method="yin",
        use_f0_cache=True,
        use_fp16=True,
        quantize_cv_8bit=True,
        allow_index=False,
        default_index_rate=0.0,
        nprobe=1,
    ),
    "balanced": PresetConfig(
        name="balanced",
        block_ms=160,
        extra_ms=600,
        lookahead_ms=35,
        sola_crossfade_ms=32,
        sola_search_ms=16,
        hifi_margin_frames=6,
        f0_method="rmvpe_fp16",
        use_f0_cache=True,
        use_fp16=True,
        quantize_cv_8bit=False,
        allow_index=True,
        default_index_rate=0.25,
        nprobe=1,
    ),
    "quality": PresetConfig(
        name="quality",
        block_ms=240,
        extra_ms=900,
        lookahead_ms=55,
        sola_crossfade_ms=40,
        sola_search_ms=16,
        hifi_margin_frames=8,
        f0_method="rmvpe_fp16",
        use_f0_cache=True,
        use_fp16=True,
        quantize_cv_8bit=False,
        allow_index=True,
        default_index_rate=0.40,
        nprobe=1,
    ),
    "ultra_quality": PresetConfig(
        name="ultra_quality",
        block_ms=300,
        extra_ms=1400,
        lookahead_ms=80,
        sola_crossfade_ms=50,
        sola_search_ms=18,
        hifi_margin_frames=10,
        f0_method="rmvpe_fp32",
        use_f0_cache=False,
        use_fp16=True,
        quantize_cv_8bit=False,
        allow_index=True,
        default_index_rate=0.45,
        nprobe=2,
    ),
}
    
        
    
    
# reads macos phys_footprint so ram matches activity monitor
def get_rss_mb() -> float:
    try:
        import ctypes

        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")

        class _TaskVMInfo(ctypes.Structure):
            _fields_ = [
                ("virtual_size", ctypes.c_uint64),
                ("region_count", ctypes.c_int32),
                ("page_size", ctypes.c_int32),
                ("resident_size", ctypes.c_uint64),
                ("resident_size_peak", ctypes.c_uint64),
                ("device", ctypes.c_uint64),
                ("device_peak", ctypes.c_uint64),
                ("internal", ctypes.c_uint64),
                ("internal_peak", ctypes.c_uint64),
                ("external", ctypes.c_uint64),
                ("external_peak", ctypes.c_uint64),
                ("reusable", ctypes.c_uint64),
                ("reusable_peak", ctypes.c_uint64),
                ("purgeable_volatile_pmap", ctypes.c_uint64),
                ("purgeable_volatile_resident", ctypes.c_uint64),
                ("purgeable_volatile_virtual", ctypes.c_uint64),
                ("compressed", ctypes.c_uint64),
                ("compressed_peak", ctypes.c_uint64),
                ("compressed_lifetime", ctypes.c_uint64),
                ("phys_footprint", ctypes.c_uint64),
            ]

        TASK_VM_INFO = 22
        info = _TaskVMInfo()
        count = ctypes.c_uint32(ctypes.sizeof(info) // 4)
        libc.mach_task_self.restype = ctypes.c_uint32
        kr = libc.task_info(libc.mach_task_self(), TASK_VM_INFO, ctypes.byref(info), ctypes.byref(count))
        if kr == 0 and info.phys_footprint > 0:
            return info.phys_footprint / (1024.0 * 1024.0)
    except Exception:
        pass
    return (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + mx.get_active_memory()
    ) / (1024.0 * 1024.0)


def _resample_to_48k(wav: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == 48000:
        return wav
    if src_sr == 40000:
        return resample_poly(wav, 6, 5).astype(np.float32)
    if src_sr == 32000:
        return resample_poly(wav, 3, 2).astype(np.float32)
    g = math.gcd(48000, src_sr)
    return resample_poly(wav, 48000 // g, src_sr // g).astype(np.float32)


class BaseVoiceEngine:
    @property
    def block_samples_48k(self) -> int:
        raise NotImplementedError

    def process_block_48k(self, block_48k: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset_state(self) -> None:
        raise NotImplementedError
    
    
        
    
    

class StreamingRVCEngine(BaseVoiceEngine):
    def __init__(self, base_models_dir: str, preset_name: str = "balanced"):
        # keep mlx metal cache small for 8gb macs
        mx.set_cache_limit(64 * 1024 * 1024)

        self.base_models_dir = os.path.abspath(base_models_dir)
        self.contentvec_path = os.path.join(self.base_models_dir, "contentvec_base.safetensors")
        self.rmvpe_path = os.path.join(self.base_models_dir, "rmvpe.safetensors")

        self.preset: PresetConfig = PRESETS[preset_name]
        self.contentvec: Optional[ContentVecMLX] = None
        self.rmvpe: Optional[RMVPEMLX] = None
        self.yin = FastYINPitch(sr=INTERNAL_SR, hop_length=160)
        self.rvc: Optional[RVCSynthesizerMLX] = None
        self.faiss_index: Optional[MMapFaissIndex] = None

        self.model_path: Optional[str] = None
        self.index_path: Optional[str] = None
        self._loaded_fp16: Optional[bool] = None
        self._loaded_cv_8bit: Optional[bool] = None
        self._loaded_rmvpe_fp16: Optional[bool] = None
        
        self.pitch_shift: float = 0.0
        self.formant_shift: float = 0.0
        self.pitch_floor_hz: float = 0.0
        self.gate_db: float = -45.0
        self.index_rate: float = self.preset.default_index_rate
        self.protect: float = 0.32
        self.noise_scale: float = 0.35
        self.voice_polish: bool = True
        self.rms_mix: float = 0.35
        self.output_gain: float = 1.0
        self.bypass: bool = False

        self._in_dsp = InputVoiceConditioner(sr=INTERNAL_SR)
        self._out_dsp = OutputVoicePolisher(sr=IO_SAMPLE_RATE)
        # 120hz hpf for contentvec only (rmvpe still gets full bass)
        self._cv_debass_sos = butter(2, 120.0, btype="highpass", fs=INTERNAL_SR, output="sos")
        self._gate_gain: float = 0.0

        self.last_infer_ms: float = 0.0
        self.ema_infer_ms: float = 0.0
        self.last_gated: bool = True
        self.last_input_rms_db: float = -90.0

        self._in_buf_16k = np.zeros(1, dtype=np.float32)
        self._f0_cache_base = np.zeros(1, dtype=np.float32)
        self._f0_cache_valid = False
        self._sola_tail_48k = np.zeros(1, dtype=np.float32)
        self._fade_in = np.zeros(1, dtype=np.float32)
        self._fade_out = np.zeros(1, dtype=np.float32)
        self._hangover_blocks = 0
        self._silent_blocks = 999
        self._configure_buffers()

    @property
    def block_samples_48k(self) -> int:
        return (IO_SAMPLE_RATE * self.preset.block_ms) // 1000

    @property
    def algorithm_latency_ms(self) -> float:
        return float(
            self.preset.block_ms
            + self.preset.lookahead_ms
            + self.preset.sola_crossfade_ms // 2
        )

    def _configure_buffers(self) -> None:
        p = self.preset
        self.block_16k = (INTERNAL_SR * p.block_ms) // 1000
        self.extra_16k = (INTERNAL_SR * p.extra_ms) // 1000
        self.lookahead_16k = (INTERNAL_SR * p.lookahead_ms) // 1000
        self.sola_cf_16k = (INTERNAL_SR * p.sola_crossfade_ms) // 1000
        self.sola_sr_16k = (INTERNAL_SR * p.sola_search_ms) // 1000

        total_16k = (
            self.extra_16k + self.sola_cf_16k + self.sola_sr_16k + self.block_16k + self.lookahead_16k
        )
        total_16k = ((total_16k + 319) // 320) * 320
        self.window_16k = total_16k
        self.window_frames_100fps = total_16k // 160 + 1
        self.block_frames_100fps = self.block_16k // 160

        # snap rmvpe tail to 32-frame grid
        f0_tail_frames = self.block_frames_100fps + max(16, (p.lookahead_ms // 10) + 12)
        f0_tail_frames = ((f0_tail_frames + 31) // 32) * 32 - 1
        self.f0_tail_samples_16k = min(self.window_16k, f0_tail_frames * 160)

        needed_16k = self.sola_cf_16k + self.sola_sr_16k + self.block_16k + self.lookahead_16k
        self.return_frames = (needed_16k + 159) // 160
        self.lookahead_samples_48k = (IO_SAMPLE_RATE * p.lookahead_ms) // 1000

        self.sola_cf_48k = (IO_SAMPLE_RATE * p.sola_crossfade_ms) // 1000
        self.sola_sr_48k = (IO_SAMPLE_RATE * p.sola_search_ms) // 1000

        t = np.linspace(0.0, 0.5 * np.pi, self.sola_cf_48k, dtype=np.float32)
        self._fade_in = np.sin(t) ** 2
        self._fade_out = np.cos(t) ** 2

        self._hangover_blocks = max(2, int(round(280.0 / p.block_ms)))
        self.reset_state()

    def reset_state(self) -> None:
        self._in_buf_16k = np.zeros(self.window_16k, dtype=np.float32)
        self._f0_cache_base = np.zeros(self.window_frames_100fps, dtype=np.float32)
        self._f0_cache_valid = False
        self._sola_tail_48k = np.zeros(self.sola_cf_48k, dtype=np.float32)
        self._in_dsp.reset()
        self._out_dsp.reset()
        self._gate_gain = 0.0
        self._silent_blocks = self._hangover_blocks + 1
        self.ema_infer_ms = 0.0

    def _ensure_base_models(self) -> None:
        want_fp16 = self.preset.use_fp16
        want_8bit = self.preset.quantize_cv_8bit
        if (
            self.contentvec is None
            or self._loaded_fp16 != want_fp16
            or self._loaded_cv_8bit != want_8bit
        ):
            self.contentvec = None
            gc.collect()
            mx.clear_cache()
            self.contentvec = ContentVecMLX.from_safetensors(
                self.contentvec_path, use_fp16=want_fp16, quantize_8bit=want_8bit
            )
            self._loaded_fp16 = want_fp16
            self._loaded_cv_8bit = want_8bit

        if self.preset.f0_method.startswith("rmvpe"):
            rmvpe_fp16 = self.preset.f0_method == "rmvpe_fp16"
            if self.rmvpe is None or self._loaded_rmvpe_fp16 != rmvpe_fp16:
                self.rmvpe = None
                gc.collect()
                mx.clear_cache()
                self.rmvpe = RMVPEMLX.from_safetensors(self.rmvpe_path, use_fp16=rmvpe_fp16)
                self._loaded_rmvpe_fp16 = rmvpe_fp16
        else:
            # drop rmvpe in yin presets to free ~185mb
            if self.rmvpe is not None:
                self.rmvpe = None
                self._loaded_rmvpe_fp16 = None
                gc.collect()
                mx.clear_cache()
    
        
    
    
    def load_model(self, pth_path: str, index_path: Optional[str] = None) -> Dict:
        self._ensure_base_models()

        if self.faiss_index is not None:
            self.faiss_index.close()
            self.faiss_index = None

        self.rvc = None
        gc.collect()
        mx.clear_cache()

        ckpt = load_rvc_pth(pth_path)
        self.rvc = RVCSynthesizerMLX(ckpt, use_fp16=self.preset.use_fp16)
        self.model_path = ckpt.path

        if index_path is None:
            model_dir = os.path.dirname(ckpt.path)
            candidates = [
                os.path.join(model_dir, f)
                for f in sorted(os.listdir(model_dir))
                if f.endswith(".index")
            ]
            if candidates:
                index_path = candidates[0]

        self.index_path = index_path
        if index_path and os.path.isfile(index_path) and self.preset.allow_index:
            self.faiss_index = MMapFaissIndex(index_path, nprobe=self.preset.nprobe)
            if self.index_rate == 0.0 and self.preset.default_index_rate > 0.0:
                self.index_rate = self.preset.default_index_rate

        del ckpt
        gc.collect()
        mx.clear_cache()
        self.reset_state()
        self.warmup()
        mx.clear_cache()
        return self.model_info()

    def set_preset(self, preset_name: str) -> Dict:
        if preset_name not in PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}")
        if preset_name == self.preset.name and self.contentvec is not None:
            return self.model_info()

        prev_fp16 = self.preset.use_fp16
        self.preset = PRESETS[preset_name]
        self._configure_buffers()
        self._ensure_base_models()

        if self.model_path is not None and (self.rvc is None or prev_fp16 != self.preset.use_fp16):
            self.rvc = None
            gc.collect()
            mx.clear_cache()
            ckpt = load_rvc_pth(self.model_path)
            self.rvc = RVCSynthesizerMLX(ckpt, use_fp16=self.preset.use_fp16)
            del ckpt

        if not self.preset.allow_index:
            if self.faiss_index is not None:
                self.faiss_index.close()
                self.faiss_index = None
            self.index_rate = 0.0
        else:
            if self.faiss_index is None and self.index_path and os.path.isfile(self.index_path):
                self.faiss_index = MMapFaissIndex(self.index_path, nprobe=self.preset.nprobe)
            elif self.faiss_index is not None:
                self.faiss_index.nprobe = self.preset.nprobe
            if self.faiss_index is not None:
                self.index_rate = self.preset.default_index_rate

        gc.collect()
        mx.clear_cache()
        self.warmup()
        mx.clear_cache()
        return self.model_info()

    def warmup(self) -> None:
        if self.rvc is None or self.contentvec is None:
            return
        dummy = np.zeros(self.window_16k, dtype=np.float32)
        self._infer_window(dummy, return_frames=self.return_frames, streaming=True)
        self.reset_state()

    def unload_all(self) -> None:
        if self.faiss_index is not None:
            self.faiss_index.close()
            self.faiss_index = None
        self.contentvec = None
        self.rmvpe = None
        self.rvc = None
        self._loaded_fp16 = None
        self._loaded_cv_8bit = None
        self._loaded_rmvpe_fp16 = None
        gc.collect()
        mx.clear_cache()

    def model_info(self) -> Dict:
        return {
            "engine": "rvc",
            "preset": self.preset.name,
            "model_name": os.path.splitext(os.path.basename(self.model_path))[0] if self.model_path else None,
            "model_path": self.model_path,
            "model_sr": self.rvc.sr if self.rvc else None,
            "model_version": self.rvc.version if self.rvc else None,
            "has_index": bool(self.index_path and os.path.isfile(self.index_path)),
            "index_loaded": self.faiss_index is not None,
            "index_ram_mb": round(self.faiss_index.ram_mb, 1) if self.faiss_index else 0.0,
            "index_rate": round(self.index_rate, 2),
            "block_ms": self.preset.block_ms,
            "lookahead_ms": self.preset.lookahead_ms,
            "block_samples_48k": self.block_samples_48k,
            "algo_latency_ms": self.algorithm_latency_ms,
            "f0_method": self.preset.f0_method,
            "rss_mb": round(get_rss_mb(), 1),
        }
    
    
        
    
    
    def _extract_f0(self, audio_16k: np.ndarray, streaming: bool = False) -> np.ndarray:
        use_cache = streaming and self.preset.use_f0_cache
        if not use_cache:
            if self.preset.f0_method == "yin" or self.rmvpe is None:
                f0_base = self.yin.infer(audio_16k)
            else:
                f0_base = self.rmvpe.infer(audio_16k)
        else:
            if not self._f0_cache_valid or self.f0_tail_samples_16k >= len(audio_16k):
                if self.preset.f0_method == "yin" or self.rmvpe is None:
                    f0_base = self.yin.infer(audio_16k)
                else:
                    f0_base = self.rmvpe.infer(audio_16k)
                n_copy = min(len(f0_base), len(self._f0_cache_base))
                self._f0_cache_base.fill(0.0)
                self._f0_cache_base[-n_copy:] = f0_base[-n_copy:]
                self._f0_cache_valid = True
            else:
                bf = self.block_frames_100fps
                self._f0_cache_base = np.roll(self._f0_cache_base, -bf)
                tail_audio = audio_16k[-self.f0_tail_samples_16k :]
                if self.preset.f0_method == "yin" or self.rmvpe is None:
                    tail_f0 = self.yin.infer(tail_audio)
                else:
                    tail_f0 = self.rmvpe.infer(tail_audio)
                update_frames = min(len(tail_f0), bf + 6)
                self._f0_cache_base[-update_frames:] = tail_f0[-update_frames:]
                f0_base = self._f0_cache_base

        f0 = f0_base.copy()
        if self.pitch_shift != 0.0:
            f0 = (f0 * (2.0 ** (self.pitch_shift / 12.0))).astype(np.float32)

        return condition_f0(
            f0,
            smooth=self.voice_polish,
            f0_min=55.0,
            f0_max=1050.0,
            pitch_floor_hz=self.pitch_floor_hz,
        )

    def _infer_window(
        self,
        audio_16k: np.ndarray,
        return_frames: Optional[int] = None,
        streaming: bool = False,
    ) -> np.ndarray:
        cv_audio = audio_16k
        if self.voice_polish and len(audio_16k) > 32:
            cv_audio = sosfiltfilt(self._cv_debass_sos, audio_16k).astype(np.float32)

        pad_audio = np.pad(cv_audio, (200, 200), mode="reflect" if len(cv_audio) > 200 else "edge")
        cv_in = mx.array(pad_audio[None, :], dtype=mx.float32)
        out_layer = 12 if self.rvc.version == "v2" else 9
        feats_50 = self.contentvec(cv_in, output_layer=out_layer)

        # overlap gpu contentvec with cpu yin
        if self.preset.f0_method == "yin" or self.rmvpe is None:
            mx.async_eval(feats_50)

        f0 = self._extract_f0(audio_16k, streaming=streaming)

        feats_100 = mx.repeat(feats_50, 2, axis=1)
        T = min(int(feats_100.shape[1]), int(f0.shape[0]))
        feats_100 = feats_100[:, :T, :]
        f0 = f0[:T]

        need_np_feats = (
            (self.faiss_index is not None and self.preset.allow_index and self.index_rate > 0.001)
            or abs(self.formant_shift) >= 0.05
        )

        alpha_formant = 1.0
        eff_return_frames = return_frames

        if need_np_feats:
            mx.eval(feats_100)
            feats_np = np.asarray(feats_100[0], dtype=np.float32).copy()

            if (
                self.faiss_index is not None
                and self.preset.allow_index
                and self.index_rate > 0.001
            ):
                orig_feats = feats_np.copy()
                if return_frames is not None and return_frames + 16 < T:
                    start = T - (return_frames + 16)
                    feats_np[start:] = self.faiss_index.blend_features(
                        feats_np[start:], index_rate=self.index_rate
                    )
                else:
                    feats_np = self.faiss_index.blend_features(feats_np, index_rate=self.index_rate)

                # consonant protect on unvoiced frames
                if self.protect > 0.001 and self.protect < 0.5:
                    unvoiced = (f0 <= 1.0).astype(np.float32)
                    if len(unvoiced) >= 3:
                        unvoiced = np.convolve(unvoiced, [0.25, 0.5, 0.25], mode="same").astype(np.float32)
                    p_mix = np.clip(unvoiced * (self.protect * 2.0), 0.0, 1.0)[:, None]
                    feats_np = feats_np * (1.0 - p_mix) + orig_feats * p_mix
                elif self.protect >= 0.5:
                    unvoiced = (f0 <= 1.0)[:, None]
                    feats_np = np.where(unvoiced, orig_feats, feats_np)

            if abs(self.formant_shift) >= 0.05:
                feats_np, f0, alpha_formant = stretch_features_for_formant(
                    feats_np, f0, self.formant_shift
                )
                if return_frames is not None:
                    eff_return_frames = max(4, int(round(return_frames * alpha_formant)))

            feats_100 = mx.array(feats_np[None, :], dtype=self.rvc.compute_dtype)

        pitch_coarse = f0_to_coarse(f0)
        pitch_mx = mx.array(pitch_coarse[None, :], dtype=mx.int32)
        pitchf_mx = mx.array(f0[None, :], dtype=mx.float32)

        wav_model_sr = self.rvc.infer(
            phone=feats_100,
            pitch=pitch_mx,
            pitchf=pitchf_mx,
            sid=0,
            return_length=eff_return_frames,
            hifi_margin=self.preset.hifi_margin_frames,
            noise_scale=self.noise_scale,
        )

        if abs(alpha_formant - 1.0) >= 1e-4:
            if return_frames is not None:
                expected_48k = return_frames * 480
            else:
                expected_48k = int(round(len(wav_model_sr) * (IO_SAMPLE_RATE / float(self.rvc.sr)) / alpha_formant))
            return resample_exact_length(wav_model_sr, expected_48k)

        return _resample_to_48k(wav_model_sr, self.rvc.sr)

    def _sola_align_and_mix(self, synth_48k: np.ndarray, block_len: int) -> np.ndarray:
        cf = self.sola_cf_48k
        sr = self.sola_sr_48k
        need = cf + sr + block_len
        if len(synth_48k) < need:
            synth_48k = np.pad(synth_48k, (need - len(synth_48k), 0))
        elif len(synth_48k) > need:
            synth_48k = synth_48k[-need:]

        prev = self._sola_tail_48k
        prev_energy = float(np.dot(prev, prev))

        best_off = 0
        if prev_energy > 1e-8 and sr > 0:
            search_seg = synth_48k[: cf + sr]
            cor = np.correlate(search_seg, prev, mode="valid")
            sq = search_seg * search_seg
            csq = np.concatenate(([0.0], np.cumsum(sq, dtype=np.float64)))
            win_energy = csq[cf : cf + sr + 1] - csq[: sr + 1]
            norm_cor = cor / np.sqrt(np.maximum(win_energy, 1e-10))
            best_off = int(np.argmax(norm_cor))

        aligned = synth_48k[best_off : best_off + cf + block_len].copy()
        aligned[:cf] = prev * self._fade_out + aligned[:cf] * self._fade_in
        self._sola_tail_48k = aligned[block_len : block_len + cf].copy()
        return aligned[:block_len]

    def process_block_48k(self, block_48k: np.ndarray) -> np.ndarray:
        block = np.asarray(block_48k, dtype=np.float32).reshape(-1)
        block_len = len(block)

        if self.bypass or self.rvc is None:
            self.last_infer_ms = 0.0
            self.last_gated = False
            return block

        in_rms = float(np.sqrt(np.mean(block * block) + 1e-12))
        in_db = 20.0 * math.log10(max(in_rms, 1e-6))
        self.last_input_rms_db = in_db

        block_16k = resample_poly(block, 1, 3).astype(np.float32)
        if len(block_16k) != self.block_16k:
            if len(block_16k) > self.block_16k:
                block_16k = block_16k[: self.block_16k]
            else:
                block_16k = np.pad(block_16k, (0, self.block_16k - len(block_16k)))
        block_16k = self._in_dsp.process(
            block_16k, polish_enabled=self.voice_polish, gate_db=self.gate_db
        )

        self._in_buf_16k = np.roll(self._in_buf_16k, -self.block_16k)
        self._in_buf_16k[-self.block_16k :] = block_16k

        below_gate = in_db < self.gate_db
        if below_gate:
            self._silent_blocks += 1
        else:
            self._silent_blocks = 0

        if self._silent_blocks > self._hangover_blocks:
            self.last_infer_ms = 0.0
            self.last_gated = True
            self._f0_cache_valid = False
            self._gate_gain = 0.0
            if np.any(self._sola_tail_48k):
                out = np.zeros(block_len, dtype=np.float32)
                cf = min(self.sola_cf_48k, block_len)
                out[:cf] = self._sola_tail_48k[:cf] * self._fade_out[:cf]
                self._sola_tail_48k.fill(0.0)
                return out
            return np.zeros(block_len, dtype=np.float32)

        self.last_gated = False
        t0 = time.perf_counter()

        wav_48k = self._infer_window(
            self._in_buf_16k, return_frames=self.return_frames, streaming=True
        )

        if self.lookahead_samples_48k > 0 and len(wav_48k) > self.lookahead_samples_48k:
            wav_48k = wav_48k[: -self.lookahead_samples_48k]

        out = self._sola_align_and_mix(wav_48k, block_len)

        if self.rms_mix > 0.0:
            out_rms = float(np.sqrt(np.mean(out * out) + 1e-12))
            if out_rms > 1e-5:
                scale = (in_rms / out_rms) ** self.rms_mix
                scale = min(max(scale, 0.22), 2.5)
                out = out * scale

        target_gate = 1.0 if not below_gate else max(0.0, 1.0 - (self._silent_blocks / (self._hangover_blocks + 1.0)))
        if self.voice_polish:
            prev_gg = self._gate_gain
            next_gg = target_gate if target_gate > prev_gg else (prev_gg * 0.4 + target_gate * 0.6)
            self._gate_gain = next_gg
            if abs(prev_gg - next_gg) > 1e-3 or next_gg < 0.999:
                out = out * np.linspace(prev_gg, next_gg, block_len, dtype=np.float32)

        if self.output_gain != 1.0:
            out = out * self.output_gain

        out = self._out_dsp.process(out, enabled=self.voice_polish)

        infer_ms = (time.perf_counter() - t0) * 1000.0
        self.last_infer_ms = infer_ms
        self.ema_infer_ms = (
            infer_ms if self.ema_infer_ms == 0.0 else (0.8 * self.ema_infer_ms + 0.2 * infer_ms)
        )
        return out

    def convert_offline_48k(self, wav_48k: np.ndarray) -> np.ndarray:
        if self.rvc is None:
            raise RuntimeError("No RVC model loaded")
        wav_16k = resample_poly(np.asarray(wav_48k, dtype=np.float32), 1, 3).astype(np.float32)
        self._in_dsp.reset()
        self._out_dsp.reset()
        wav_16k = self._in_dsp.process(wav_16k, polish_enabled=self.voice_polish, gate_db=-60.0)
        wav_out = self._infer_window(wav_16k, return_frames=None, streaming=False)
        return self._out_dsp.process(wav_out, enabled=self.voice_polish)
