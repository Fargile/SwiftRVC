# beatrice v2 streaming engine (calls libbeatrice_bridge.dylib via ctypes)

from __future__ import annotations

import ctypes
import math
import os
import time
import tomllib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.signal import resample_poly

from .dsp import InputVoiceConditioner, OutputVoicePolisher
from .streaming import BaseVoiceEngine, IO_SAMPLE_RATE, INTERNAL_SR, get_rss_mb


@dataclass(frozen=True)
class BeatricePreset:
    name: str
    block_ms: int
    vq_neighbors: int


BEATRICE_PRESETS: Dict[str, BeatricePreset] = {
    "ultra_performance": BeatricePreset("ultra_performance", block_ms=20, vq_neighbors=0),
    "performance": BeatricePreset("performance", block_ms=30, vq_neighbors=1),
    "balanced": BeatricePreset("balanced", block_ms=40, vq_neighbors=2),
    "quality": BeatricePreset("quality", block_ms=60, vq_neighbors=4),
    "ultra_quality": BeatricePreset("ultra_quality", block_ms=80, vq_neighbors=6),
}


def _midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def _parse_version_int(ver: str) -> int:
    v = ver.lower().strip()
    if "rc" in v:
        return 2
    if "beta" in v or "20b" in v:
        return 1
    if "alpha" in v or "20a" in v:
        return 0
    return 2
    
        
    
    
def inspect_beatrice_toml(toml_path: str) -> Dict[str, Any]:
    toml_path = os.path.abspath(toml_path)
    model_dir = os.path.dirname(toml_path)
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    model_tbl = data.get("model", {})
    version = str(model_tbl.get("version", "2.0.0-rc.0"))
    name = str(model_tbl.get("name", os.path.basename(model_dir)))
    
    voices_tbl = data.get("voice", {})
    speakers: List[Dict[str, Any]] = []
    for k in sorted(voices_tbl.keys(), key=lambda s: int(s) if str(s).isdigit() else 9999):
        if not str(k).isdigit():
            continue
        idx = int(k)
        vt = voices_tbl[k]
        avg_midi = float(vt.get("average_pitch", 55.0))
        avg_hz = _midi_to_hz(avg_midi)
        vname = str(vt.get("name", f"speaker_{idx}"))
        tag = "High" if avg_hz >= 210 else ("Mid" if avg_hz >= 165 else "Low")
        speakers.append({
            "id": idx,
            "name": vname,
            "label": f"{vname} (~{int(round(avg_hz))}Hz · {tag})",
            "average_pitch_midi": round(avg_midi, 2),
            "average_pitch_hz": round(avg_hz, 1),
        })

    return {
        "engine": "beatrice",
        "name": name,
        "path": toml_path,
        "dir": model_dir,
        "version": version,
        "version_int": _parse_version_int(version),
        "sample_rate": 24000,
        "speakers": speakers,
        "num_speakers": len(speakers),
    }


def _load_bridge_lib(dylib_path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(dylib_path)
    lib.br_create.restype = ctypes.c_void_p
    lib.br_create.argtypes = []
    lib.br_destroy.restype = None
    lib.br_destroy.argtypes = [ctypes.c_void_p]
    lib.br_load.restype = ctypes.c_int
    lib.br_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.br_set_speaker.restype = None
    lib.br_set_speaker.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.br_set_params.restype = None
    lib.br_set_params.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    ]
    lib.br_reset.restype = None
    lib.br_reset.argtypes = [ctypes.c_void_p]
    lib.br_process_hops.restype = None
    lib.br_process_hops.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    return lib
    
    
        
    
    

class StreamingBeatriceEngine(BaseVoiceEngine):
    def __init__(self, dylib_path: str, preset_name: str = "balanced"):
        self.dylib_path = os.path.abspath(dylib_path)
        self._lib = _load_bridge_lib(self.dylib_path)
        self._ctx = self._lib.br_create()
        self.preset: BeatricePreset = BEATRICE_PRESETS.get(preset_name, BEATRICE_PRESETS["balanced"])

        self.meta: Optional[Dict[str, Any]] = None
        self.model_path: Optional[str] = None
        self.speaker_id: int = 0
        self.n_speakers: int = 0
        
        self.pitch_shift: float = 0.0
        self.formant_shift: float = 0.0
        self.pitch_floor_hz: float = 0.0
        self.intonation: float = 1.0
        self.gate_db: float = -45.0
        self.voice_polish: bool = True
        self.rms_mix: float = 0.25
        self.output_gain: float = 1.0
        self.bypass: bool = False

        self._in_dsp = InputVoiceConditioner(sr=INTERNAL_SR)
        self._out_dsp = OutputVoicePolisher(sr=IO_SAMPLE_RATE)
        self._gate_gain: float = 0.0
        self._silent_blocks: int = 999
        self._hangover_blocks: int = 7

        self.last_infer_ms: float = 0.0
        self.ema_infer_ms: float = 0.0
        self.last_gated: bool = True
        self.last_input_rms_db: float = -90.0

    @property
    def block_samples_48k(self) -> int:
        return (IO_SAMPLE_RATE * self.preset.block_ms) // 1000

    @property
    def algorithm_latency_ms(self) -> float:
        # 37.5ms core latency + one block
        return float(self.preset.block_ms + 37.5)

    def _sync_native_params(self) -> None:
        if not self._ctx:
            return
        self._lib.br_set_params(
            self._ctx,
            float(self.pitch_shift),
            float(max(-2.0, min(2.0, self.formant_shift))),
            float(self.intonation),
            int(self.preset.vq_neighbors),
            48.0,  # ~130hz avg src
            34.0,  # ~58hz min src for deep voices
            84.0,  # ~1046hz max src
            float(self.pitch_floor_hz),
        )

    def load_model(self, toml_path: str, speaker_id: int = 0) -> Dict[str, Any]:
        meta = inspect_beatrice_toml(toml_path)
        n_spk = self._lib.br_load(
            self._ctx,
            meta["dir"].encode("utf-8"),
            int(meta["version_int"]),
        )
        if n_spk <= 0:
            raise RuntimeError(f"libbeatrice failed to load {toml_path} (err={n_spk})")
        
        self.meta = meta
        self.model_path = meta["path"]
        self.n_speakers = int(n_spk)
        self.speaker_id = max(0, min(int(speaker_id), self.n_speakers - 1))
        self._lib.br_set_speaker(self._ctx, self.speaker_id)
        self._sync_native_params()
        self.reset_state()
        return self.model_info()

    def set_speaker(self, speaker_id: int) -> Dict[str, Any]:
        if self.n_speakers > 0:
            self.speaker_id = max(0, min(int(speaker_id), self.n_speakers - 1))
            self._lib.br_set_speaker(self._ctx, self.speaker_id)
        return self.model_info()

    def set_preset(self, preset_name: str) -> Dict[str, Any]:
        if preset_name not in BEATRICE_PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}")
        self.preset = BEATRICE_PRESETS[preset_name]
        self._hangover_blocks = max(3, int(round(280.0 / self.preset.block_ms)))
        self._sync_native_params()
        return self.model_info()

    def apply_params(self) -> None:
        self._sync_native_params()

    def reset_state(self) -> None:
        if self._ctx:
            self._lib.br_reset(self._ctx)
        self._in_dsp.reset()
        self._out_dsp.reset()
        self._gate_gain = 0.0
        self._hangover_blocks = max(3, int(round(280.0 / self.preset.block_ms)))
        self._silent_blocks = self._hangover_blocks + 1
        self.ema_infer_ms = 0.0

    def model_info(self) -> Dict[str, Any]:
        return {
            "engine": "beatrice",
            "preset": self.preset.name,
            "model_name": self.meta["name"] if self.meta else None,
            "model_path": self.model_path,
            "model_sr": 24000,
            "model_version": self.meta["version"] if self.meta else "2.0.0-rc.0",
            "speaker_id": self.speaker_id,
            "speakers": self.meta["speakers"] if self.meta else [],
            "has_index": False,
            "index_loaded": False,
            "index_ram_mb": 0.0,
            "index_rate": 0.0,
            "block_ms": self.preset.block_ms,
            "lookahead_ms": 0,
            "block_samples_48k": self.block_samples_48k,
            "algo_latency_ms": self.algorithm_latency_ms,
            "f0_method": f"beatrice (VQ={self.preset.vq_neighbors})",
            "rss_mb": round(get_rss_mb(), 1),
        }
    
        
    
    def process_block_48k(self, block_48k: np.ndarray) -> np.ndarray:
        block = np.asarray(block_48k, dtype=np.float32).reshape(-1)
        block_len = len(block)

        if self.bypass or self.meta is None:
            self.last_infer_ms = 0.0
            self.last_gated = False
            return block

        in_rms = float(np.sqrt(np.mean(block * block) + 1e-12))
        in_db = 20.0 * math.log10(max(in_rms, 1e-6))
        self.last_input_rms_db = in_db

        below_gate = in_db < self.gate_db
        if below_gate:
            self._silent_blocks += 1
        else:
            self._silent_blocks = 0

        if self._silent_blocks > self._hangover_blocks:
            self.last_infer_ms = 0.0
            self.last_gated = True
            self._gate_gain = 0.0
            return np.zeros(block_len, dtype=np.float32)

        self.last_gated = False
        t0 = time.perf_counter()
        
        # 48k -> 16k
        x16 = resample_poly(block, 1, 3).astype(np.float32)
        n_hops = max(1, len(x16) // 160)
        need_16k = n_hops * 160
        if len(x16) < need_16k:
            x16 = np.pad(x16, (0, need_16k - len(x16)))
        elif len(x16) > need_16k:
            x16 = x16[:need_16k]

        x16 = self._in_dsp.process(x16, polish_enabled=self.voice_polish, gate_db=self.gate_db)
        x16 = np.ascontiguousarray(x16, dtype=np.float32)
        y24 = np.empty(n_hops * 240, dtype=np.float32)

        self._lib.br_process_hops(
            self._ctx,
            x16.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y24.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_hops,
        )

        # 24k -> 48k
        out = resample_poly(y24, 2, 1).astype(np.float32)
        if len(out) > block_len:
            out = out[:block_len]
        elif len(out) < block_len:
            out = np.pad(out, (0, block_len - len(out)))

        if self.rms_mix > 0.0:
            out_rms = float(np.sqrt(np.mean(out * out) + 1e-12))
            if out_rms > 1e-5:
                scale = min(max((in_rms / out_rms) ** self.rms_mix, 0.25), 2.5)
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
        self.ema_infer_ms = infer_ms if self.ema_infer_ms == 0.0 else (0.8 * self.ema_infer_ms + 0.2 * infer_ms)
        return out

    def close(self) -> None:
        if getattr(self, "_ctx", None):
            self._lib.br_destroy(self._ctx)
            self._ctx = None

    def __del__(self) -> None:
        self.close()
