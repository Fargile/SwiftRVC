# ipc worker for swiftrvc (json stdin/stdout + unix socket 48khz f32 pcm)

from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import struct
import sys
import threading
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.beatrice_engine import StreamingBeatriceEngine, inspect_beatrice_toml
from engine.pth_loader import inspect_rvc_pth
from engine.streaming import PRESETS, StreamingRVCEngine, get_rss_mb


_stdout_lock = threading.Lock()


def emit(event: Dict[str, Any]) -> None:
    line = json.dumps(event, separators=(",", ":"))
    with _stdout_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def _recv_exact(conn: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)
    
        
    
    
class WorkerDaemon:
    def __init__(
        self,
        base_models_dir: str,
        models_dir: str,
        beatrice_models_dir: str,
        beatrice_dylib: str,
        sock_path: str,
        preset: str,
    ):
        self.base_models_dir = os.path.abspath(base_models_dir)
        self.models_dir = os.path.abspath(models_dir)
        self.beatrice_models_dir = os.path.abspath(beatrice_models_dir)
        self.beatrice_dylib = os.path.abspath(beatrice_dylib)
        self.sock_path = sock_path
        self.preset_name = preset

        self.active_engine_name: str = "rvc"
        self.rvc_engine: Optional[StreamingRVCEngine] = StreamingRVCEngine(
            base_models_dir=self.base_models_dir, preset_name=preset
        )
        self.beatrice_engine: Optional[StreamingBeatriceEngine] = None

        self.engine_lock = threading.Lock()
        self.running = True
        self._server_sock: Optional[socket.socket] = None
        self._stats_counter = 0

    @property
    def active_engine(self):
        if self.active_engine_name == "beatrice":
            return self.beatrice_engine
        return self.rvc_engine

    def scan_rvc_models(self) -> List[Dict[str, Any]]:
        found = []
        if not os.path.isdir(self.models_dir):
            return found
        for root, _dirs, files in os.walk(self.models_dir):
            pth_files = sorted(f for f in files if f.endswith(".pth"))
            idx_files = sorted(f for f in files if f.endswith(".index"))
            for pf in pth_files:
                full_pth = os.path.join(root, pf)
                try:
                    meta = inspect_rvc_pth(full_pth)
                    meta["engine"] = "rvc"
                    meta["index_path"] = os.path.join(root, idx_files[0]) if idx_files else None
                    found.append(meta)
                except Exception as exc:
                    found.append({
                        "engine": "rvc",
                        "name": os.path.splitext(pf)[0],
                        "path": full_pth,
                        "error": str(exc),
                    })
        found.sort(key=lambda m: m.get("name", "").lower())
        return found

    def scan_beatrice_models(self) -> List[Dict[str, Any]]:
        found = []
        if not os.path.isdir(self.beatrice_models_dir):
            return found
        for root, _dirs, files in os.walk(self.beatrice_models_dir):
            toml_files = sorted(f for f in files if f.endswith(".toml"))
            has_bins = "phone_extractor.bin" in files and "waveform_generator.bin" in files
            if not has_bins:
                continue
            for tf in toml_files:
                full_toml = os.path.join(root, tf)
                try:
                    meta = inspect_beatrice_toml(full_toml)
                    found.append(meta)
                except Exception as exc:
                    found.append({
                        "engine": "beatrice",
                        "name": os.path.splitext(tf)[0],
                        "path": full_toml,
                        "error": str(exc),
                    })
        found.sort(key=lambda m: m.get("name", "").lower())
        return found

    def _switch_engine(self, target: str) -> Dict[str, Any]:
        if target not in ("rvc", "beatrice"):
            raise ValueError(f"Unknown engine: {target}")
        if target == self.active_engine_name and self.active_engine is not None:
            return self.active_engine.model_info()

        if target == "beatrice":
            if self.rvc_engine is not None:
                self.rvc_engine.unload_all()
            if not os.path.isfile(self.beatrice_dylib):
                raise FileNotFoundError(f"Beatrice library not found: {self.beatrice_dylib}")
            if self.beatrice_engine is None:
                self.beatrice_engine = StreamingBeatriceEngine(
                    dylib_path=self.beatrice_dylib, preset_name=self.preset_name
                )
            else:
                self.beatrice_engine.set_preset(self.preset_name)
            self.active_engine_name = "beatrice"
            gc.collect()
            return self.beatrice_engine.model_info()
        else:
            if self.beatrice_engine is not None:
                self.beatrice_engine.close()
                self.beatrice_engine = None
            if self.rvc_engine is None:
                self.rvc_engine = StreamingRVCEngine(
                    base_models_dir=self.base_models_dir, preset_name=self.preset_name
                )
            else:
                self.rvc_engine.set_preset(self.preset_name)
            self.active_engine_name = "rvc"
            gc.collect()
            return self.rvc_engine.model_info()
    
    
        
    
    def _audio_loop(self) -> None:
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.sock_path)
        srv.listen(2)
        srv.settimeout(0.5)
        self._server_sock = srv
        emit({"type": "audio_socket_ready", "socket": self.sock_path})

        while self.running:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                conn.settimeout(None)
                while self.running:
                    hdr = _recv_exact(conn, 4)
                    if hdr is None:
                        break
                    (n_bytes,) = struct.unpack("<I", hdr)
                    if n_bytes == 0:
                        continue
                    payload = _recv_exact(conn, n_bytes)
                    if payload is None:
                        break
                    in_arr = np.frombuffer(payload, dtype="<f4")
                    with self.engine_lock:
                        eng = self.active_engine
                        if eng is None:
                            out_arr = in_arr
                            infer_ms, ema_ms, gated, in_db, block_ms, algo_ms = 0.0, 0.0, True, -90.0, 160, 0.0
                        else:
                            out_arr = eng.process_block_48k(in_arr)
                            infer_ms = eng.last_infer_ms
                            ema_ms = eng.ema_infer_ms
                            gated = eng.last_gated
                            in_db = eng.last_input_rms_db
                            block_ms = eng.preset.block_ms
                            algo_ms = eng.algorithm_latency_ms

                    out_bytes = np.ascontiguousarray(out_arr, dtype="<f4").tobytes()
                    conn.sendall(struct.pack("<I", len(out_bytes)) + out_bytes)

                    self._stats_counter += 1
                    stride = max(1, int(round(300.0 / max(block_ms, 20))))
                    if self._stats_counter % stride == 0:
                        rtf = (ema_ms / block_ms) if block_ms > 0 else 0.0
                        emit({
                            "type": "stats",
                            "engine": self.active_engine_name,
                            "infer_ms": round(infer_ms, 1),
                            "ema_infer_ms": round(ema_ms, 1),
                            "total_latency_ms": round(algo_ms + ema_ms, 1),
                            "rtf": round(rtf, 3),
                            "gated": gated,
                            "input_db": round(in_db, 1),
                            "rss_mb": round(get_rss_mb(), 1),
                        })
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

        try:
            srv.close()
        except OSError:
            pass
        if os.path.exists(self.sock_path):
            try:
                os.unlink(self.sock_path)
            except OSError:
                pass
    
        
    
    
    def handle_command(self, cmd: Dict[str, Any]) -> None:
        op = cmd.get("cmd")
        req_id = cmd.get("id")

        try:
            if op == "ping":
                emit({"type": "reply", "id": req_id, "ok": True, "rss_mb": round(get_rss_mb(), 1)})
            elif op == "list_models":
                emit({
                    "type": "reply",
                    "id": req_id,
                    "ok": True,
                    "models": self.scan_rvc_models(),
                    "beatrice_models": self.scan_beatrice_models(),
                })
            elif op == "set_engine":
                target = cmd["engine"]
                with self.engine_lock:
                    info = self._switch_engine(target)
                emit({"type": "reply", "id": req_id, "ok": True, "info": info})
            elif op == "load_model":
                engine_type = cmd.get("engine", self.active_engine_name)
                path = cmd["path"]
                with self.engine_lock:
                    self._switch_engine(engine_type)
                    if engine_type == "beatrice":
                        spk = int(cmd.get("speaker_id", 0))
                        info = self.beatrice_engine.load_model(path, speaker_id=spk)
                    else:
                        idx = cmd.get("index_path")
                        info = self.rvc_engine.load_model(path, idx)
                emit({"type": "reply", "id": req_id, "ok": True, "info": info})
            elif op == "set_speaker":
                spk = int(cmd["speaker_id"])
                with self.engine_lock:
                    if self.beatrice_engine is not None and self.active_engine_name == "beatrice":
                        info = self.beatrice_engine.set_speaker(spk)
                    else:
                        info = self.active_engine.model_info() if self.active_engine else {}
                emit({"type": "reply", "id": req_id, "ok": True, "info": info})
            elif op == "set_preset":
                preset = cmd["preset"]
                self.preset_name = preset
                with self.engine_lock:
                    info = self.active_engine.set_preset(preset)
                emit({"type": "reply", "id": req_id, "ok": True, "info": info})
            elif op == "set_params":
                with self.engine_lock:
                    eng = self.active_engine
                    if eng is not None:
                        if "pitch_shift" in cmd:
                            eng.pitch_shift = float(cmd["pitch_shift"])
                        if "formant_shift" in cmd:
                            eng.formant_shift = float(cmd["formant_shift"])
                        if "pitch_floor_hz" in cmd:
                            eng.pitch_floor_hz = float(cmd["pitch_floor_hz"])
                        if "gate_db" in cmd:
                            eng.gate_db = float(cmd["gate_db"])
                        if "voice_polish" in cmd:
                            eng.voice_polish = bool(cmd["voice_polish"])
                        if "rms_mix" in cmd:
                            eng.rms_mix = float(cmd["rms_mix"])
                        if "output_gain" in cmd:
                            eng.output_gain = float(cmd["output_gain"])
                        if "bypass" in cmd:
                            eng.bypass = bool(cmd["bypass"])

                        if self.active_engine_name == "rvc":
                            if "index_rate" in cmd:
                                eng.index_rate = float(cmd["index_rate"])
                            if "protect" in cmd:
                                eng.protect = float(cmd["protect"])
                            if "noise_scale" in cmd:
                                eng.noise_scale = float(cmd["noise_scale"])
                        else:
                            if "intonation" in cmd:
                                eng.intonation = float(cmd["intonation"])
                            eng.apply_params()
                emit({"type": "reply", "id": req_id, "ok": True})
            elif op == "reset":
                with self.engine_lock:
                    if self.active_engine is not None:
                        self.active_engine.reset_state()
                emit({"type": "reply", "id": req_id, "ok": True})
            elif op == "quit":
                self.running = False
                emit({"type": "reply", "id": req_id, "ok": True})
            else:
                emit({"type": "reply", "id": req_id, "ok": False, "error": f"Unknown cmd: {op}"})
        except Exception as exc:
            emit({
                "type": "reply",
                "id": req_id,
                "ok": False,
                "error": str(exc),
                "trace": traceback.format_exc(limit=4),
            })

    def run(self) -> None:
        audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        audio_thread.start()
        emit({
            "type": "ready",
            "presets": list(PRESETS.keys()),
            "beatrice_available": os.path.isfile(self.beatrice_dylib),
            "info": self.rvc_engine.model_info() if self.rvc_engine else {},
            "models": self.scan_rvc_models(),
            "beatrice_models": self.scan_beatrice_models(),
        })

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
            except json.JSONDecodeError as exc:
                emit({"type": "error", "error": f"Invalid JSON: {exc}"})
                continue
            self.handle_command(cmd)
            if not self.running:
                break

        self.running = False
        if self.beatrice_engine is not None:
            self.beatrice_engine.close()
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass
        audio_thread.join(timeout=1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="SwiftRVC audio worker")
    parser.add_argument("--base-models", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--beatrice-models", default="")
    parser.add_argument("--beatrice-dylib", default="")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--preset", default="balanced", choices=list(PRESETS.keys()))
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    b_models = args.beatrice_models or os.path.join(root, "beatrice_models")
    b_dylib = args.beatrice_dylib or os.path.join(root, "beatrice", "libbeatrice_bridge.dylib")

    daemon = WorkerDaemon(
        base_models_dir=args.base_models,
        models_dir=args.models_dir,
        beatrice_models_dir=b_models,
        beatrice_dylib=b_dylib,
        sock_path=args.socket,
        preset=args.preset,
    )
    daemon.run()


if __name__ == "__main__":
    main()
