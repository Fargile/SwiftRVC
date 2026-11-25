# reads standard rvc .pth zip checkpoints without importing torch

from __future__ import annotations

import collections
import io
import os
import pickle
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


_STORAGE_DTYPE_MAP: Dict[str, Tuple[np.dtype, int]] = {
    "FloatStorage": (np.dtype("<f4"), 4),
    "HalfStorage": (np.dtype("<f2"), 2),
    "DoubleStorage": (np.dtype("<f8"), 8),
    "LongStorage": (np.dtype("<i8"), 8),
    "IntStorage": (np.dtype("<i4"), 4),
    "ShortStorage": (np.dtype("<i2"), 2),
    "ByteStorage": (np.dtype("<u1"), 1),
    "CharStorage": (np.dtype("<i1"), 1),
    "BoolStorage": (np.dtype("?"), 1),
    # bf16 comes in as u16 bits, convert to f32 later
    "BFloat16Storage": (np.dtype("<u2"), 2),
}


@dataclass
class _TensorRef:
    storage_type: str
    storage_key: str
    storage_size: int
    offset: int
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]


class _StorageRef:
    __slots__ = ("dtype_name", "key", "device", "size")

    def __init__(self, dtype_name: str, key: str, device: Any, size: int):
        self.dtype_name = dtype_name
        self.key = str(key)
        self.device = device
        self.size = int(size)
    
        
    

def _rebuild_tensor_v2(
    storage: _StorageRef,
    storage_offset: int,
    size: Tuple[int, ...],
    stride: Tuple[int, ...],
    requires_grad: bool = False,
    backward_hooks: Any = None,
    metadata: Any = None,
) -> _TensorRef:
    return _TensorRef(
        storage_type=storage.dtype_name,
        storage_key=storage.key,
        storage_size=storage.size,
        offset=int(storage_offset),
        shape=tuple(int(s) for s in size),
        stride=tuple(int(s) for s in stride),
    )


def _rebuild_tensor(
    storage: _StorageRef,
    storage_offset: int,
    size: Tuple[int, ...],
    stride: Tuple[int, ...],
) -> _TensorRef:
    return _rebuild_tensor_v2(storage, storage_offset, size, stride)


# restricted unpickler so we only accept tensor metadata + plain builtins
class _SafeTorchUnpickler(pickle.Unpickler):
    _SAFE_BUILTINS = {
        "slice": slice,
        "range": range,
        "set": set,
        "frozenset": frozenset,
        "tuple": tuple,
        "list": list,
        "dict": dict,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "bytes": bytes,
    }

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor_v2 if name == "_rebuild_tensor_v2" else _rebuild_tensor
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch" and name.endswith("Storage"):
            return name
        if module == "torch" and name in ("Size",):
            return tuple
        if module == "builtins" and name in self._SAFE_BUILTINS:
            return self._SAFE_BUILTINS[name]
        raise pickle.UnpicklingError(f"Disallowed pickle global: {module}.{name}")

    def persistent_load(self, pid: Any) -> Any:
        if isinstance(pid, tuple) and len(pid) >= 5 and pid[0] == "storage":
            return _StorageRef(pid[1], pid[2], pid[3], pid[4])
        raise pickle.UnpicklingError(f"Unsupported persistent_load pid: {pid!r}")


def _bfloat16_to_float32(raw_u16: np.ndarray) -> np.ndarray:
    u32 = raw_u16.astype(np.uint32) << 16
    return u32.view(np.float32)
    
    
        
    
    
def _materialize_tensor(
    zf: zipfile.ZipFile,
    prefix: str,
    ref: _TensorRef,
    storage_cache: Dict[str, bytes],
) -> np.ndarray:
    if ref.storage_type not in _STORAGE_DTYPE_MAP:
        raise ValueError(f"Unsupported torch storage type: {ref.storage_type}")
    np_dtype, itemsize = _STORAGE_DTYPE_MAP[ref.storage_type]
    entry = f"{prefix}data/{ref.storage_key}"
    raw = storage_cache.get(entry)
    if raw is None:
        raw = zf.read(entry)
        storage_cache[entry] = raw
    
    arr_1d = np.frombuffer(raw, dtype=np_dtype)
    if ref.storage_type == "BFloat16Storage":
        arr_1d = _bfloat16_to_float32(arr_1d)
        itemsize = 4

    if len(ref.shape) == 0:
        return np.array(arr_1d[ref.offset])

    # fast path when c-contiguous
    expected_stride = []
    acc = 1
    for dim in reversed(ref.shape):
        expected_stride.append(acc)
        acc *= dim
    expected_stride.reverse()
    total_elems = acc
    
    if tuple(expected_stride) == ref.stride:
        out = arr_1d[ref.offset : ref.offset + total_elems].reshape(ref.shape)
        return out.copy()

    # strided fallback
    byte_strides = tuple(s * itemsize for s in ref.stride)
    base = arr_1d[ref.offset :]
    view = np.lib.stride_tricks.as_strided(base, shape=ref.shape, strides=byte_strides)
    return np.ascontiguousarray(view)


@dataclass
class RVCCheckpoint:
    weights: Dict[str, np.ndarray]
    config: List[Any]
    sample_rate: int
    version: str
    f0: int
    info: str
    path: str


def _parse_sr(raw_sr: Any, config: List[Any]) -> int:
    if isinstance(raw_sr, int):
        return raw_sr
    if isinstance(raw_sr, str):
        s = raw_sr.lower().strip()
        if s.endswith("k"):
            return int(float(s[:-1]) * 1000)
        if s.isdigit():
            return int(s)
    if config and isinstance(config[-1], int) and config[-1] in (32000, 40000, 48000):
        return int(config[-1])
    return 40000
    
        
    
    
def load_rvc_pth(path: str) -> RVCCheckpoint:
    path = os.path.abspath(path)
    if not zipfile.is_zipfile(path):
        raise ValueError(
            f"{path} is not a modern PyTorch ZIP checkpoint (.pth). "
            "Legacy tar-based checkpoints are not supported."
        )
    
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        pkl_candidates = [n for n in names if n.endswith("data.pkl")]
        if not pkl_candidates:
            raise ValueError(f"No data.pkl found inside {path}")
        pkl_name = pkl_candidates[0]
        prefix = pkl_name[: -len("data.pkl")]

        with zf.open(pkl_name, "r") as pf:
            obj = _SafeTorchUnpickler(io.BytesIO(pf.read())).load()

        if not isinstance(obj, dict) or "weight" not in obj:
            raise ValueError(
                f"Checkpoint {os.path.basename(path)} does not look like an RVC inference checkpoint "
                "(missing 'weight' key)."
            )

        raw_weights = obj["weight"]
        storage_cache: Dict[str, bytes] = {}
        weights: Dict[str, np.ndarray] = {}
        for k, v in raw_weights.items():
            if isinstance(v, _TensorRef):
                weights[k] = _materialize_tensor(zf, prefix, v, storage_cache)
            else:
                weights[k] = np.asarray(v)
        storage_cache.clear()
    
    config = list(obj.get("config", []))
    sr = _parse_sr(obj.get("sr"), config)
    version = str(obj.get("version", "v1"))
    f0 = int(obj.get("f0", 1))
    info = str(obj.get("info", ""))

    # infer v2 vs v1 from emb_phone dim if version field is missing
    if "enc_p.emb_phone.weight" in weights:
        in_dim = weights["enc_p.emb_phone.weight"].shape[1]
        if in_dim == 768:
            version = "v2"
        elif in_dim == 256:
            version = "v1"

    return RVCCheckpoint(
        weights=weights,
        config=config,
        sample_rate=sr,
        version=version,
        f0=f0,
        info=info,
        path=path,
    )


def inspect_rvc_pth(path: str) -> Dict[str, Any]:
    path = os.path.abspath(path)
    with zipfile.ZipFile(path, "r") as zf:
        pkl_name = [n for n in zf.namelist() if n.endswith("data.pkl")][0]
        with zf.open(pkl_name, "r") as pf:
            obj = _SafeTorchUnpickler(io.BytesIO(pf.read())).load()
    
    config = list(obj.get("config", []))
    sr = _parse_sr(obj.get("sr"), config)
    w = obj.get("weight", {})
    version = str(obj.get("version", "v1"))
    if "enc_p.emb_phone.weight" in w and isinstance(w["enc_p.emb_phone.weight"], _TensorRef):
        in_dim = w["enc_p.emb_phone.weight"].shape[1]
        version = "v2" if in_dim == 768 else "v1"
    return {
        "name": os.path.splitext(os.path.basename(path))[0],
        "path": path,
        "sample_rate": sr,
        "version": version,
        "f0": int(obj.get("f0", 1)),
        "info": str(obj.get("info", "")),
        "num_tensors": len(w),
    }
