# mmap faiss .index reader (ivf-flat + flat) so we only keep centroids in ram

from __future__ import annotations

import mmap
import os
import struct
from typing import Optional

import numpy as np


def _u32(buf: bytes, off: int):
    return struct.unpack_from("<I", buf, off)[0], off + 4


def _i32(buf: bytes, off: int):
    return struct.unpack_from("<i", buf, off)[0], off + 4


def _i64(buf: bytes, off: int):
    return struct.unpack_from("<q", buf, off)[0], off + 8


def _u64(buf: bytes, off: int):
    return struct.unpack_from("<Q", buf, off)[0], off + 8
    
    
        
    

class MMapFaissIndex:
    def __init__(self, path: str, nprobe: int = 1):
        self.path = os.path.abspath(path)
        self._fp = open(self.path, "rb")
        self._mm = mmap.mmap(self._fp.fileno(), 0, access=mmap.ACCESS_READ)
        self.is_ivf = False
        self.d = 0
        self.ntotal = 0
        self.nlist = 0
        self.nprobe = max(1, int(nprobe))
        self.centroids: Optional[np.ndarray] = None
        self.centroid_norms: Optional[np.ndarray] = None
        self._list_offsets: Optional[np.ndarray] = None
        self._list_sizes: Optional[np.ndarray] = None
        self._code_size = 0
        self._flat_offset = 0

        self._parse()

    def _parse(self) -> None:
        mm = self._mm
        magic = mm[0:4]
        off = 4

        self.d, off = _i32(mm, off)
        self.ntotal, off = _i64(mm, off)
        _dummy1, off = _i64(mm, off)
        _dummy2, off = _i64(mm, off)
        _is_trained = mm[off]
        off += 1
        _metric_type, off = _i32(mm, off)
        
        if magic in (b"IxF2", b"IxFI"):
            # flat index
            xb_len, off = _u64(mm, off)
            expected = self.ntotal * self.d
            if xb_len != expected:
                raise ValueError(f"IndexFlat length mismatch: {xb_len} vs {expected}")
            self.is_ivf = False
            self._flat_offset = off
            return

        if magic not in (b"IwFl", b"IxIF"):
            raise ValueError(
                f"Unsupported FAISS index type {magic!r} in {os.path.basename(self.path)}."
            )

        self.is_ivf = True
        self.nlist, off = _u64(mm, off)
        file_nprobe, off = _u64(mm, off)
        if self.nprobe <= 0:
            self.nprobe = max(1, int(file_nprobe))

        # coarse quantizer (indexflatl2)
        q_magic = mm[off : off + 4]
        off += 4
        if q_magic not in (b"IxF2", b"IxFI"):
            raise ValueError(f"Unsupported IVF quantizer magic: {q_magic!r}")
        q_d, off = _i32(mm, off)
        q_ntotal, off = _i64(mm, off)
        _qd1, off = _i64(mm, off)
        _qd2, off = _i64(mm, off)
        _qtr = mm[off]
        off += 1
        _qmt, off = _i32(mm, off)
        xb_len, off = _u64(mm, off)
        if q_d != self.d or xb_len != q_ntotal * q_d:
            raise ValueError("Corrupted IVF quantizer block.")
        
        # keep only centroids in ram (~6mb)
        c_bytes = xb_len * 4
        self.centroids = np.frombuffer(mm, dtype="<f4", count=xb_len, offset=off).reshape(
            q_ntotal, q_d
        ).copy()
        self.centroid_norms = np.sum(self.centroids * self.centroids, axis=1)
        off += c_bytes

        # skip direct map
        dm_type = mm[off]
        off += 1
        dm_len, off = _u64(mm, off)
        off += dm_len * 8
        if dm_type == 2:
            ht_len, off = _u64(mm, off)
            off += ht_len * 16

        # inverted lists header
        inv_magic = mm[off : off + 4]
        off += 4
        if inv_magic != b"ilar":
            raise ValueError(f"Unsupported inverted list format: {inv_magic!r}")
        inv_nlist, off = _u64(mm, off)
        inv_code_size, off = _u64(mm, off)
        list_type = mm[off : off + 4]
        off += 4
        self._code_size = int(inv_code_size)

        if list_type == b"full":
            sizes_len, off = _u64(mm, off)
            sizes = np.frombuffer(mm, dtype="<u8", count=sizes_len, offset=off).copy()
            off += sizes_len * 8
        elif list_type == b"sprs":
            pairs_len, off = _u64(mm, off)
            pairs = np.frombuffer(mm, dtype="<u8", count=pairs_len, offset=off).reshape(-1, 2)
            sizes = np.zeros(inv_nlist, dtype=np.uint64)
            sizes[pairs[:, 0]] = pairs[:, 1]
            off += pairs_len * 8
        else:
            raise ValueError(f"Unsupported inverted list type: {list_type!r}")

        # each list has s*code_size bytes of vectors followed by s*8 bytes of ids
        offsets = np.empty(inv_nlist, dtype=np.int64)
        cur = off
        stride_per_vec = self._code_size + 8
        for i in range(inv_nlist):
            offsets[i] = cur
            cur += int(sizes[i]) * stride_per_vec

        self._list_offsets = offsets
        self._list_sizes = sizes.astype(np.int64)
    
    
        
    
    def _get_list_vectors(self, list_id: int) -> np.ndarray:
        n = int(self._list_sizes[list_id])
        if n <= 0:
            return np.empty((0, self.d), dtype=np.float32)
        byte_off = int(self._list_offsets[list_id])
        return np.frombuffer(
            self._mm, dtype="<f4", count=n * self.d, offset=byte_off
        ).reshape(n, self.d)

    def blend_features(
        self,
        feats: np.ndarray,
        index_rate: float = 0.35,
        k: int = 8,
        nprobe: Optional[int] = None,
    ) -> np.ndarray:
        if index_rate <= 0.0 or self.ntotal == 0 or feats.shape[1] != self.d:
            return feats

        T = feats.shape[0]
        q = np.ascontiguousarray(feats, dtype=np.float32)
        q_norms = np.sum(q * q, axis=1, keepdims=True)
        
        if not self.is_ivf:
            xb = np.frombuffer(
                self._mm, dtype="<f4", count=self.ntotal * self.d, offset=self._flat_offset
            ).reshape(self.ntotal, self.d)
            xb_norms = np.sum(xb * xb, axis=1, keepdims=True).T
            dists = np.maximum(q_norms + xb_norms - 2.0 * (q @ xb.T), 1e-6)
            kk = min(k, self.ntotal)
            idx = np.argpartition(dists, kk - 1, axis=1)[:, :kk]
            rows = np.arange(T)[:, None]
            d_k = dists[rows, idx]
            w = 1.0 / (d_k * d_k)
            w /= np.sum(w, axis=1, keepdims=True)
            retrieved = np.einsum("tk,tkd->td", w, xb[idx])
            return (index_rate * retrieved + (1.0 - index_rate) * q).astype(np.float32)

        np_probe = max(1, int(nprobe if nprobe is not None else self.nprobe))
        c_dists = q_norms + self.centroid_norms[None, :] - 2.0 * (q @ self.centroids.T)
        if np_probe == 1:
            cell_ids = np.argmin(c_dists, axis=1)[:, None]
        else:
            cell_ids = np.argpartition(c_dists, np_probe - 1, axis=1)[:, :np_probe]
        
        retrieved = np.empty_like(q)
        # batch frames that land in the same cell
        if np_probe == 1:
            unique_cells, inv = np.unique(cell_ids[:, 0], return_inverse=True)
            for ui, cid in enumerate(unique_cells):
                frame_mask = np.where(inv == ui)[0]
                vecs = self._get_list_vectors(int(cid))
                if vecs.shape[0] == 0:
                    retrieved[frame_mask] = q[frame_mask]
                    continue
                sub_q = q[frame_mask]
                sub_qn = q_norms[frame_mask]
                vn = np.sum(vecs * vecs, axis=1)[None, :]
                dists = np.maximum(sub_qn + vn - 2.0 * (sub_q @ vecs.T), 1e-6)
                kk = min(k, vecs.shape[0])
                if kk < vecs.shape[0]:
                    top_i = np.argpartition(dists, kk - 1, axis=1)[:, :kk]
                    rows = np.arange(len(frame_mask))[:, None]
                    d_k = dists[rows, top_i]
                    v_k = vecs[top_i]
                else:
                    d_k = dists
                    v_k = np.broadcast_to(vecs[None, :, :], (len(frame_mask), vecs.shape[0], self.d))
                w = 1.0 / (d_k * d_k)
                w /= np.sum(w, axis=1, keepdims=True)
                retrieved[frame_mask] = np.einsum("tk,tkd->td", w, v_k)
        else:
            for t in range(T):
                chunks = [self._get_list_vectors(int(c)) for c in cell_ids[t]]
                chunks = [c for c in chunks if c.shape[0] > 0]
                if not chunks:
                    retrieved[t] = q[t]
                    continue
                vecs = np.concatenate(chunks, axis=0)
                diff = vecs - q[t]
                dists = np.maximum(np.sum(diff * diff, axis=1), 1e-6)
                kk = min(k, vecs.shape[0])
                top_i = np.argpartition(dists, kk - 1)[:kk] if kk < vecs.shape[0] else np.arange(kk)
                d_k = dists[top_i]
                w = 1.0 / (d_k * d_k)
                w /= np.sum(w)
                retrieved[t] = w @ vecs[top_i]

        return (index_rate * retrieved + (1.0 - index_rate) * q).astype(np.float32)

    @property
    def ram_mb(self) -> float:
        if self.centroids is None:
            return 0.0
        return (self.centroids.nbytes + self.centroid_norms.nbytes) / (1024 * 1024)

    def close(self) -> None:
        if getattr(self, "_mm", None) is not None:
            self._mm.close()
            self._mm = None
        if getattr(self, "_fp", None) is not None:
            self._fp.close()
            self._fp = None

    def __del__(self) -> None:
        self.close()
