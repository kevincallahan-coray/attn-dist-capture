"""On-disk format for captured attention distributions.

Layout
------
    <out_dir>/
        manifest.json            run-level metadata (model, config, versions)
        index.jsonl              one flat JSON row per record, no arrays
        shard_00000.npz          arrays for the records in that shard
        shard_00001.npz
        ...

Every ``index.jsonl`` row carries ``shard``, ``record_id`` and summary stats
(entropy, top-k mass, nk, layer, head, decode step). That means you can select a
stratum -- "diffuse mid-layer distributions at nk > 3000" -- by reading only the
index, then touch just the shards you need.

Array keys inside a shard:
    r<record_id>/probs    float32 [nk]      post-softmax distribution, sums to 1
    r<record_id>/av       float32 [d]       exact attention output (ground truth)
    r<record_id>/scores   float32 [nk]      optional pre-softmax masked scores
    v/<e>/<s>/<l>/<kvh>   float16 [nk, d]   optional V rows, shared across a
                                            GQA group to avoid 4x duplication
"""

from __future__ import annotations

import json
import pathlib
from typing import Iterator

import numpy as np


class ShardWriter:
    def __init__(self, out_dir: str | pathlib.Path, shard_records: int = 512,
                 compress: bool = True) -> None:
        self.dir = pathlib.Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.shard_records = shard_records
        self.compress = compress
        self.index_path = self.dir / "index.jsonl"
        self._buf: dict[str, np.ndarray] = {}
        self._meta: list[dict] = []
        self._shard = 0
        self._next_id = 0
        self._v_seen: set[str] = set()
        self.bytes_written = 0

    def add(self, meta: dict, arrays: dict) -> int:
        rid = self._next_id
        self._next_id += 1
        meta = dict(meta)
        meta["record_id"] = rid
        meta["shard"] = f"shard_{self._shard:05d}.npz"

        shared = arrays.pop("__v_shared__", None)
        for name, arr in arrays.items():
            self._buf[f"r{rid}/{name}"] = arr
        if shared is not None:
            v_key, v_arr = shared
            # A GQA group shares V; only store it once per shard.
            if v_key not in self._v_seen:
                self._buf[v_key] = v_arr
                self._v_seen.add(v_key)
        self._meta.append(meta)

        if len(self._meta) >= self.shard_records:
            self.flush()
        return rid

    def has_v(self, v_key: str) -> bool:
        """True if this V matrix is already stored in the current shard."""
        return v_key in self._v_seen

    def flush(self) -> None:
        if not self._meta:
            return
        path = self.dir / f"shard_{self._shard:05d}.npz"
        saver = np.savez_compressed if self.compress else np.savez
        saver(path, **self._buf)
        self.bytes_written += path.stat().st_size
        with open(self.index_path, "a") as f:
            for m in self._meta:
                f.write(json.dumps(m) + "\n")
        self._buf, self._meta, self._v_seen = {}, [], set()
        self._shard += 1

    def write_manifest(self, manifest: dict) -> None:
        with open(self.dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    def close(self, manifest: dict | None = None) -> None:
        self.flush()
        if manifest is not None:
            self.write_manifest(manifest)


# --------------------------------------------------------------------- reading


class AttnDump:
    """Read-side helper. Lazily opens shards and caches the handles."""

    def __init__(self, out_dir: str | pathlib.Path) -> None:
        self.dir = pathlib.Path(out_dir)
        with open(self.dir / "index.jsonl") as f:
            self.index: list[dict] = [json.loads(line) for line in f if line.strip()]
        mpath = self.dir / "manifest.json"
        self.manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
        self._open: dict[str, np.lib.npyio.NpzFile] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _shard(self, name: str):
        if name not in self._open:
            self._open[name] = np.load(self.dir / name)
        return self._open[name]

    def probs(self, rec: dict) -> np.ndarray:
        return self._shard(rec["shard"])[f"r{rec['record_id']}/probs"]

    def av(self, rec: dict) -> np.ndarray:
        return self._shard(rec["shard"])[f"r{rec['record_id']}/av"]

    def scores(self, rec: dict) -> np.ndarray:
        return self._shard(rec["shard"])[f"r{rec['record_id']}/scores"]

    def values(self, rec: dict) -> np.ndarray | None:
        key = rec.get("v_key")
        if not key:
            return None
        return self._shard(rec["shard"])[key]

    def select(self, **conds) -> list[dict]:
        """Filter the index. Scalars match exactly; ``(lo, hi)`` tuples are
        inclusive ranges; callables are applied to the value."""
        out = []
        for rec in self.index:
            keep = True
            for field, cond in conds.items():
                val = rec.get(field)
                if callable(cond):
                    keep = keep and bool(cond(val))
                elif isinstance(cond, tuple) and len(cond) == 2:
                    keep = keep and (cond[0] <= val <= cond[1])
                else:
                    keep = keep and (val == cond)
                if not keep:
                    break
            if keep:
                out.append(rec)
        return out

    def iter_records(self, recs: list[dict] | None = None) -> Iterator[tuple[dict, np.ndarray]]:
        for rec in (recs if recs is not None else self.index):
            yield rec, self.probs(rec)
