from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from safetensors.numpy import save_file

SCHEMA_VERSION = 2


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while piece := stream.read(chunk):
            digest.update(piece)
    return digest.hexdigest()


@dataclass(frozen=True)
class CaptureManifest:
    model: str
    presentation: str
    dataset: str
    dataset_sha256: str
    shard: int
    shards: int
    layers: int
    hidden_size: int
    pairs: int
    buckets: int
    dtype: str
    skip_tokens: int
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not 0 <= self.shard < self.shards:
            raise ValueError(f"invalid shard {self.shard}/{self.shards}")
        if min(self.hidden_size, self.pairs, self.buckets) <= 0:
            raise ValueError("manifest dimensions must be positive")

    @property
    def merge_key(self) -> str:
        payload = {key: value for key, value in asdict(self).items() if key != "shard"}
        return sha256(canonical_json(payload).encode()).hexdigest()

    def metadata(self, summary: dict[str, Any]) -> dict[str, str]:
        return {"manifest": canonical_json({**asdict(self), "merge_key": self.merge_key, "summary": summary})}


def atomic_safetensors(tensors: dict[str, Any], destination: Path, metadata: dict[str, str]) -> None:
    """Write beside the destination, fsync, then atomically publish."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    try:
        save_file(tensors, str(partial), metadata=metadata)
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def read_capture(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a capture and return its tensors in canonical form (full f32 sums, square moments)."""
    from safetensors import safe_open

    with safe_open(str(path), framework="np") as handle:
        metadata = handle.metadata()
        if not metadata or "manifest" not in metadata:
            raise ValueError(f"{path} has no capture manifest")
        manifest = json.loads(metadata["manifest"])
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    ##
    # NOTE: a --minimal capture has no moments_tri but still stores fp16-packed means;
    # gating on moments alone would return them count-scaled and silently wrong.
    ##
    if "moments_tri" in tensors or tensors["sums"].dtype == "float16":
        tensors = unpack_capture(tensors)
    return manifest, tensors


MEAN16 = {
    "sums": "counts",
    "unit_diff_sum": "row_counts",
    "genre_sum": "genre_count",
    "language_sum": "language_count",
    "generator_sum": "generator_count",
}


def _denominator(name: str, tensors: "dict[str, Any]") -> Any:
    import numpy as np

    if MEAN16[name] == "row_counts":
        return np.maximum(tensors["counts"][:, 0, :], 1)[..., None]
    return np.maximum(tensors[MEAN16[name]], 1)[..., None]


def pack_capture(tensors: dict[str, Any], *, mean16: bool = True) -> dict[str, Any]:
    """Canonical accumulators -> compact storage: triangle-packed f32 moments, optionally fp16 means.
    mean16=False for near-null contrasts (syntax), where fp16 mean noise dominates the diff."""
    import numpy as np

    packed = dict(tensors)
    moments = packed.pop("moments", None)
    if moments is not None:
        rows, columns = np.triu_indices(moments.shape[-1])
        packed["moments_tri"] = moments[..., rows, columns].astype(np.float32)
    if mean16:
        for name in MEAN16:
            if name in packed:
                packed[name] = (packed[name] / _denominator(name, tensors)).astype(np.float16)
    return packed


def unpack_capture(packed: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    tensors = dict(packed)
    triangle = tensors.pop("moments_tri", None)
    if triangle is not None:
        hidden = int((np.sqrt(8 * triangle.shape[-1] + 1) - 1) / 2)
        moments = np.zeros((*triangle.shape[:-1], hidden, hidden), dtype=np.float64)
        rows, columns = np.triu_indices(hidden)
        moments[..., rows, columns] = triangle
        moments[..., columns, rows] = triangle
        tensors["moments"] = moments
    for name in MEAN16:
        if name in tensors and tensors[name].dtype == np.float16:
            tensors[name] = tensors[name].astype(np.float32) * _denominator(name, tensors).astype(np.float32)
    return tensors
