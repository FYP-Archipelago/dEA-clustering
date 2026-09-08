from __future__ import annotations

import numpy as np

from .. import genome as genome_mod
from ..config import Config
from ..reader import RunDirectory, RunMetadata
from ..types import FeatureSet
from .adapters import build_adapter


def build_features(
    run_dir: RunDirectory, metadata: RunMetadata, config: Config
) -> tuple[FeatureSet, list[str]]:
    """Vectorise every evaluation. Returns the features and the row ids, in order."""
    encoding = metadata.genome_encoding
    adapter = build_adapter(
        encoding, metadata.benchmark_params, None, config.seed
    )

    vectors: list[np.ndarray] = []
    sets: list[np.ndarray] = []
    hashes: list[str] = []
    row_ids: list[str] = []
    mask: list[bool] = []
    times: list[float] = []
    generations: list[int] = []
    islands: list[int] = []
    offsets = {i: metadata.clock_offset_seconds(i) for i in metadata.islands}
    width = 0

    for ev in run_dir.iter_evaluations():
        hashes.append(ev.genome_hash)
        row_ids.append(ev.individual_id)
        times.append(ev.t_wall + offsets.get(ev.island_id, 0.0))
        generations.append(ev.generation if ev.generation is not None else 0)
        islands.append(ev.island_id if ev.island_id is not None else -1)
        decoded = genome_mod.decode(
            ev.genome_repr, ev.genome_repr_mode, ev.genome_encoding
        )
        if decoded is None:
            vectors.append(None)  # type: ignore[arg-type]
            sets.append(np.empty(0, dtype=np.uint64))
            mask.append(False)
            continue
        vector, ids = adapter.encode(decoded)
        width = max(width, vector.size)
        vectors.append(vector)
        sets.append(ids if ids is not None else np.empty(0, dtype=np.uint64))
        mask.append(True)

    matrix = np.zeros((len(vectors), width), dtype=np.float64)
    for i, vector in enumerate(vectors):
        if vector is not None:
            matrix[i, : vector.size] = vector

    return (
        FeatureSet(
            matrix=matrix,
            metric=adapter.metric,
            hashes=hashes,
            mask=np.asarray(mask, dtype=bool),
            sets=sets if adapter.metric.value == "jaccard" else None,
            times=np.asarray(times, dtype=np.float64),
            generations=np.asarray(generations, dtype=np.int64),
            islands=np.asarray(islands, dtype=np.int64),
        ),
        row_ids,
    )
