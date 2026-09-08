from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .reader import RunDirectory
from .pipeline import run_stages
from .stn.artifact import write_artifact
from .stn.builder import STN, STNBuilder


@dataclass
class Result:
    stn: STN
    out_dir: Path
    level0_nodes: int
    elapsed_seconds: float
    diagnostics: dict[str, Any]
    resolved: dict[str, Any]
    layout: dict[str, Any]
    baseline_stn: STN | None = None
    baseline_layout: dict[str, Any] | None = None


def _centroids(row_keys: list[str], features) -> dict[str, np.ndarray]:
    """Mean feature vector per node key."""
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for i, key in enumerate(row_keys):
        if not features.mask[i]:
            continue
        vector = features.matrix[i]
        if key in sums:
            sums[key] += vector
            counts[key] += 1
        else:
            sums[key] = vector.astype(np.float64, copy=True)
            counts[key] = 1
    return {key: sums[key] / counts[key] for key in sums}


def _attach_radius(stn: STN, centroids: dict[str, np.ndarray], features, row_keys) -> None:
    """RMS distance of a node's members from their centroid."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for i, key in enumerate(row_keys):
        centre = centroids.get(key)
        if centre is None or not features.mask[i]:
            continue
        sums[key] = sums.get(key, 0.0) + float(np.linalg.norm(features.matrix[i] - centre) ** 2)
        counts[key] = counts.get(key, 0) + 1
    for node in stn.nodes:
        if counts.get(node.key):
            node.radius = float(np.sqrt(sums[node.key] / counts[node.key]))


def run(
    run_path: str | Path,
    out_dir: str | Path,
    config: Config | None = None,
    *,
    write: bool = True,
    baseline: bool = False,
) -> Result:
    """Cluster one run. With baseline=True also build the unclustered graph."""
    started = time.perf_counter()
    config = config or Config()
    run_dir = RunDirectory(run_path)
    metadata = run_dir.read_metadata()

    needs_features = config.any_stage_enabled or config.layout.method != "none"
    keys: dict[str, str] | None = None
    row_keys: list[str] = []
    features = None
    node_vectors: dict[str, np.ndarray] = {}
    node_weights: dict[str, float] = {}
    diagnostics: dict[str, Any] = {}
    resolved: dict[str, Any] = {}
    layout_info: dict[str, Any] = {"method": "none"}
    baseline_layout: dict[str, Any] | None = None
    level0_hashes: set[str] = set()

    if needs_features:
        from .features.registry import build_features

        features, row_ids = build_features(run_dir, metadata, config)
        level0_hashes = set(features.hashes)
        resolved.update(
            {
                "metric": features.metric.value,
                "feature_dimension": int(features.matrix.shape[1]) if features.matrix.size else 0,
                "rows_with_genome_body": int(features.mask.sum()),
                "stage_order": [name for name, on in config.stage_flags().items() if on],
            }
        )

        row_weights = None
        labels = None
        if config.any_stage_enabled:
            assignment = run_stages(features, config)
            diagnostics = assignment.diagnostics
            labels = assignment.labels
            row_weights = assignment.row_weights

        # A row that couldn't be clustered (no genome body) falls back to its location.
        keys = {}
        for i, individual_id in enumerate(row_ids):
            key = f"c{int(labels[i])}" if (labels is not None and features.mask[i]) else features.hashes[i]
            keys[individual_id] = key
            row_keys.append(key)

        node_vectors = _centroids(row_keys, features)
        if row_weights is not None:
            for i, key in enumerate(row_keys):
                node_weights[key] = max(node_weights.get(key, 0.0), float(row_weights[i]))

    builder = STNBuilder(metadata)
    baseline_builder = STNBuilder(metadata) if baseline else None
    for ev in run_dir.iter_evaluations():
        builder.add(ev, keys[ev.individual_id] if keys is not None else ev.genome_hash)
        if baseline_builder is not None:
            baseline_builder.add(ev, ev.genome_hash)
        if keys is None:
            level0_hashes.add(ev.genome_hash)

    migrations = run_dir.read_migrations()
    stn = builder.finish(migrations)
    baseline_stn = baseline_builder.finish(migrations) if baseline_builder else None

    for node in stn.nodes:
        node.weight = node_weights.get(node.key, 0.0)

    if node_vectors and features is not None:
        from .stn import layout

        _attach_radius(stn, node_vectors, features, row_keys)
        projection = layout.fit(
            features.matrix[features.usable], config, features.metric
        )
        layout_info = layout.apply(stn, node_vectors, projection)

        if baseline_stn is not None:
            baseline_vectors = _centroids(features.hashes, features)
            baseline_layout = layout.apply(baseline_stn, baseline_vectors, projection)

    elapsed = time.perf_counter() - started
    out_path = Path(out_dir)
    if write:
        write_artifact(
            out_path,
            stn,
            config,
            source_path=Path(run_path),
            level0_nodes=len(level0_hashes),
            resolved=resolved,
            diagnostics=diagnostics,
            layout=layout_info,
            elapsed_seconds=elapsed,
        )
    return Result(
        stn=stn,
        out_dir=out_path,
        level0_nodes=len(level0_hashes),
        elapsed_seconds=elapsed,
        diagnostics=diagnostics,
        resolved=resolved,
        layout=layout_info,
        baseline_stn=baseline_stn,
        baseline_layout=baseline_layout,
    )
