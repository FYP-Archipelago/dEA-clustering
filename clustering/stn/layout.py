from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from ..config import Config
from ..types import Metric
from .builder import STN

COMPONENTS = 3


@dataclass
class Projection:
    """A fitted map from feature space to 3D."""

    method: str
    metric: Metric
    source_dimension: int
    explained_variance: list[float] = field(default_factory=list)
    _pca: PCA | None = None
    _anchors: np.ndarray | None = None
    _basis: np.ndarray | None = None
    _row_means: np.ndarray | None = None
    _scale: np.ndarray | None = None

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return np.zeros((0, COMPONENTS))
        if self._pca is not None:
            return _pad(self._pca.transform(matrix))
        squared = _squared_distances(matrix, self._anchors)
        return _pad(-0.5 * (squared - self._row_means[None, :]) @ self._basis * self._scale)

    def info(self, positioned: int) -> dict[str, Any]:
        return {
            "method": self.method,
            "components": COMPONENTS,
            "explained_variance_ratio": self.explained_variance,
            "explained_variance_total": round(float(sum(self.explained_variance)), 6),
            "nodes_positioned": positioned,
            "source_dimension": self.source_dimension,
            "metric": self.metric.value,
            # Both panels of a comparison plot are transformed by this same fit.
            "fitted_on": "evaluations",
        }


def _squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(
        (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T, 0.0
    )


def _pad(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[1] >= COMPONENTS:
        return matrix[:, :COMPONENTS]
    padded = np.zeros((matrix.shape[0], COMPONENTS))
    padded[:, : matrix.shape[1]] = matrix
    return padded


def fit(matrix: np.ndarray, config: Config, metric: Metric) -> Projection | None:
    """Fit the projection on the run's evaluations."""
    method = config.layout.method
    if method == "none" or matrix.size == 0 or matrix.shape[0] < 2:
        return None

    if method == "mds":
        return _fit_mds(matrix, config, metric)

    components = min(COMPONENTS, matrix.shape[0], matrix.shape[1])
    model = PCA(n_components=components).fit(matrix)
    return Projection(
        method="pca",
        metric=metric,
        source_dimension=int(matrix.shape[1]),
        explained_variance=[round(float(v), 6) for v in model.explained_variance_ratio_],
        _pca=model,
    )


def _fit_mds(matrix: np.ndarray, config: Config, metric: Metric) -> Projection:
    rng = np.random.default_rng(config.seed)
    take = min(config.layout.mds_landmarks, matrix.shape[0])
    index = rng.choice(matrix.shape[0], size=take, replace=False) if take < matrix.shape[0] else np.arange(take)
    anchors = matrix[index]

    squared = _squared_distances(anchors, anchors)
    row_means = squared.mean(axis=1)
    grand = row_means.mean()
    centred = -0.5 * (squared - row_means[:, None] - row_means[None, :] + grand)
    values, vectors = np.linalg.eigh(centred)
    order = np.argsort(values)[::-1][:COMPONENTS]
    values, vectors = np.clip(values[order], 0.0, None), vectors[:, order]

    total = float(np.clip(np.linalg.eigvalsh(centred), 0.0, None).sum())
    variance = [round(float(v / total), 6) for v in values] if total > 0 else []
    scale = np.where(values > 0, 1.0 / np.sqrt(np.where(values > 0, values, 1.0)), 0.0)

    return Projection(
        method="mds",
        metric=metric,
        source_dimension=int(matrix.shape[1]),
        explained_variance=variance,
        _anchors=anchors,
        _basis=vectors,
        _row_means=row_means,
        _scale=scale,
    )


def apply(stn: STN, node_vectors: dict[str, np.ndarray], projection: Projection | None) -> dict[str, Any]:
    if projection is None or not node_vectors:
        return {"method": "none"}
    keys = [n.key for n in stn.nodes if n.key in node_vectors]
    if not keys:
        return {"method": "none", "reason": "no node had a feature vector"}

    coordinates = projection.transform(np.stack([node_vectors[k] for k in keys]))
    placed = dict(zip(keys, coordinates))
    centre = coordinates.mean(axis=0)
    for node in stn.nodes:
        node.centroid = placed.get(node.key, centre)
    return projection.info(len(keys))
