from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

import numpy as np


class Metric(str, Enum):
    """The space an encoding lives in. Decides which LSH family applies."""

    L2 = "l2"
    HAMMING = "hamming"
    JACCARD = "jaccard"


class ClusterStage(Protocol):
    name: str

    def fit_predict(self, features: "FeatureSet", groups: np.ndarray) -> np.ndarray:
        """Label every row, refining the partition in `groups`."""

    def diagnostics(self) -> dict[str, Any]:
        """What the stage did. Ends up in clustering.json."""


class FeatureSet:
    """Vectorised genomes, plus the sparse sets LSH needs for Jaccard spaces."""

    def __init__(
        self,
        matrix: np.ndarray,
        metric: Metric,
        hashes: list[str],
        mask: np.ndarray,
        sets: list[frozenset[int]] | None = None,
        times: np.ndarray | None = None,
        generations: np.ndarray | None = None,
        islands: np.ndarray | None = None,
    ):
        self.matrix = matrix
        self.metric = metric
        self.hashes = hashes
        self.mask = mask
        self.sets = sets
        self.times = times
        self.generations = generations
        self.islands = islands

    def __len__(self) -> int:
        return len(self.hashes)

    @property
    def usable(self) -> np.ndarray:
        """Indices of the rows that can actually be clustered."""
        return np.flatnonzero(self.mask)
