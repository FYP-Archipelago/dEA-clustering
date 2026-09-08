from __future__ import annotations

import re
from typing import Any, Protocol

import numpy as np

from .. import genome as genome_mod
from ..types import Metric
from ..hashing import feature_hash, hash_tokens

PERMUTATION_DIMENSION = 64
TREE_DIMENSION = 128


class Adapter(Protocol):
    metric: Metric
    dimension: int

    def encode(self, genome: Any) -> tuple[np.ndarray, np.ndarray | None]:
        """Dense vector, plus element ids for the set-valued spaces."""


class RealVectorAdapter:
    """Continuous positions, min-max normalised onto the unit cube."""

    metric = Metric.L2

    def __init__(self, encoding: str, dimension: int, lower: float | None, upper: float | None):
        self.encoding = encoding
        self.dimension = dimension
        self.lower = lower
        self.upper = upper
        self.normalised = lower is not None and upper is not None and upper > lower

    def encode(self, genome: Any) -> tuple[np.ndarray, np.ndarray | None]:
        x = np.asarray(genome_mod.position(genome, self.encoding), dtype=np.float64)
        if self.normalised:
            x = (x - self.lower) / (self.upper - self.lower)
        return x, None


class BitstringAdapter:
    metric = Metric.HAMMING

    def __init__(self, dimension: int):
        self.dimension = dimension

    def encode(self, genome: Any) -> tuple[np.ndarray, np.ndarray | None]:
        return np.asarray(genome, dtype=np.float64), None


class PermutationAdapter:
    """A tour as its set of undirected edges."""

    metric = Metric.JACCARD

    def __init__(self, num_cities: int, seed: int):
        self.num_cities = num_cities
        self.dimension = PERMUTATION_DIMENSION
        self.seed = seed

    def encode(self, genome: Any) -> tuple[np.ndarray, np.ndarray | None]:
        tour = np.asarray(genome, dtype=np.int64)
        nxt = np.roll(tour, -1)
        low = np.minimum(tour, nxt)
        high = np.maximum(tour, nxt)
        ids = (low * self.num_cities + high).astype(np.uint64)
        ids = np.unique(ids)
        return feature_hash(ids, self.dimension, self.seed), ids


_TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|-?\d+\.?\d*|[(),]")


def subtree_shingles(expression: str) -> list[str]:
    """Every subtree of a canonical prefix expression, as its own string."""
    tokens = _TOKEN.findall(expression)
    shingles: list[str] = []
    position = 0

    def parse() -> str:
        nonlocal position
        if position >= len(tokens):
            return ""
        head = tokens[position]
        position += 1
        if position < len(tokens) and tokens[position] == "(":
            position += 1  # consume "("
            args = []
            while position < len(tokens) and tokens[position] != ")":
                args.append(parse())
                if position < len(tokens) and tokens[position] == ",":
                    position += 1
            if position < len(tokens):
                position += 1  # consume ")"
            node = f"{head}({','.join(args)})"
        else:
            node = head
        shingles.append(node)
        return node

    parse()
    return shingles


class ExpressionTreeAdapter:
    metric = Metric.JACCARD

    def __init__(self, seed: int):
        self.dimension = TREE_DIMENSION
        self.seed = seed

    def encode(self, genome: Any) -> tuple[np.ndarray, np.ndarray | None]:
        shingles = subtree_shingles(str(genome))
        ids = np.unique(hash_tokens([s.encode() for s in shingles], self.seed))
        return feature_hash(ids, self.dimension, self.seed), ids


def build_adapter(encoding: str, benchmark_params: dict[str, Any], dimension: int | None, seed: int) -> Adapter:
    if encoding in (
        genome_mod.REAL_VECTOR,
        genome_mod.REAL_VECTOR_VELOCITY,
        genome_mod.REAL_VECTOR_SIGMA,
    ):
        return RealVectorAdapter(
            encoding,
            dimension or int(benchmark_params.get("dimension", 0)),
            benchmark_params.get("lower"),
            benchmark_params.get("upper"),
        )
    if encoding == genome_mod.BITSTRING:
        return BitstringAdapter(
            dimension or int(benchmark_params.get("num_items", benchmark_params.get("dimension", 0)))
        )
    if encoding == genome_mod.PERMUTATION:
        return PermutationAdapter(
            int(benchmark_params.get("num_cities", benchmark_params.get("dimension", 0))), seed
        )
    if encoding == genome_mod.EXPRESSION_TREE:
        return ExpressionTreeAdapter(seed)
    raise ValueError(f"no feature adapter for encoding: {encoding}")
