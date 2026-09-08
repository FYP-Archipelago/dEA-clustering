from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..reader import Evaluation, MigrationEdge, RunMetadata

MIGRATION_OPERATOR = "migration_arrival"


@dataclass
class NodeAccumulator:
    index: int
    key: str
    genome_hash: str
    members: int = 0
    visits: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    best_fitness: float = float("nan")
    fitness_sum: float = 0.0
    first_generation: int | None = None
    last_generation: int | None = None
    first_seen: float = float("inf")
    last_seen: float = float("-inf")
    holds_migrant: bool = False
    holds_island_best: bool = False
    final_best_islands: set[int] = field(default_factory=set)
    centroid: np.ndarray | None = None
    radius: float = 0.0
    weight: float = 0.0

    def add(self, ev: Evaluation, t_wall: float, maximising: bool) -> None:
        self.members += 1
        if ev.island_id is not None:
            self.visits[ev.island_id] += 1
        self.fitness_sum += ev.fitness
        if self.members == 1 or (
            ev.fitness > self.best_fitness
            if maximising
            else ev.fitness < self.best_fitness
        ):
            self.best_fitness = ev.fitness
        if ev.generation is not None:
            if self.first_generation is None or ev.generation < self.first_generation:
                self.first_generation = ev.generation
            if self.last_generation is None or ev.generation > self.last_generation:
                self.last_generation = ev.generation
        self.first_seen = min(self.first_seen, t_wall)
        self.last_seen = max(self.last_seen, t_wall)
        self.holds_migrant |= ev.operator == MIGRATION_OPERATOR
        self.holds_island_best |= bool(ev.is_island_best)


@dataclass
class EdgeAccumulator:
    source: int
    target: int
    island: int | None
    weight: int = 0
    operators: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    first_generation: int | None = None
    last_generation: int | None = None

    def add(self, operator: str, generation: int | None) -> None:
        self.weight += 1
        self.operators[operator] += 1
        if generation is not None:
            if self.first_generation is None or generation < self.first_generation:
                self.first_generation = generation
            if self.last_generation is None or generation > self.last_generation:
                self.last_generation = generation


@dataclass
class NodeMigration:
    source_node: int
    target_node: int
    source_island: int
    dest_island: int
    migration_id: str
    migrants: int = 1
    latency_seconds: float | None = None
    generational_drift: int | None = None


@dataclass
class STN:
    nodes: list[NodeAccumulator]
    edges: list[EdgeAccumulator]
    node_migrations: list[NodeMigration]
    island_migrations: list[MigrationEdge]
    metadata: RunMetadata
    total_evaluations: int
    self_loops: int
    unresolved_parents: int

    @property
    def compression(self) -> float:
        return len(self.nodes) / self.total_evaluations if self.total_evaluations else 0.0


class STNBuilder:
    """Accumulates a graph from evaluations that already carry a cluster key."""

    def __init__(self, metadata: RunMetadata):
        self.metadata = metadata
        self._nodes: dict[str, NodeAccumulator] = {}
        self._edges: dict[tuple[int, int, int | None], EdgeAccumulator] = {}
        self._individual_node: dict[str, int] = {}
        self._node_list: list[NodeAccumulator] = []
        self._offsets = {
            iid: metadata.clock_offset_seconds(iid) for iid in metadata.islands
        }
        self._final_best: dict[str, list[int]] = {}
        for island_id, outcome in metadata.outcomes.items():
            if outcome.best_genome_hash:
                self._final_best.setdefault(outcome.best_genome_hash, []).append(island_id)
        self.total = 0
        self.self_loops = 0
        self.unresolved_parents = 0

    def _node_for(self, key: str, ev: Evaluation) -> NodeAccumulator:
        node = self._nodes.get(key)
        if node is None:
            node = NodeAccumulator(len(self._node_list), key, ev.genome_hash)
            self._nodes[key] = node
            self._node_list.append(node)
        return node

    def add(self, ev: Evaluation, key: str) -> None:
        """Record one evaluation under its cluster key."""
        node = self._node_for(key, ev)
        t_wall = ev.t_wall + self._offsets.get(ev.island_id, 0.0)
        node.add(ev, t_wall, self.metadata.maximising)
        node.final_best_islands.update(self._final_best.get(ev.genome_hash, ()))
        self._individual_node[ev.individual_id] = node.index
        self.total += 1

        if ev.operator == MIGRATION_OPERATOR:
            return
        for parent in ev.parent_ids:
            source = self._individual_node.get(parent)
            if source is None:
                self.unresolved_parents += 1
                continue
            if source == node.index:
                self.self_loops += 1
            edge_key = (source, node.index, ev.island_id)
            edge = self._edges.get(edge_key)
            if edge is None:
                edge = EdgeAccumulator(source, node.index, ev.island_id)
                self._edges[edge_key] = edge
            edge.add(ev.operator, ev.generation)

    def finish(self, migrations: list[MigrationEdge]) -> STN:
        node_migrations = self._build_node_migrations(migrations)
        return STN(
            nodes=self._node_list,
            edges=list(self._edges.values()),
            node_migrations=node_migrations,
            island_migrations=migrations,
            metadata=self.metadata,
            total_evaluations=self.total,
            self_loops=self.self_loops,
            unresolved_parents=self.unresolved_parents,
        )

    def _build_node_migrations(self, migrations: list[MigrationEdge]) -> list[NodeMigration]:
        merged: dict[tuple[int, int, str], NodeMigration] = {}
        for edge in migrations:
            for origin, arrived in zip(
                edge.origin_individual_ids, edge.arrived_individual_ids
            ):
                source = self._individual_node.get(origin)
                target = self._individual_node.get(arrived)
                if source is None or target is None:
                    continue
                key = (source, target, edge.migration_id)
                existing = merged.get(key)
                if existing is None:
                    merged[key] = NodeMigration(
                        source_node=source,
                        target_node=target,
                        source_island=edge.source_island,
                        dest_island=edge.dest_island,
                        migration_id=edge.migration_id,
                        latency_seconds=edge.latency_seconds,
                        generational_drift=edge.generational_drift,
                    )
                else:
                    existing.migrants += 1
        return list(merged.values())


def island_summaries(stn: STN) -> list[dict[str, Any]]:
    meta = stn.metadata
    per_island_nodes: dict[int, int] = defaultdict(int)
    per_island_evals: dict[int, int] = defaultdict(int)
    for node in stn.nodes:
        for island_id, count in node.visits.items():
            per_island_nodes[island_id] += 1
            per_island_evals[island_id] += count

    summaries = []
    for island_id in sorted(set(meta.islands) | set(per_island_nodes)):
        island = meta.islands.get(island_id)
        outcome = meta.outcomes.get(island_id)
        summaries.append(
            {
                "island_id": island_id,
                "algorithm": island.algorithm if island else meta.algorithm,
                "hostname": island.hostname if island else None,
                "population_size": island.population_size if island else None,
                "neighbours_out": list(island.neighbours_out) if island else [],
                "neighbours_in": list(island.neighbours_in) if island else [],
                "evaluations": per_island_evals.get(island_id, 0),
                "nodes_visited": per_island_nodes.get(island_id, 0),
                "termination_reason": outcome.termination_reason if outcome else None,
                "generations_completed": (
                    outcome.generations_completed if outcome else None
                ),
                "best_fitness": outcome.best_fitness if outcome else None,
                "wallclock_seconds": outcome.wallclock_seconds if outcome else None,
                "faulted": bool(outcome.faulted) if outcome else False,
                "fault_kind": outcome.fault_kind if outcome else None,
                "records_dropped": outcome.records_dropped if outcome else 0,
            }
        )
    return summaries
