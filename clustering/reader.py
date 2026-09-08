from __future__ import annotations

import csv
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

csv.field_size_limit(sys.maxsize)


def _int(value: str | None) -> int | None:
    return int(value) if value else None


def _float(value: str | None) -> float | None:
    return float(value) if value else None


def _bool(value: str | None) -> bool | None:
    return None if not value else value == "1"


class Evaluation(NamedTuple):
    island_id: int | None
    seq: int
    eval_index: int
    generation: int | None
    t_wall: float
    individual_id: str
    parent_ids: tuple[str, ...]
    operator: str
    fitness: float
    objective: float | None
    feasible: bool | None
    is_island_best: bool | None
    genome_encoding: str
    genome_hash: str
    genome_repr: str | None
    genome_repr_mode: str
    genome_precision_decimals: int | None
    genome_dim: int | None
    origin_island: int | None
    origin_individual_id: str | None


@dataclass(frozen=True)
class Island:
    island_id: int
    genome_encoding: str
    algorithm: str
    algorithm_params: dict[str, Any]
    benchmark: str
    benchmark_params: dict[str, Any]
    population_size: int
    hostname: str | None
    clock_offset_ns: int | None
    neighbours_out: tuple[int, ...]
    neighbours_in: tuple[int, ...]


@dataclass(frozen=True)
class MigrationEdge:
    migration_id: str
    source_island: int
    dest_island: int
    source_generation: int
    dest_generation: int | None
    delivered: bool
    accepted: bool
    num_migrants: int
    origin_individual_ids: tuple[str, ...]
    arrived_individual_ids: tuple[str, ...]
    replaced_individual_ids: tuple[str, ...]
    migrant_genome_hashes: tuple[str, ...]
    migrant_fitnesses: tuple[float, ...]
    latency_seconds: float | None
    generational_drift: int | None
    topology: str
    selection_policy: str
    replacement_policy: str | None
    t_wall_send: float
    t_wall_arrive: float | None


@dataclass
class IslandOutcome:
    island_id: int
    termination_reason: str | None = None
    generations_completed: int | None = None
    evaluations_total: int | None = None
    best_fitness: float | None = None
    best_genome_hash: str | None = None
    wallclock_seconds: float | None = None
    faulted: bool = False
    fault_kind: str | None = None
    fault_detail: str | None = None
    records_dropped: int = 0


@dataclass
class RunMetadata:
    run_id: str
    schema_version: str
    algorithm: str
    benchmark: str
    num_islands: int
    evaluation_budget: int
    maximising: bool
    started_at: str
    genome_encoding: str
    islands: dict[int, Island] = field(default_factory=dict)
    outcomes: dict[int, IslandOutcome] = field(default_factory=dict)
    migration_policy: dict[str, Any] = field(default_factory=dict)
    benchmark_params: dict[str, Any] = field(default_factory=dict)
    datasets: dict[str, Any] = field(default_factory=dict)
    run_end: dict[str, Any] = field(default_factory=dict)

    @property
    def intended_topology(self) -> dict[int, list[int]]:
        raw = self.migration_policy.get("neighbours_out") or {}
        return {int(k): list(v) for k, v in raw.items()}

    @property
    def migration_enabled(self) -> bool:
        return bool(self.migration_policy.get("enabled", True))

    def clock_offset_seconds(self, island_id: int | None) -> float:
        island = self.islands.get(island_id) if island_id is not None else None
        if island is None or island.clock_offset_ns is None:
            return 0.0
        return island.clock_offset_ns / 1e9


class RunDirectory:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not (self.path / "run.jsonl").exists():
            raise FileNotFoundError(f"not an Archipelago run directory: {self.path}")

    def _read_schema_version(self) -> str:
        sidecar = self.path / "evaluations.schema.json"
        return str(json.loads(sidecar.read_text())["schema_version"])

    def iter_events(self) -> Iterator[dict[str, Any]]:
        with (self.path / "run.jsonl").open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def iter_evaluations(self) -> Iterator[Evaluation]:
        with (self.path / "evaluations.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                parents = row["parent_ids"]
                yield Evaluation(
                    island_id=_int(row["island_id"]),
                    seq=int(row["seq"]),
                    eval_index=int(row["eval_index"]),
                    generation=_int(row["generation"]),
                    t_wall=float(row["t_wall"]),
                    individual_id=row["individual_id"],
                    parent_ids=tuple(parents.split(";")) if parents else (),
                    operator=row["operator"],
                    fitness=float(row["fitness"]),
                    objective=_float(row["objective"]),
                    feasible=_bool(row["feasible"]),
                    is_island_best=_bool(row["is_island_best"]),
                    genome_encoding=row["genome_encoding"],
                    genome_hash=row["genome_hash"],
                    genome_repr=row["genome_repr"] or None,
                    genome_repr_mode=row["genome_repr_mode"],
                    genome_precision_decimals=_int(row["genome_precision_decimals"]),
                    genome_dim=_int(row["genome_dim"]),
                    origin_island=_int(row["origin_island"]),
                    origin_individual_id=row["origin_individual_id"] or None,
                )

    def read_metadata(self) -> RunMetadata:
        schema_version = self._read_schema_version()
        start: dict[str, Any] | None = None
        islands: dict[int, Island] = {}
        outcomes: dict[int, IslandOutcome] = {}
        run_end: dict[str, Any] = {}
        maximising: bool | None = None

        for event in self.iter_events():
            kind = event["type"]
            if kind == "run_start":
                start = event
            elif kind == "island_start":
                iid = event["island_id"]
                islands[iid] = Island(
                    island_id=iid,
                    genome_encoding=event["genome_encoding"],
                    algorithm=event["algorithm"],
                    algorithm_params=event.get("algorithm_params", {}),
                    benchmark=event["benchmark"],
                    benchmark_params=event.get("benchmark_params", {}),
                    population_size=event["population_size"],
                    hostname=event.get("hostname"),
                    clock_offset_ns=event.get("clock_offset_ns"),
                    neighbours_out=tuple(event.get("neighbours_out") or ()),
                    neighbours_in=tuple(event.get("neighbours_in") or ()),
                )
                outcomes.setdefault(iid, IslandOutcome(iid))
            elif kind == "generation_end" and maximising is None:
                maximising = event.get("maximising")
            elif kind == "island_end":
                iid = event["island_id"]
                outcome = outcomes.setdefault(iid, IslandOutcome(iid))
                outcome.termination_reason = event.get("termination_reason")
                outcome.generations_completed = event.get("generations_completed")
                outcome.evaluations_total = event.get("evaluations_total")
                outcome.best_fitness = event.get("best_fitness")
                outcome.best_genome_hash = event.get("best_genome_hash")
                outcome.wallclock_seconds = event.get("wallclock_seconds")
                sinks = (event.get("control_channel") or {}).get("sink_stats") or {}
                outcome.records_dropped = sum(
                    s.get("dropped", 0) for s in sinks.values()
                )
            elif kind == "node_fault":
                iid = event.get("island_id")
                if iid is not None:
                    outcome = outcomes.setdefault(iid, IslandOutcome(iid))
                    outcome.faulted = True
                    outcome.fault_kind = event.get("fault")
                    outcome.fault_detail = event.get("detail")
            elif kind == "run_end":
                run_end = event

        if start is None:
            raise RuntimeError(f"{self.path.name}: no run_start record")

        datasets = start.get("datasets") or {}
        any_island = next(iter(islands.values()), None)
        return RunMetadata(
            run_id=start["run_id"],
            schema_version=schema_version,
            algorithm=start["algorithm"],
            benchmark=start["benchmark"],
            num_islands=start["num_islands"],
            evaluation_budget=start["evaluation_budget"],
            maximising=bool(
                maximising
                if maximising is not None
                else datasets.get("maximising", False)
            ),
            started_at=start.get("started_at", ""),
            genome_encoding=any_island.genome_encoding if any_island else "",
            islands=islands,
            outcomes=outcomes,
            migration_policy=start.get("migration") or {},
            benchmark_params=any_island.benchmark_params if any_island else {},
            datasets=datasets,
            run_end=run_end,
        )

    def read_migrations(self) -> list[MigrationEdge]:
        sends: dict[str, dict[str, Any]] = {}
        arrives: dict[str, dict[str, Any]] = {}
        for event in self.iter_events():
            if event["type"] == "migration_send":
                sends[event["migration_id"]] = event
            elif event["type"] == "migration_arrive":
                arrives[event["migration_id"]] = event

        edges = []
        for migration_id, send in sends.items():
            arrive = arrives.get(migration_id)
            edges.append(
                MigrationEdge(
                    migration_id=migration_id,
                    source_island=send["source_island"],
                    dest_island=send["dest_island"],
                    source_generation=send["source_generation"],
                    dest_generation=arrive.get("dest_generation") if arrive else None,
                    delivered=arrive is not None,
                    accepted=bool(arrive.get("accepted", True)) if arrive else False,
                    num_migrants=send["num_migrants"],
                    origin_individual_ids=tuple(
                        arrive.get("origin_individual_ids", ())
                        if arrive
                        else send.get("migrant_individual_ids", ())
                    ),
                    arrived_individual_ids=tuple(
                        arrive.get("arrived_individual_ids", ()) if arrive else ()
                    ),
                    replaced_individual_ids=tuple(
                        arrive.get("replaced_individual_ids", ()) or () if arrive else ()
                    ),
                    migrant_genome_hashes=tuple(send.get("migrant_genome_hashes", ())),
                    migrant_fitnesses=tuple(send.get("migrant_fitnesses", ()) or ()),
                    latency_seconds=arrive.get("latency_seconds") if arrive else None,
                    generational_drift=(
                        arrive.get("generational_drift") if arrive else None
                    ),
                    topology=send.get("topology", ""),
                    selection_policy=send.get("selection_policy", ""),
                    replacement_policy=(
                        arrive.get("replacement_policy") if arrive else None
                    ),
                    t_wall_send=send["t_wall"],
                    t_wall_arrive=arrive["t_wall"] if arrive else None,
                )
            )
        return edges
