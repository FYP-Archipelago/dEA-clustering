from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from .. import ARTIFACT_SCHEMA_VERSION, __version__
from .builder import STN, island_summaries

FILES = (
    "manifest.json",
    "nodes.json",
    "edges.json",
    "migrations.json",
    "islands.json",
    "clustering.json",
)


def _clean(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _dump(path: Path, payload: dict[str, Any]) -> int:
    text = json.dumps(payload, indent=2, allow_nan=False)
    path.write_text(text)
    return len(text)


def _node_records(stn: STN) -> list[dict[str, Any]]:
    records = []
    for node in stn.nodes:
        visits = dict(sorted(node.visits.items()))
        record: dict[str, Any] = {
            "id": node.index,
            "genome_hash": node.genome_hash,
            "members": node.members,
            "visits": {str(k): v for k, v in visits.items()},
            "islands": list(visits),
            "shared": len(visits) > 1,
            "best_fitness": _clean(node.best_fitness),
            "mean_fitness": _clean(node.fitness_sum / node.members)
            if node.members
            else None,
            "first_generation": node.first_generation,
            "last_generation": node.last_generation,
            "first_seen": _clean(node.first_seen),
            "last_seen": _clean(node.last_seen),
            "holds_migrant": node.holds_migrant,
            "holds_island_best": node.holds_island_best,
            "final_best_islands": sorted(node.final_best_islands),
        }
        if node.radius:
            record["radius"] = _clean(node.radius)
        if node.weight:
            record["weight"] = _clean(node.weight)
        if node.centroid is not None and len(node.centroid) >= 2:
            position = np.asarray(node.centroid, dtype=float)
            record["x"] = _clean(position[0])
            record["y"] = _clean(position[1])
            record["z"] = _clean(position[2]) if len(position) > 2 else 0.0
        records.append(record)
    return records


def _migration_payload(stn: STN) -> dict[str, Any]:
    meta = stn.metadata
    observed: dict[str, dict[str, int]] = {}
    for edge in stn.island_migrations:
        if not edge.delivered:
            continue
        row = observed.setdefault(str(edge.source_island), {})
        key = str(edge.dest_island)
        row[key] = row.get(key, 0) + edge.num_migrants

    intended = {str(k): v for k, v in meta.intended_topology.items()}
    unused = [
        {"source": int(src), "dest": dest}
        for src, dests in intended.items()
        for dest in dests
        if observed.get(src, {}).get(str(dest), 0) == 0
    ]

    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": meta.run_id,
        "enabled": meta.migration_enabled,
        "policy": {
            k: v
            for k, v in meta.migration_policy.items()
            if k not in ("neighbours_out", "neighbours_in")
        },
        "intended_topology": intended,
        "observed_island_edges": observed,
        "unused_intended_edges": unused,
        "node_edges": [
            {
                "source": m.source_node,
                "target": m.target_node,
                "source_island": m.source_island,
                "dest_island": m.dest_island,
                "migration_id": m.migration_id,
                "migrants": m.migrants,
                "latency_seconds": _clean(m.latency_seconds),
                "generational_drift": m.generational_drift,
            }
            for m in stn.node_migrations
        ],
        "island_edges": [
            {
                "migration_id": e.migration_id,
                "source_island": e.source_island,
                "dest_island": e.dest_island,
                "source_generation": e.source_generation,
                "dest_generation": e.dest_generation,
                "delivered": e.delivered,
                "accepted": e.accepted,
                "migrants": e.num_migrants,
                "latency_seconds": _clean(e.latency_seconds),
                "generational_drift": e.generational_drift,
                "replaced": list(e.replaced_individual_ids),
                "t_wall_send": _clean(e.t_wall_send),
                "t_wall_arrive": _clean(e.t_wall_arrive),
            }
            for e in stn.island_migrations
        ],
    }


def write_artifact(
    out_dir: Path,
    stn: STN,
    config: Config,
    *,
    source_path: Path,
    level0_nodes: int,
    resolved: dict[str, Any],
    diagnostics: dict[str, Any],
    layout: dict[str, Any],
    elapsed_seconds: float,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = stn.metadata

    sizes = {
        "nodes.json": _dump(
            out_dir / "nodes.json",
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "run_id": meta.run_id,
                "nodes": _node_records(stn),
            },
        ),
        "edges.json": _dump(
            out_dir / "edges.json",
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "run_id": meta.run_id,
                "edges": [
                    {
                        "source": e.source,
                        "target": e.target,
                        "island": e.island,
                        "weight": e.weight,
                        "operators": dict(e.operators),
                        "first_generation": e.first_generation,
                        "last_generation": e.last_generation,
                    }
                    for e in stn.edges
                ],
            },
        ),
        "migrations.json": _dump(out_dir / "migrations.json", _migration_payload(stn)),
        "islands.json": _dump(
            out_dir / "islands.json",
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "run_id": meta.run_id,
                "islands": island_summaries(stn),
            },
        ),
        "clustering.json": _dump(
            out_dir / "clustering.json",
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "run_id": meta.run_id,
                "stages": config.stage_flags(),
                "resolved_parameters": resolved,
                "diagnostics": diagnostics,
            },
        ),
    }

    faulted = [o.island_id for o in meta.outcomes.values() if o.faulted]
    dropped = sum(o.records_dropped for o in meta.outcomes.values())
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "generator": {"name": "archipelago-clustering", "version": __version__},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "source_run": {
            "run_id": meta.run_id,
            "path": str(source_path),
            "log_schema_version": meta.schema_version,
            "algorithm": meta.algorithm,
            "benchmark": meta.benchmark,
            "genome_encoding": meta.genome_encoding,
            "num_islands": meta.num_islands,
            "evaluation_budget": meta.evaluation_budget,
            "maximising": meta.maximising,
            "started_at": meta.started_at,
            "datasets": meta.datasets,
            "benchmark_params": meta.benchmark_params,
        },
        "config": config.model_dump(mode="json"),
        "stages": config.stage_flags(),
        "resolved_parameters": resolved,
        "counts": {
            "evaluations": stn.total_evaluations,
            "nodes": len(stn.nodes),
            "edges": len(stn.edges),
            "node_migration_edges": len(stn.node_migrations),
            "island_migration_edges": len(stn.island_migrations),
            "self_loops": stn.self_loops,
            "unresolved_parents": stn.unresolved_parents,
        },
        "baseline": {
            "level0_nodes": level0_nodes,
            "nodes_per_evaluation": round(stn.compression, 6),
            "reduction_vs_level0": round(1.0 - len(stn.nodes) / level0_nodes, 6)
            if level0_nodes
            else 0.0,
        },
        "layout": layout,
        "integrity": {
            "records_dropped": dropped,
            "faulted_islands": faulted,
            # A dropped record upstream is a seq gap: the graph is real but partial.
            "complete": dropped == 0 and stn.unresolved_parents == 0,
        },
        "files": [{"name": name, "bytes": size} for name, size in sizes.items()],
    }
    _dump(out_dir / "manifest.json", manifest)
    return out_dir
