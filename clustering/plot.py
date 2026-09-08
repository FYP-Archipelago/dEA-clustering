from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.cm as cm  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

from .reader import RunMetadata  # noqa: E402
from .stn.builder import STN  # noqa: E402

LEGEND_SWATCH = "#8a8f98"


def island_palette(island_ids: list[int]) -> dict[int, tuple]:
    ordered = sorted(island_ids)
    count = len(ordered)
    if count <= 10:
        colours = matplotlib.colormaps["tab10"].colors[:count]
    elif count <= 20:
        colours = matplotlib.colormaps["tab20"].colors[:count]
    else:
        cmap = matplotlib.colormaps["turbo"]
        colours = [cmap(i / max(count - 1, 1)) for i in range(count)]
    return dict(zip(ordered, colours))

MAX_EDGES_DRAWN = 4000


def _positions(stn: STN) -> np.ndarray:
    out = np.zeros((len(stn.nodes), 3))
    for i, node in enumerate(stn.nodes):
        if node.centroid is not None and len(node.centroid) >= 3:
            out[i] = node.centroid[:3]
    return out


def _draw(axes, stn: STN, title: str, layout: dict | None, palette: dict[int, tuple],
          limits: np.ndarray | None = None) -> None:
    positions = _positions(stn)
    members = np.array([n.members for n in stn.nodes], dtype=float)
    fallback = (0.6, 0.6, 0.6, 1.0)
    colours = [
        palette.get(max(n.visits, key=n.visits.get), fallback) if n.visits else fallback
        for n in stn.nodes
    ]
    rims = ["#111111" if len(n.visits) > 1 else "none" for n in stn.nodes]

    edges = stn.edges
    if len(edges) > MAX_EDGES_DRAWN:
        edges = edges[:: len(edges) // MAX_EDGES_DRAWN + 1]
    if edges:
        segments = np.array([[positions[e.source], positions[e.target]] for e in edges])
        axes.add_collection3d(
            Line3DCollection(
                segments, colors="#9aa0a6", linewidths=0.3, alpha=0.35, zorder=1
            )
        )

    sizes = 12.0 + 70.0 * (members / members.max()) if members.max() > 0 else 12.0
    axes.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        s=sizes, c=colours, edgecolors=rims, linewidths=0.5, alpha=0.85,
        depthshade=False, zorder=2,
    )

    for edge in stn.node_migrations:
        start, end = positions[edge.source_node], positions[edge.target_node]
        axes.plot(*zip(start, end), color="#d81b60", linewidth=2.0, alpha=0.95, zorder=3)
        axes.scatter(*end, s=90, marker="^", color="#d81b60", zorder=4, depthshade=False)

    spread = 0.012 * float(np.ptp(positions, axis=0).max() or 1.0)
    star_xyz, star_colours = [], []
    for i, node in enumerate(stn.nodes):
        islands = sorted(node.final_best_islands)
        for k, island_id in enumerate(islands):
            offset = np.zeros(3)
            if len(islands) > 1:
                angle = 2.0 * np.pi * k / len(islands)
                offset[:2] = spread * np.array([np.cos(angle), np.sin(angle)])
            star_xyz.append(positions[i] + offset)
            star_colours.append(palette.get(island_id, fallback))
    if star_xyz:
        star_xyz = np.array(star_xyz)
        axes.scatter(
            star_xyz[:, 0], star_xyz[:, 1], star_xyz[:, 2],
            s=760, marker="*", color="#ffd600", edgecolors="none",
            zorder=10, depthshade=False,
        )

    variance = (layout or {}).get("explained_variance_total")
    quality = f" - 3D keeps {variance * 100:.0f}% of variance" if variance else ""
    axes.set_title(
        f"{title}\n{len(stn.nodes):,} nodes - {len(stn.edges):,} edges - "
        f"{len(stn.node_migrations)} migrations{quality}",
        fontsize=13,
        pad=18,
    )
    axes.set_xticklabels([])
    axes.set_yticklabels([])
    axes.set_zticklabels([])
    axes.grid(False)
    if limits is not None:
        axes.set_xlim(limits[0]); axes.set_ylim(limits[1]); axes.set_zlim(limits[2])
    axes.set_box_aspect((1, 1, 1), zoom=1.35)


def render(
    clustered: STN,
    metadata: RunMetadata,
    path: Path,
    *,
    baseline: STN | None = None,
    layout: dict | None = None,
    baseline_layout: dict | None = None,
    stages: str = "",
) -> Path:
    panels = 2 if baseline is not None else 1
    figure = plt.figure(figsize=(24 if panels == 2 else 13, 12))
    island_ids = sorted(metadata.islands) or sorted(
        {i for n in clustered.nodes for i in n.visits}
    )
    palette = island_palette(island_ids)

    drawn = [_positions(clustered)] + ([_positions(baseline)] if baseline is not None else [])
    stacked = np.vstack([p for p in drawn if p.size])
    if stacked.size:
        low, high = stacked.min(axis=0), stacked.max(axis=0)
        pad = np.maximum((high - low) * 0.05, 1e-9)
        limits = np.stack([low - pad, high + pad], axis=1)
    else:
        limits = None

    if baseline is not None:
        _draw(figure.add_subplot(121, projection="3d", computed_zorder=False),
              baseline, "Level 0 - no clustering", baseline_layout, palette, limits)
        _draw(figure.add_subplot(122, projection="3d", computed_zorder=False),
              clustered, f"Clustered - {stages}", layout, palette, limits)
        reduction = (1 - len(clustered.nodes) / len(baseline.nodes)) * 100 if baseline.nodes else 0.0
        tail = f"   -   node reduction {reduction:.1f}%"
    else:
        _draw(figure.add_subplot(111, projection="3d", computed_zorder=False),
              clustered, stages or "Level 0 - no clustering", layout, palette, limits)
        tail = ""

    figure.suptitle(
        f"{metadata.run_id}   -   {metadata.algorithm} / {metadata.benchmark} "
        f"({metadata.genome_encoding})   -   {len(metadata.islands)} islands   -   "
        f"{clustered.total_evaluations:,} evaluations{tail}",
        fontsize=17,
        y=0.975,
    )
    figure.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=LEGEND_SWATCH, markersize=10,
                   label="node (colour = island, size = visits)"),
            Line2D([], [], marker="o", ls="", mfc=LEGEND_SWATCH, mec="#111", mew=1.5,
                   markersize=10, label="visited by >1 island"),
            Line2D([], [], color="#9aa0a6", lw=2, label="trajectory edge"),
            Line2D([], [], color="#d81b60", marker="^", lw=2, markersize=10,
                   label="migration edge"),
            Line2D([], [], marker="*", ls="", color="#ffd600", markersize=24,
                   label="island's final best (one per island)"),
        ],
        loc="lower center", ncol=5, frameon=False, fontsize=13,
        bbox_to_anchor=(0.5, 0.012),
    )
    figure.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.15, wspace=0.01)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def _fitness_positions(
    stn: STN,
    norm: mcolors.Normalize,
    maximising: bool,
    z_scale: float,
) -> np.ndarray:
    pca_pos = _positions(stn)
    out = pca_pos.copy()
    for i, node in enumerate(stn.nodes):
        f = node.best_fitness
        if np.isfinite(f):
            z_norm = float(norm(f))
            z = z_norm if maximising else (1.0 - z_norm)
        else:
            z = 0.0
        out[i, 2] = z * z_scale
    return out


def _draw_fitness(
    axes,
    stn: STN,
    title: str,
    layout: dict | None,
    palette: dict[int, tuple],
    norm: mcolors.Normalize,
    maximising: bool,
    z_scale: float,
    limits: np.ndarray | None = None,
) -> None:
    """Draw one STN panel: XY from PCA, Z from fitness, colour from island."""
    positions = _fitness_positions(stn, norm, maximising, z_scale)
    members = np.array([n.members for n in stn.nodes], dtype=float)

    fallback = (0.6, 0.6, 0.6, 1.0)
    colours = [
        palette.get(max(n.visits, key=n.visits.get), fallback) if n.visits else fallback
        for n in stn.nodes
    ]
    rims = ["#111111" if len(n.visits) > 1 else "none" for n in stn.nodes]

    edges = stn.edges
    if len(edges) > MAX_EDGES_DRAWN:
        edges = edges[:: len(edges) // MAX_EDGES_DRAWN + 1]
    if edges:
        segments = np.array([[positions[e.source], positions[e.target]] for e in edges])
        axes.add_collection3d(
            Line3DCollection(
                segments, colors="#9aa0a6", linewidths=0.3, alpha=0.25, zorder=1
            )
        )

    sizes = 12.0 + 70.0 * (members / members.max()) if members.max() > 0 else 12.0
    axes.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        s=sizes, c=colours, edgecolors=rims, linewidths=0.5,
        alpha=0.88, depthshade=False, zorder=2,
    )

    for edge in stn.node_migrations:
        start, end = positions[edge.source_node], positions[edge.target_node]
        axes.plot(*zip(start, end), color="#d81b60", linewidth=2.0, alpha=0.95, zorder=3)
        axes.scatter(*end, s=90, marker="^", color="#d81b60", zorder=4, depthshade=False)

    spread = 0.012 * float(np.ptp(positions, axis=0).max() or 1.0)
    star_xyz, star_colours = [], []
    for i, node in enumerate(stn.nodes):
        islands = sorted(node.final_best_islands)
        for k, island_id in enumerate(islands):
            offset = np.zeros(3)
            if len(islands) > 1:
                angle = 2.0 * np.pi * k / len(islands)
                offset[:2] = spread * np.array([np.cos(angle), np.sin(angle)])
            star_xyz.append(positions[i] + offset)
            star_colours.append(palette.get(island_id, fallback))
    if star_xyz:
        star_xyz = np.array(star_xyz)
        axes.scatter(
            star_xyz[:, 0], star_xyz[:, 1], star_xyz[:, 2],
            s=760, marker="*", color="#ffd600", edgecolors="none",
            zorder=10, depthshade=False,
        )

    variance = (layout or {}).get("explained_variance_total")
    quality = f" - XY keeps {variance * 100:.0f}% of var." if variance else ""
    direction = "↑ better" if maximising else "↑ better (min.)"
    axes.set_title(
        f"{title}\n{len(stn.nodes):,} nodes · {len(stn.edges):,} edges · "
        f"{len(stn.node_migrations)} migrations{quality}",
        fontsize=13,
        pad=14,
    )
    axes.set_xlabel("PCA 1", fontsize=9, labelpad=1)
    axes.set_ylabel("PCA 2", fontsize=9, labelpad=1)
    axes.set_zlabel(f"fitness  {direction}", fontsize=9, labelpad=1)
    axes.set_xticklabels([])
    axes.set_yticklabels([])
    axes.set_zticklabels([])
    axes.grid(False)
    if limits is not None:
        axes.set_xlim(limits[0]); axes.set_ylim(limits[1]); axes.set_zlim(limits[2])
    axes.view_init(elev=30, azim=-60)
    axes.set_box_aspect((1, 1, 0.8))


def render_fitness(
    clustered: STN,
    metadata: RunMetadata,
    path: Path,
    *,
    baseline: STN | None = None,
    layout: dict | None = None,
    baseline_layout: dict | None = None,
    stages: str = "",
) -> Path:
    """Render STN with XY = PCA layout and Z = best_fitness"""
    panels = 2 if baseline is not None else 1
    figure = plt.figure(figsize=(26 if panels == 2 else 14, 13))

    island_ids = sorted(metadata.islands) or sorted(
        {i for n in clustered.nodes for i in n.visits}
    )
    palette = island_palette(island_ids)

    all_stns = [clustered] + ([baseline] if baseline is not None else [])
    all_fits = [
        n.best_fitness
        for stn in all_stns
        for n in stn.nodes
        if np.isfinite(n.best_fitness)
    ]
    if all_fits:
        vmin, vmax = min(all_fits), max(all_fits)
        if vmin == vmax:
            vmin -= 1e-9
    else:
        vmin, vmax = 0.0, 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    stacked_pca = np.vstack([p for p in [_positions(stn) for stn in all_stns] if p.size])
    if stacked_pca.size:
        xy_range = float(np.ptp(stacked_pca[:, :2]))
        z_scale = max(xy_range * 0.6, 1e-6)
    else:
        z_scale = 1.0

    fit_pos_list = [
        _fitness_positions(stn, norm, metadata.maximising, z_scale) for stn in all_stns
    ]
    stacked = np.vstack([p for p in fit_pos_list if p.size])
    if stacked.size:
        low, high = stacked.min(axis=0), stacked.max(axis=0)
        pad = np.maximum((high - low) * 0.06, 1e-9)
        limits = np.stack([low - pad, high + pad], axis=1)
    else:
        limits = None

    kw = dict(palette=palette, norm=norm, maximising=metadata.maximising,
              z_scale=z_scale, limits=limits)

    if baseline is not None:
        _draw_fitness(figure.add_subplot(121, projection="3d", computed_zorder=False),
                      baseline, "Level 0 - no clustering", baseline_layout, **kw)
        _draw_fitness(figure.add_subplot(122, projection="3d", computed_zorder=False),
                      clustered, f"Clustered - {stages}", layout, **kw)
        reduction = (1 - len(clustered.nodes) / len(baseline.nodes)) * 100 if baseline.nodes else 0.0
        tail = f"   -   node reduction {reduction:.1f}%"
    else:
        _draw_fitness(figure.add_subplot(111, projection="3d", computed_zorder=False),
                      clustered, stages or "Level 0 - no clustering", layout, **kw)
        tail = ""

    figure.suptitle(
        f"{metadata.run_id}   -   {metadata.algorithm} / {metadata.benchmark} "
        f"({metadata.genome_encoding})   -   {len(metadata.islands)} islands   -   "
        f"{clustered.total_evaluations:,} evaluations{tail}",
        fontsize=17,
        y=0.98,
    )
    figure.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=LEGEND_SWATCH, markersize=10,
                   label="node (colour = island, Z = fitness, size = visits)"),
            Line2D([], [], marker="o", ls="", mfc=LEGEND_SWATCH, mec="#111", mew=1.5,
                   markersize=10, label="visited by >1 island"),
            Line2D([], [], color="#9aa0a6", lw=2, label="trajectory edge"),
            Line2D([], [], color="#d81b60", marker="^", lw=2, markersize=10,
                   label="migration edge"),
            Line2D([], [], marker="*", ls="", color="#ffd600", markersize=24,
                   label="island's final best"),
        ],
        loc="lower center", ncol=5, frameon=False, fontsize=12,
        bbox_to_anchor=(0.5, 0.01),
    )
    figure.subplots_adjust(left=0.04, right=0.96, top=0.90, bottom=0.13, wspace=0.05)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return path
