"""
visualization.py
================
Grid Visualisation Module — v2.

New in v2:
  • plot_fuzzy_age_membership() — age input MF chart
  • MF parameters imported from fuzzy.py (single source of truth)
  • _apply_grid_style() helper DRYs up repeated tick/grid setup
  • save_all_figures() helper for batch PNG/PDF export

Public functions:
  plot_comparison()           — side-by-side BFS vs A* + probability overlay
  plot_single()               — single-algorithm plot
  plot_fuzzy_membership()     — height MFs + vertical marker
  plot_fuzzy_age_membership() — age MFs + vertical marker
  plot_fuzzy_output()         — output universe + aggregated MF + defuzz line
  plot_monte_carlo()          — Monte-Carlo density heatmap
  get_animation_frames()      — list of figures for step-by-step animation
  save_all_figures()          — save a list of figures to PNG files or a PDF
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import numpy as np

# Import MF parameters from fuzzy (single source of truth — no duplicate literals)
from fuzzy import (
    trimf, trapmf, trimf_scalar, trapmf_scalar,
    _H_SHORT_PARAMS, _H_MEDIUM_PARAMS, _H_TALL_PARAMS,
    _A_YOUNG_PARAMS, _A_ADULT_PARAMS, _A_SENIOR_PARAMS,
    get_output_mfs,
)


# ─────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────
_C_OBSTACLE      = "#4a4a4a"   # charcoal obstacles
_C_FREE          = "#f7f7f5"   # off-white free cells
_C_VISITED_BFS   = "#2980b9"   # blue
_C_VISITED_ASTAR = "#8e44ad"   # purple
_C_PATH          = "#c0392b"   # crimson path
_C_START         = "#16a085"   # teal start
_C_GOAL          = "#f39c12"   # amber goal
_C_HOTSPOT       = "#f39c12"   # amber hotspot tint
_C_PROB_CM       = "YlOrRd"    # light-friendly heatmap
_C_BG            = "#ffffff"   # figure background
_C_SURFACE       = "#fafafa"   # axes background


# ─────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────

def _apply_grid_style(ax: plt.Axes, rows: int, cols: int) -> None:
    """Apply consistent tick marks and grid lines to a grid panel."""
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4, zorder=5)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(cols));  ax.set_xticklabels(range(cols), fontsize=7)
    ax.set_yticks(range(rows));  ax.set_yticklabels(range(rows), fontsize=7)
    ax.tick_params(axis="both", labelsize=7, length=2)


def _draw_grid_panel(
    ax: plt.Axes,
    grid,
    start: tuple,
    goal: tuple,
    visited: set,
    path: list | None,
    restricted_zones: list | None,
    title: str,
    nodes_explored: int,
    path_length,
    visited_colour: str,
    prob_matrix: np.ndarray | None = None,
    show_prob_overlay: bool = False,
) -> None:
    """
    Render one grid panel with ordered layers:
      1. Base (free / obstacle)
      2. Probability heatmap overlay (optional)
      3. High-probability hotspot zone tint (amber)
      4. Visited cells overlay
      5. Path line
      6. Start / Goal markers
      7. Grid lines
    """
    rows, cols = grid.rows, grid.cols
    data = np.array(grid.grid, dtype=float)

    # ── 1. Base ──────────────────────────────────────────────────
    cmap_base = mcolors.ListedColormap([_C_FREE, _C_OBSTACLE])
    ax.imshow(data, cmap=cmap_base, vmin=0, vmax=1, zorder=1)

    # ── 2. Probability heatmap ───────────────────────────────────
    if show_prob_overlay and prob_matrix is not None:
        masked = np.ma.masked_where(data == 1, prob_matrix)
        ax.imshow(masked, cmap=_C_PROB_CM, vmin=0, vmax=1,
                  alpha=0.45, zorder=2)

    # ── 3. High-probability hotspot zones (amber tint) ───────────
    if restricted_zones:
        zone_map = np.zeros((rows, cols))
        for (r1, c1, r2, c2) in restricted_zones:
            for r in range(max(0, r1), min(rows, r2 + 1)):
                for c in range(max(0, c1), min(cols, c2 + 1)):
                    zone_map[r][c] = 1
        zone_cmap = mcolors.ListedColormap(["none", _C_HOTSPOT])
        ax.imshow(zone_map, cmap=zone_cmap, vmin=0, vmax=1,
                  alpha=0.40, zorder=3)

    # ── 4. Visited cells ─────────────────────────────────────────
    if visited:
        vis_map  = np.zeros((rows, cols))
        for (r, c) in visited:
            if 0 <= r < rows and 0 <= c < cols:
                vis_map[r][c] = 1
        vis_rgba = np.zeros((rows, cols, 4))
        rgb      = mcolors.to_rgb(visited_colour)
        vis_rgba[vis_map == 1] = (*rgb, 0.38)
        ax.imshow(vis_rgba, zorder=4)

    # ── 5. Path ──────────────────────────────────────────────────
    if path and len(path) > 1:
        py = [r for r, c in path]
        px = [c for r, c in path]
        ax.plot(px, py, color=_C_PATH, linewidth=2.8,
                solid_capstyle="round", zorder=6)

    # ── 6. Markers ───────────────────────────────────────────────
    ax.scatter(start[1], start[0], s=160, color=_C_START,
               edgecolors="white", linewidths=1.5, zorder=7, marker="o")
    ax.text(start[1] + 0.35, start[0] - 0.35, "S",
            fontsize=7, color="white", fontweight="bold",
            ha="left", va="top", zorder=8)
    ax.scatter(goal[1], goal[0], s=200, color=_C_GOAL,
               edgecolors="white", linewidths=1.5, zorder=7, marker="*")
    ax.text(goal[1] + 0.35, goal[0] - 0.35, "G",
            fontsize=7, color="white", fontweight="bold",
            ha="left", va="top", zorder=8)

    # ── 7. Grid lines ────────────────────────────────────────────
    _apply_grid_style(ax, rows, cols)

    pl = str(path_length) if path_length is not None else "No path"
    ax.set_title(
        f"{title}\nNodes: {nodes_explored}  |  Path: {pl}",
        fontsize=10, fontweight="bold", pad=8,
    )


def _build_legend(ax: plt.Axes, has_zones: bool, has_prob: bool) -> None:
    handles = [
        mpatches.Patch(facecolor=_C_FREE,     edgecolor="#aaa",
                       linewidth=0.5, label="Free cell"),
        mpatches.Patch(facecolor=_C_OBSTACLE, label="Obstacle"),
    ]
    if has_zones:
        handles.append(mpatches.Patch(
            facecolor=_C_HOTSPOT, alpha=0.6, label="High-prob hotspot"))
    if has_prob:
        handles.append(mpatches.Patch(
            facecolor="#d73027", alpha=0.5, label="Prob. heatmap"))
    handles += [
        mpatches.Patch(facecolor=_C_VISITED_BFS, alpha=0.5, label="Explored"),
        mlines.Line2D([], [], color=_C_PATH, linewidth=2, label="Path"),
        mlines.Line2D([], [], marker="o", color="w",
                      markerfacecolor=_C_START, markersize=8, label="Start"),
        mlines.Line2D([], [], marker="*", color="w",
                      markerfacecolor=_C_GOAL, markersize=10, label="Goal"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7.5,
              framealpha=0.92, edgecolor="#ccc", handlelength=1.4)


# ─────────────────────────────────────────────────────────────────
# Public: side-by-side comparison
# ─────────────────────────────────────────────────────────────────

def plot_comparison(
    grid,
    start: tuple,
    goal: tuple,
    visited_bfs:   set,
    path_bfs:      list | None,
    visited_astar: set,
    path_astar:    list | None,
    restricted_zones: list | None = None,
    risk_level: str = "",
    suspicion_score: float = 0.0,
    prob_matrix: np.ndarray | None = None,
    show_prob_overlay: bool = True,
) -> plt.Figure:
    """Side-by-side BFS vs A* comparison with optional probability overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(_C_BG)

    panels = [
        (axes[0], visited_bfs,   path_bfs,   "BFS — Breadth-First Search", _C_VISITED_BFS),
        (axes[1], visited_astar, path_astar, "A* — Heuristic Search",       _C_VISITED_ASTAR),
    ]
    for ax, visited, path, title, vis_col in panels:
        _draw_grid_panel(
            ax=ax, grid=grid, start=start, goal=goal,
            visited=visited, path=path,
            restricted_zones=restricted_zones, title=title,
            nodes_explored=len(visited),
            path_length=len(path) if path else None,
            visited_colour=vis_col,
            prob_matrix=prob_matrix,
            show_prob_overlay=show_prob_overlay,
        )
        ax.set_facecolor(_C_SURFACE)

    _build_legend(axes[1], bool(restricted_zones), show_prob_overlay)

    risk_str  = f"Risk: {risk_level.upper()}" if risk_level else ""
    score_str = f"Suspicion: {suspicion_score:.2f}" if suspicion_score else ""
    suptitle  = "  ·  ".join(filter(None, [risk_str, score_str]))
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.01)

    plt.tight_layout(pad=2.0)
    return fig


# ─────────────────────────────────────────────────────────────────
# Public: single-algorithm plot
# ─────────────────────────────────────────────────────────────────

def plot_single(
    grid,
    start: tuple,
    goal: tuple,
    visited: set,
    path: list | None,
    restricted_zones: list | None = None,
    algorithm_name: str = "Search",
    prob_matrix: np.ndarray | None = None,
    show_prob_overlay: bool = True,
) -> plt.Figure:
    """Single-algorithm grid plot (used for animation frames and prob map tab)."""
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(_C_BG)
    _draw_grid_panel(
        ax=ax, grid=grid, start=start, goal=goal,
        visited=visited, path=path,
        restricted_zones=restricted_zones,
        title=algorithm_name,
        nodes_explored=len(visited),
        path_length=len(path) if path else None,
        visited_colour=_C_VISITED_BFS,
        prob_matrix=prob_matrix,
        show_prob_overlay=show_prob_overlay,
    )
    ax.set_facecolor(_C_SURFACE)
    _build_legend(ax, bool(restricted_zones), show_prob_overlay)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────
# Public: fuzzy HEIGHT membership chart
# ─────────────────────────────────────────────────────────────────

def plot_fuzzy_membership(height_cm: int) -> plt.Figure:
    """
    Plot triangular / trapezoidal MFs for the height input variable.
    Uses MF parameters imported from fuzzy.py (single source of truth).
    """
    xs     = np.linspace(140, 210, 500)
    short  = trapmf(xs, *_H_SHORT_PARAMS)
    medium = trimf (xs, *_H_MEDIUM_PARAMS)
    tall   = trapmf(xs, *_H_TALL_PARAMS)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(xs, short,  label="Short",  color="#2980b9", linewidth=2)
    ax.plot(xs, medium, label="Medium", color="#e67e22", linewidth=2)
    ax.plot(xs, tall,   label="Tall",   color="#c0392b", linewidth=2)
    ax.axvline(x=height_cm, color="#555", linewidth=1.5, linestyle="--",
               alpha=0.7, label=f"Input: {height_cm} cm")

    for mu, col in [
        (trapmf_scalar(height_cm, *_H_SHORT_PARAMS),  "#2980b9"),
        (trimf_scalar (height_cm, *_H_MEDIUM_PARAMS), "#e67e22"),
        (trapmf_scalar(height_cm, *_H_TALL_PARAMS),   "#c0392b"),
    ]:
        if mu > 0:
            ax.plot(height_cm, mu, "o", color=col, markersize=8, zorder=5)

    ax.set_xlabel("Height (cm)", fontsize=10)
    ax.set_ylabel("Membership degree (x)", fontsize=10)
    ax.set_title("Fuzzy MFs --- Height Input Variable", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.05, 1.15); ax.set_xlim(140, 210)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.4, linewidth=0.5)
    ax.set_facecolor(_C_SURFACE); fig.patch.set_facecolor(_C_BG)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────
# Public: fuzzy AGE membership chart
# ─────────────────────────────────────────────────────────────────

def plot_fuzzy_age_membership(age: int) -> plt.Figure:
    """
    Plot MFs for the age input variable.
    Shows young / adult / senior overlapping sets plus a vertical marker.
    Uses MF parameters imported from fuzzy.py (single source of truth).
    """
    xs     = np.linspace(10, 85, 500)
    young  = trapmf(xs, *_A_YOUNG_PARAMS)
    adult  = trimf (xs, *_A_ADULT_PARAMS)
    senior = trapmf(xs, *_A_SENIOR_PARAMS)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(xs, young,  label="Young",  color="#27ae60", linewidth=2)
    ax.plot(xs, adult,  label="Adult",  color="#e67e22", linewidth=2)
    ax.plot(xs, senior, label="Senior", color="#8e44ad", linewidth=2)
    ax.axvline(x=age, color="#555", linewidth=1.5, linestyle="--",
               alpha=0.7, label=f"Input: {age} yrs")

    for mu, col in [
        (trapmf_scalar(float(age), *_A_YOUNG_PARAMS),  "#27ae60"),
        (trimf_scalar (float(age), *_A_ADULT_PARAMS),  "#e67e22"),
        (trapmf_scalar(float(age), *_A_SENIOR_PARAMS), "#8e44ad"),
    ]:
        if mu > 0:
            ax.plot(age, mu, "o", color=col, markersize=8, zorder=5)

    ax.set_xlabel("Age (years)", fontsize=10)
    ax.set_ylabel("Membership degree (x)", fontsize=10)
    ax.set_title("Fuzzy MFs --- Age Input Variable", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.05, 1.15); ax.set_xlim(10, 85)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.4, linewidth=0.5)
    ax.set_facecolor(_C_SURFACE); fig.patch.set_facecolor(_C_BG)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────
# Public: fuzzy OUTPUT MF chart
# ─────────────────────────────────────────────────────────────────

def plot_fuzzy_output(breakdown: dict) -> plt.Figure:
    """
    Plot the output universe showing:
      • LOW / MEDIUM / HIGH output MFs (faded)
      • Aggregated output fuzzy set (solid blue fill)
      • Vertical centroid defuzzification line (red dashed)
    """
    out_mfs = get_output_mfs()
    x       = out_mfs["x"]
    agg     = breakdown["agg_y"]
    score   = breakdown["total_score"]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.fill_between(x, out_mfs["low"],    alpha=0.15, color="#2980b9")
    ax.fill_between(x, out_mfs["medium"], alpha=0.15, color="#e67e22")
    ax.fill_between(x, out_mfs["high"],   alpha=0.15, color="#c0392b")
    ax.plot(x, out_mfs["low"],    color="#2980b9", linewidth=1.5,
            linestyle="--", label="LOW")
    ax.plot(x, out_mfs["medium"], color="#e67e22", linewidth=1.5,
            linestyle="--", label="MEDIUM")
    ax.plot(x, out_mfs["high"],   color="#c0392b", linewidth=1.5,
            linestyle="--", label="HIGH")
    ax.fill_between(x, agg, alpha=0.40, color="#2980b9",
                    label="Aggregated output")
    ax.plot(x, agg, color="#1a5276", linewidth=2)
    ax.axvline(x=score, color="#c0392b", linewidth=2, linestyle="-.",
               label=f"Defuzz = {score:.2f}")

    ax.set_xlabel("Suspicion score (output universe)", fontsize=10)
    ax.set_ylabel("Membership degree", fontsize=10)
    ax.set_title("Fuzzy Output --- Aggregated MF & Centroid Defuzzification",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.4, linewidth=0.5)
    ax.set_facecolor(_C_SURFACE); fig.patch.set_facecolor(_C_BG)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────
# Public: Monte-Carlo density heatmap
# ─────────────────────────────────────────────────────────────────

def plot_monte_carlo(
    grid,
    density: np.ndarray,
    start: tuple,
    title: str = "Monte-Carlo Agent Density",
    n_runs: int = 0,
    steps: int = 0,
) -> plt.Figure:
    """Visualise the Monte-Carlo simulation density heatmap."""
    rows, cols = grid.rows, grid.cols
    data = np.array(grid.grid, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(_C_BG)

    cmap_base = mcolors.ListedColormap([_C_FREE, _C_OBSTACLE])
    ax.imshow(data, cmap=cmap_base, vmin=0, vmax=1, zorder=1)

    masked = np.ma.masked_where(data == 1, density)
    im     = ax.imshow(masked, cmap="YlOrRd", vmin=0, vmax=1, alpha=0.70, zorder=2)

    ax.scatter(start[1], start[0], s=200, color=_C_START,
               edgecolors="white", linewidths=1.8, zorder=4, marker="o")
    ax.text(start[1] + 0.3, start[0] - 0.3, "S",
            fontsize=8, color="white", fontweight="bold",
            ha="left", va="top", zorder=5)

    _apply_grid_style(ax, rows, cols)

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Normalised visit density", color="#555")
    cbar.ax.yaxis.set_tick_params(color="#555")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#555")

    subtitle = f"{n_runs} runs x {steps} steps" if n_runs else ""
    ax.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight="bold", pad=8)
    ax.set_facecolor(_C_SURFACE)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────
# Public: animation frames
# ─────────────────────────────────────────────────────────────────

def get_animation_frames(
    grid,
    start: tuple,
    goal: tuple,
    path: list | None,
    visited_ordered: list[tuple],
    restricted_zones: list | None = None,
    algorithm_name: str = "Search",
    prob_matrix: np.ndarray | None = None,
    max_frames: int = 60,
) -> list[plt.Figure]:
    """
    Generate a list of matplotlib Figures showing progressive exploration.
    Returns up to `max_frames` figures.
    """
    total  = len(visited_ordered)
    step   = max(1, total // max_frames)
    frames = []

    for i in range(0, total, step):
        subset = set(visited_ordered[:i + 1])
        fig    = plot_single(
            grid=grid, start=start, goal=goal,
            visited=subset, path=None,
            restricted_zones=restricted_zones,
            algorithm_name=f"{algorithm_name} — Step {i+1}/{total}",
            prob_matrix=prob_matrix, show_prob_overlay=True,
        )
        frames.append(fig)

    if path:
        fig = plot_single(
            grid=grid, start=start, goal=goal,
            visited=set(visited_ordered), path=path,
            restricted_zones=restricted_zones,
            algorithm_name=f"{algorithm_name} — Complete",
            prob_matrix=prob_matrix, show_prob_overlay=True,
        )
        frames.append(fig)

    return frames


# ─────────────────────────────────────────────────────────────────
# Public: batch export helper
# ─────────────────────────────────────────────────────────────────

def save_all_figures(
    figs: list[plt.Figure],
    output_path: str,
    dpi: int = 150,
) -> None:
    """
    Save a list of figures to either a multi-page PDF or individual PNGs.

    Args:
        figs        : list of matplotlib Figure objects
        output_path : destination path.  If it ends in '.pdf', all figures
                      are combined into one PDF; otherwise treated as a
                      directory prefix and each figure is saved as a PNG.
        dpi         : resolution for raster output
    """
    import os
    if output_path.endswith(".pdf"):
        import matplotlib.backends.backend_pdf as pdf_be
        with pdf_be.PdfPages(output_path) as pdf:
            for fig in figs:
                pdf.savefig(fig, bbox_inches="tight", dpi=dpi)
    else:
        os.makedirs(output_path, exist_ok=True)
        for i, fig in enumerate(figs):
            fig.savefig(
                os.path.join(output_path, f"figure_{i+1:02d}.png"),
                bbox_inches="tight", dpi=dpi,
            )