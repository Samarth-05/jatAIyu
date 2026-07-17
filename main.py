"""
main.py
=======
Terminal entry point — AI-Based Missing Person Search Simulation v2.

Run with:
    python main.py

New in v2:
  • Age input → fuzzy engine (fuzzify_age)
  • Hours-since-last-seen → probability grid time-decay
  • Age membership chart added to output plots
  • Consistent random.seed + np.random.seed

Uses SearchSimulator (utils.py) as the single orchestrator.
Outputs:
  • Step-by-step console report
  • Fuzzy membership charts (height + age)
  • Fuzzy output chart (aggregated MF + defuzzification)
  • BFS vs A* comparison grid
  • Monte-Carlo density heatmap
"""

__version__ = "2.0"

import sys
import io
import textwrap
import random
import numpy as np
import matplotlib.pyplot as plt

# Ensure UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from utils import SearchSimulator
from visualization import (
    plot_comparison,
    plot_fuzzy_membership,
    plot_fuzzy_age_membership,
    plot_fuzzy_output,
    plot_monte_carlo,
)


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

CONFIG = dict(
    height           = 178,
    age              = 24,
    behavior         = "nervous",
    clothing         = "black",
    grid_rows        = 10,
    grid_cols        = 10,
    obstacle_ratio   = 0.12,
    use_diagonals    = False,    # True → 8-directional movement
    random_seed      = 42,
    last_seen        = None,     # e.g. (2, 3) or None
    hours_since_seen = 6.0,
    mc_runs          = 50,
    mc_steps         = 20,
)


# ═══════════════════════════════════════════════════════════════
# Helper printers
# ═══════════════════════════════════════════════════════════════

def _sep(c: str = "-", w: int = 62) -> None:
    print(c * w)

def _h(t: str) -> None:
    _sep("="); print(f"  {t}"); _sep("=")

def _s(t: str) -> None:
    print(); _sep(); print(f"  {t}"); _sep()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    _h(f"AI-Based Missing Person Search Simulation  v{__version__}")
    print(textwrap.dedent("""\
    Pipeline: Witness Input (height + age + behavior + clothing)
              → Fuzzy Inference (Mamdani + Centroid)
              → Probability Grid (CSP + time decay)
              → Biased Goal → BFS + A*
              → Monte-Carlo Agent Simulation
    """))

    seed = CONFIG.get("random_seed")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    sim = SearchSimulator(**CONFIG)
    r   = sim.run(run_mc=True)

    # ── STEP 1 — Fuzzy ────────────────────────────────────────
    _s("STEP 1 — Fuzzy Logic Inference (Mamdani + Centroid Defuzz)")
    bd = r.fuzzy_breakdown
    print(f"  Input:")
    print(f"    Height   : {r.height} cm")
    print(f"    Age      : {r.age} years")
    print(f"    Behavior : {r.behavior}")
    print(f"    Clothing : {r.clothing}")

    print(f"\n  Height Membership Sets:")
    for label, mu in bd["height_sets"].items():
        bar = "#" * int(mu * 20)
        print(f"    {label:8s}: mu = {mu:.3f}  {bar}")

    print(f"\n  Age Membership Sets:")
    for label, mu in bd["age_sets"].items():
        bar = "#" * int(mu * 20)
        print(f"    {label:8s}: mu = {mu:.3f}  {bar}")

    print(f"\n  Behavior Memberships (→ output terms):")
    for k, v in bd["behavior_sets"].items():
        print(f"    {k:8s}: {v:.3f}")

    print(f"\n  Clothing Memberships (→ output terms):")
    for k, v in bd["clothing_sets"].items():
        print(f"    {k:8s}: {v:.3f}")

    print(f"\n  Per-feature scalar contributions (display only):")
    print(f"    Height   : {bd['h_score']:.3f}")
    print(f"    Age      : {bd['a_score']:.3f}")
    print(f"    Behavior : {bd['b_score']:.3f}")
    print(f"    Clothing : {bd['c_score']:.3f}")

    print(f"\n  ► Defuzzified Suspicion Score : {r.suspicion_score}")
    print(f"  ► Risk Level                  : {r.risk_level.upper()}")
    print(f"  ► A* heuristic weight         : {r.suspicion_weight:.2f}")

    # ── STEP 2 — CSP / Probability Grid ───────────────────────
    _s("STEP 2 — CSP: Probability Grid + High-Probability Hotspot Zones")

    print(f"  High-probability hotspot zones (risk = '{r.risk_level}'):")
    for i, (z1, z2, z3, z4) in enumerate(r.hp_zones, 1):
        print(f"    Zone {i}: rows [{z1}–{z3}], cols [{z2}–{z4}]  "
              f"← likely location for this risk profile")

    print(f"\n  Constraint Engine:")
    for sc in r.constraint_summary["soft_constraints"]:
        print(f"    Soft — {sc['name']} (weight {sc['weight']:.2f})")
    for hc in r.constraint_summary["hard_constraints"]:
        print(f"    Hard — {hc['name']}")

    print(f"\n  Probability grid (top-3 most probable free cells):")
    g = r.grid
    cells = [
        (float(r.prob_matrix[row, col]), row, col)
        for row in range(g.rows) for col in range(g.cols)
        if g.grid[row][col] == 0
    ]
    for prob, row, col in sorted(cells, reverse=True)[:3]:
        print(f"    Cell ({row},{col}) → P = {prob:.3f}")

    # ── STEP 3 — Grid ─────────────────────────────────────────
    _s("STEP 3 — Grid Construction")
    print(f"  {r.grid}")
    print(f"  Start (fixed) : {r.start}")
    print(f"  Goal (biased) : {r.goal}  ← sampled from probability distribution")
    print()
    r.grid.display()

    # ── STEP 4 — Search ───────────────────────────────────────
    _s("STEP 4 — BFS & A* Search Results")

    bfs_path_list   = list(r.bfs_path)   if r.bfs_path   else None
    astar_path_list = list(r.astar_path) if r.astar_path else None

    print(f"  BFS:")
    print(f"    Nodes explored : {r.bfs_nodes}")
    print(f"    Path length    : {r.bfs_path_len or 'No path'}")
    print(f"    Time           : {r.bfs_time_s:.4f} s")
    if bfs_path_list:
        print(f"    Path           : {' → '.join(str(c) for c in bfs_path_list)}")

    print(f"\n  A* (w = {r.suspicion_weight:.2f}):")
    print(f"    Nodes explored : {r.astar_nodes}")
    print(f"    Path length    : {r.astar_path_len or 'No path'}")
    print(f"    Time           : {r.astar_time_s:.4f} s")
    if astar_path_list:
        print(f"    Path           : {' → '.join(str(c) for c in astar_path_list)}")

    # ── STEP 5 — Comparison ───────────────────────────────────
    _s("STEP 5 — Algorithm Comparison")
    imp    = r.improvement_pct
    winner = "A*" if imp > 0 else ("BFS" if imp < 0 else "Tie")
    print(f"  {'Metric':<25} {'BFS':>10} {'A*':>10}")
    _sep("-", 50)
    print(f"  {'Nodes explored':<25} {r.bfs_nodes:>10} {r.astar_nodes:>10}")
    print(f"  {'Path length':<25} "
          f"{str(r.bfs_path_len):>10} {str(r.astar_path_len):>10}")
    print(f"  {'Time (s)':<25} "
          f"{r.bfs_time_s:>10.4f} {r.astar_time_s:>10.4f}")
    print(f"  {'Goal reached?':<25} "
          f"{'Yes' if r.bfs_path   else 'No':>10} "
          f"{'Yes' if r.astar_path else 'No':>10}")
    _sep("-", 50)
    print(f"  A* explored {abs(imp):.1f}% "
          f"{'fewer' if imp > 0 else 'more'} nodes → {winner} more efficient.")

    # ── STEP 6 — Monte-Carlo ──────────────────────────────────
    _s("STEP 6 — Monte-Carlo Agent Simulation")
    if r.mc_density is not None:
        flat_idx = int(r.mc_density.argmax())
        top_r = flat_idx // r.grid.cols
        top_c = flat_idx  % r.grid.cols
        print(f"  Runs: {CONFIG['mc_runs']}  |  Steps per run: {CONFIG['mc_steps']}")
        print(f"  Highest density cell : ({top_r}, {top_c})  "
              f"→ density index = {r.mc_density[top_r, top_c]:.2f}")

    # ── STEP 7 — Plots ────────────────────────────────────────
    _s("STEP 7 — Visualisation")
    print("  Opening plots (close all to exit)…")

    fig_mf_h = plot_fuzzy_membership(r.height)
    fig_mf_a = plot_fuzzy_age_membership(r.age)
    fig_out  = plot_fuzzy_output(r.fuzzy_breakdown)
    fig_main = plot_comparison(
        grid              = r.grid,
        start             = r.start,
        goal              = r.goal,
        visited_bfs       = r.bfs_visited,
        path_bfs          = list(r.bfs_path)   if r.bfs_path   else None,
        visited_astar     = r.astar_visited,
        path_astar        = list(r.astar_path) if r.astar_path else None,
        restricted_zones  = list(r.hp_zones),
        risk_level        = r.risk_level,
        suspicion_score   = r.suspicion_score,
        prob_matrix       = r.prob_matrix,
        show_prob_overlay = True,
    )

    figs = [fig_mf_h, fig_mf_a, fig_out, fig_main]

    if r.mc_density is not None:
        fig_mc = plot_monte_carlo(
            grid=r.grid, density=r.mc_density, start=r.start,
            n_runs=CONFIG["mc_runs"], steps=CONFIG["mc_steps"],
        )
        figs.append(fig_mc)

    for fig in figs:
        plt.figure(fig.number)
    plt.show()

    _h("Simulation Complete")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Aborted by user]")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
        sys.exit(1)