"""
utils.py
========
Shared utilities and the SearchSimulator orchestrator — v2.

New in v2:
  • age parameter in SearchSimulator (passed through to fuzzy engine)
  • hours_since_seen parameter (passed to probability grid time-decay)
  • Input validation (grid size, height, age)
  • Consistent random seed handling (random + numpy)
  • SimulationResult is frozen (immutable) for safety
  • _build_constraint_engine uses named inner functions (no closure bugs)
  • average_witness_score() moved here from app.py

SearchSimulator ties together all modules:
  grid → fuzzy → csp (probability grid + goal) → search → results
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from grid  import Grid
from fuzzy import calculate_suspicion, get_risk_level, get_score_breakdown
from csp   import (
    generate_high_probability_zones,
    generate_probability_grid,
    generate_biased_goal,
    get_zone_coverage,
    ConstraintEngine,
    SoftConstraint,
    HardConstraint,
    validate_grid_size,
    validate_height,
    validate_age,
)
from search import run_both, run_monte_carlo


# ─────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SimulationResult:
    """
    All outputs from one simulation run — immutable, passed to visualisation.

    Marking frozen=True prevents accidental mutation and makes the object
    hashable (useful for caching and equality checks).
    """

    # Inputs
    height:   int
    age:      int
    behavior: str
    clothing: str

    # Fuzzy
    fuzzy_breakdown:  dict
    suspicion_score:  float
    risk_level:       str
    suspicion_weight: float

    # Grid
    grid:        Grid
    start:       tuple
    goal:        tuple
    prob_matrix: np.ndarray
    hp_zones:    tuple       # high-probability hotspot zones (tuple for hashability)

    # CSP engine summary
    constraint_summary: dict

    # Search — BFS
    bfs_visited:  frozenset
    bfs_path:     Optional[tuple]
    bfs_nodes:    int
    bfs_path_len: Optional[int]
    bfs_time_s:   float

    # Search — A*
    astar_visited:  frozenset
    astar_path:     Optional[tuple]
    astar_nodes:    int
    astar_path_len: Optional[int]
    astar_time_s:   float

    improvement_pct: float

    # Monte-Carlo (optional)
    mc_density: Optional[np.ndarray] = None


# ─────────────────────────────────────────────────────────────────
# Multi-witness helper (formerly inline in app.py)
# ─────────────────────────────────────────────────────────────────

def average_witness_score(witness_inputs: list[dict]) -> float:
    """
    Run fuzzy inference for each witness and return the mean suspicion score.

    Args:
        witness_inputs: list of dicts with keys height, behavior, clothing, age

    Returns:
        Mean suspicion score ∈ [0.0, 1.0]
    """
    scores = [
        calculate_suspicion(w["height"], w["behavior"], w["clothing"], w["age"])
        for w in witness_inputs
    ]
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────────
# SearchSimulator
# ─────────────────────────────────────────────────────────────────

class SearchSimulator:
    """
    Orchestrates the complete AI pipeline for one simulation run.

    Parameters mirror the Streamlit sidebar / CLI inputs.
    Call run() to execute the full pipeline and receive a SimulationResult.
    """

    def __init__(
        self,
        height:           int   = 175,
        age:              int   = 28,
        behavior:         str   = "nervous",
        clothing:         str   = "black",
        grid_rows:        int   = 10,
        grid_cols:        int   = 10,
        obstacle_ratio:   float = 0.12,
        use_diagonals:    bool  = False,
        random_seed:      Optional[int] = 42,
        last_seen:        Optional[tuple] = None,
        hours_since_seen: float = 0.0,
        mc_runs:          int   = 50,
        mc_steps:         int   = 20,
    ):
        # ── Validate inputs ───────────────────────────────────────
        validate_grid_size(grid_rows, grid_cols)
        validate_height(height)
        validate_age(age)

        self.height           = height
        self.age              = age
        self.behavior         = behavior
        self.clothing         = clothing
        self.grid_rows        = grid_rows
        self.grid_cols        = grid_cols
        self.obstacle_ratio   = obstacle_ratio
        self.use_diagonals    = use_diagonals
        self.random_seed      = random_seed
        self.last_seen        = last_seen
        self.hours_since_seen = hours_since_seen
        self.mc_runs          = mc_runs
        self.mc_steps         = mc_steps

    def run(self, run_mc: bool = False) -> SimulationResult:
        """
        Execute the full pipeline and return a SimulationResult.

        Pipeline:
          1. Seed RNG (random + numpy — consistent with app.py)
          2. Fuzzy inference (height + age + behavior + clothing)
          3. Build grid
          4. Build probability grid (CSP + fuzzy bias + time decay)
          5. Sample biased goal
          6. Build ConstraintEngine
          7. BFS + A*
          8. (optional) Monte-Carlo agent simulation
        """
        # ── 1. Seed ───────────────────────────────────────────────
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

        # ── 2. Fuzzy ──────────────────────────────────────────────
        breakdown        = get_score_breakdown(
            self.height, self.behavior, self.clothing, self.age)
        score            = breakdown["total_score"]
        risk             = breakdown["risk_level"]
        suspicion_weight = round(1.0 + score, 3)  # maps [0,1] → [1.0, 2.0]

        # ── 3. Grid ───────────────────────────────────────────────
        grid = Grid(
            self.grid_rows, self.grid_cols,
            self.obstacle_ratio, self.use_diagonals,
        )
        start = (0, 0)
        grid.clear_cell(*start)  # guaranteed free, but defence in depth

        # ── 4. Probability grid ───────────────────────────────────
        prob_matrix = generate_probability_grid(
            grid             = grid,
            suspicion_score  = score,
            risk_level       = risk,
            behavior         = self.behavior,
            clothing         = self.clothing,
            last_seen        = self.last_seen,
            hours_since_seen = self.hours_since_seen,
        )

        # ── 5. Biased goal ────────────────────────────────────────
        goal = generate_biased_goal(grid, prob_matrix, forbidden=start)
        grid.clear_cell(*goal)

        # ── 6. ConstraintEngine ───────────────────────────────────
        engine             = self._build_constraint_engine(grid, score, risk)
        constraint_summary = engine.summary()

        # High-probability hotspot zones (visualisation — NOT hard blocks)
        hp_zones = generate_high_probability_zones(
            risk, self.grid_rows, self.grid_cols)

        # ── 7. Search ─────────────────────────────────────────────
        # Hotspot zones are ATTRACTORS, not blockers — pass None here.
        results = run_both(
            grid, start, goal,
            restricted_zones = None,
            suspicion_weight = suspicion_weight,
        )

        # ── 8. Monte-Carlo (optional) ─────────────────────────────
        mc_density = None
        if run_mc:
            mc_density = run_monte_carlo(
                grid        = grid,
                start       = start,
                prob_matrix = prob_matrix,
                behavior    = self.behavior,
                clothing    = self.clothing,
                suspicion   = score,
                n_runs      = self.mc_runs,
                steps       = self.mc_steps,
            )

        # Convert mutable collections to immutable types for frozen dataclass
        bfs_path   = tuple(results["bfs_path"])   if results["bfs_path"]   else None
        astar_path = tuple(results["astar_path"]) if results["astar_path"] else None

        return SimulationResult(
            height             = self.height,
            age                = self.age,
            behavior           = self.behavior,
            clothing           = self.clothing,
            fuzzy_breakdown    = breakdown,
            suspicion_score    = score,
            risk_level         = risk,
            suspicion_weight   = suspicion_weight,
            grid               = grid,
            start              = start,
            goal               = goal,
            prob_matrix        = prob_matrix,
            hp_zones           = tuple(hp_zones),
            constraint_summary = constraint_summary,
            bfs_visited        = frozenset(results["bfs_visited"]),
            bfs_path           = bfs_path,
            bfs_nodes          = results["bfs_nodes"],
            bfs_path_len       = results["bfs_path_len"],
            bfs_time_s         = results["bfs_time_s"],
            astar_visited      = frozenset(results["astar_visited"]),
            astar_path         = astar_path,
            astar_nodes        = results["astar_nodes"],
            astar_path_len     = results["astar_path_len"],
            astar_time_s       = results["astar_time_s"],
            improvement_pct    = results["improvement_pct"],
            mc_density         = mc_density,
        )

    def _build_constraint_engine(
        self, grid: Grid, score: float, risk: str
    ) -> ConstraintEngine:
        """
        Construct the multi-constraint engine for this run.

        Soft constraints (attractors — high score = more likely location):
          • Edge proximity bias scaled by suspicion (nervous person → edges)
          • Quadrant bias (dark/nervous → bottom-right)
          • Uniform base (always-on prior)

        Hard constraint:
          • Obstacle cells are always infeasible (score forced to 0)

        Note: high-probability hotspot zones are NOT modelled as hard
        constraints here — they are captured in the probability grid.
        """
        engine = ConstraintEngine(grid.rows, grid.cols)
        rows, cols = grid.rows, grid.cols

        # Hard: obstacle cells are infeasible
        # Use a named function to avoid late-binding closure issues
        grid_ref = grid  # explicit capture

        def obstacle_check(r: int, c: int) -> bool:
            return grid_ref.grid[r][c] == 0

        engine.add_hard(HardConstraint(
            name     = "obstacle_block",
            check_fn = obstacle_check,
        ))

        # Soft: edge proximity (nervous person prefers edges and corners)
        nervous_w = score if self.behavior in ("nervous", "very nervous") else 0.1

        def edge_score(r: int, c: int, R: int, C: int) -> float:
            edge_d = min(r, c, R - 1 - r, C - 1 - c)
            return 1.0 - (edge_d / (min(R, C) / 2.0 + 1e-9))

        engine.add_soft(SoftConstraint(
            name     = "edge_proximity",
            weight   = nervous_w,
            score_fn = edge_score,
        ))

        # Soft: quadrant bias (dark clothing → bottom-right)
        dark    = {"black", "brown", "blue", "gray"}
        cloth_w = score if self.clothing in dark else 0.05

        def quadrant_score(r: int, c: int, R: int, C: int) -> float:
            return ((r / max(R - 1, 1)) + (c / max(C - 1, 1))) / 2.0

        engine.add_soft(SoftConstraint(
            name     = "dark_quadrant",
            weight   = cloth_w,
            score_fn = quadrant_score,
        ))

        # Soft: uniform base (always-on — ensures non-zero prior)
        engine.add_soft(SoftConstraint(
            name     = "base_uniform",
            weight   = 0.3,
            score_fn = lambda r, c, R, C: 1.0,
        ))

        return engine