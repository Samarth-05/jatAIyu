"""
csp.py
======
Constraint Satisfaction & Probability Mapping Module — v2.

Key features:
  • ConstraintEngine — holds multiple named hard + soft constraints,
    returns a per-cell feasibility score ∈ [0, 1].
  • High-probability zones are ATTRACTORS (hotspots), NOT blockers.
    High fuzzy score → zones become HIGH-PROBABILITY hotspots.
  • generate_probability_grid() — builds a float matrix where each
    cell's value reflects constraint scores + fuzzy bias +
    time-since-last-seen decay.  Uses vectorised numpy operations.
  • is_valid_cell() acts as the hard-constraint gate for BFS / A*.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Callable


# ─────────────────────────────────────────────────────────────────
# Input validation helpers
# ─────────────────────────────────────────────────────────────────

def validate_grid_size(rows: int, cols: int) -> None:
    """Raise ValueError if grid dimensions are out of range [5, 30]."""
    if not (5 <= rows <= 30 and 5 <= cols <= 30):
        raise ValueError(
            f"Grid size must be between 5×5 and 30×30. Got {rows}×{cols}."
        )


def validate_height(height_cm: int) -> None:
    """Raise ValueError if height is outside the plausible human range."""
    if not (100 <= height_cm <= 220):
        raise ValueError(
            f"Height must be between 100 and 220 cm. Got {height_cm}."
        )


def validate_age(age: int) -> None:
    """Raise ValueError if age is outside the accepted range."""
    if not (10 <= age <= 90):
        raise ValueError(f"Age must be between 10 and 90. Got {age}.")


# ─────────────────────────────────────────────────────────────────
# Hard constraint: obstacle + boundary check
# ─────────────────────────────────────────────────────────────────

def is_valid_cell(
    grid,
    position: tuple[int, int],
    restricted_zones: list[tuple] | None = None,
) -> bool:
    """
    Hard CSP constraint gate used by BFS and A* during search.

    A cell is valid if:
      1. It lies within grid bounds.
      2. It is not an obstacle (grid[r][c] == 0).
      3. It does not fall inside any explicitly hard-blocked zone.

    Note: high_probability_zones (hotspots) are NOT blocked —
    they attract the search via the probability heatmap.
    """
    r, c = position
    if not (0 <= r < grid.rows and 0 <= c < grid.cols):
        return False
    if grid.grid[r][c] == 1:
        return False
    if restricted_zones:
        for (r1, c1, r2, c2) in restricted_zones:
            if r1 <= r <= r2 and c1 <= c <= c2:
                return False
    return True


# ─────────────────────────────────────────────────────────────────
# Soft constraint descriptor
# ─────────────────────────────────────────────────────────────────

@dataclass
class SoftConstraint:
    """
    A soft constraint that assigns a score ∈ [0, 1] to each grid cell.

    Attributes:
        name     : human-readable label
        weight   : importance weight (normalised to sum=1 across all soft constraints)
        score_fn : callable(row, col, rows, cols) → float ∈ [0, 1]
    """
    name:     str
    weight:   float
    score_fn: Callable[[int, int, int, int], float]


@dataclass
class HardConstraint:
    """
    A hard constraint: returns True (cell allowed) or False (cell infeasible).
    Hard constraints override all soft scores — infeasible cells get score = 0.
    """
    name:     str
    check_fn: Callable[[int, int], bool]


# ─────────────────────────────────────────────────────────────────
# ConstraintEngine
# ─────────────────────────────────────────────────────────────────

class ConstraintEngine:
    """
    Multi-constraint engine combining hard and soft constraints into
    a per-cell feasibility score matrix.

    Semantics:
      • Hard constraints: binary blockers (obstacle, out-of-bounds).
        Cells failing any hard constraint receive score = 0.
      • Soft constraints: weighted attractors. High score = more likely
        the missing person is in that cell.
      • High-probability zones are modelled as soft attractors, NOT hard blocks.

    Usage:
        engine = ConstraintEngine(rows, cols)
        engine.add_soft(SoftConstraint(...))
        engine.add_hard(HardConstraint(...))
        score_matrix = engine.compute_scores()   # ndarray [rows × cols]
    """

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self._soft: list[SoftConstraint] = []
        self._hard: list[HardConstraint] = []

    def add_soft(self, constraint: SoftConstraint) -> "ConstraintEngine":
        """Register a soft constraint. Returns self for chaining."""
        self._soft.append(constraint)
        return self

    def add_hard(self, constraint: HardConstraint) -> "ConstraintEngine":
        """Register a hard constraint. Returns self for chaining."""
        self._hard.append(constraint)
        return self

    def compute_scores(self) -> np.ndarray:
        """
        Compute the feasibility score for every cell (vectorised).

        Algorithm:
          1. For each soft constraint, evaluate score_fn for every cell.
          2. Weighted sum → combined_score ∈ [0, 1].
          3. Any cell failing a hard constraint → score zeroed out.

        Returns:
            ndarray of shape (rows, cols) with values in [0, 1].
        """
        rows, cols = self.rows, self.cols

        if not self._soft:
            scores = np.ones((rows, cols), dtype=float)
        else:
            scores    = np.zeros((rows, cols), dtype=float)
            total_w   = sum(sc.weight for sc in self._soft) or 1.0
            # Build row/col index arrays once
            r_idx, c_idx = np.meshgrid(
                np.arange(rows), np.arange(cols), indexing="ij"
            )
            for sc in self._soft:
                w = sc.weight / total_w
                # Vectorise score_fn over all cells
                layer = np.vectorize(sc.score_fn)(r_idx, c_idx, rows, cols)
                scores += w * layer.astype(float)

        # Apply hard constraints: zero out infeasible cells
        for hc in self._hard:
            mask = np.vectorize(hc.check_fn)(
                np.arange(rows)[:, None], np.arange(cols)[None, :]
            )
            scores[~mask] = 0.0

        return np.clip(scores, 0.0, 1.0)

    def summary(self) -> dict:
        """Return a human-readable summary of active constraints."""
        return {
            "soft_constraints": [
                {"name": sc.name, "weight": sc.weight} for sc in self._soft
            ],
            "hard_constraints": [{"name": hc.name} for hc in self._hard],
        }


# ─────────────────────────────────────────────────────────────────
# Hotspot zone definitions — HIGH-PROBABILITY zones (NOT blocked)
# ─────────────────────────────────────────────────────────────────
# Semantics: nervous person in dark clothing is MORE LIKELY to be
# found in low-visibility / shadowed / edge areas.

_SHADOW_ZONES: list[tuple] = [
    (7, 6, 9, 9),  # Bottom-right corner (dark, shadowed)
    (6, 0, 8, 3),  # Bottom-left service area
    (8, 4, 9, 7),  # Bottom corridor
]

_MEDIUM_ZONES: list[tuple] = [
    (0, 7, 2, 9),  # Top-right (side access)
    (1, 4, 3, 6),  # Central-upper (crowd area)
]

_LOW_ZONES: list[tuple] = [
    (3, 3, 5, 5),  # Central open area — least likely for nervous person
]


def generate_high_probability_zones(
    risk_level: str,
    grid_rows: int = 10,
    grid_cols: int = 10,
) -> list[tuple]:
    """
    Return hotspot rectangles (r1, c1, r2, c2) where the missing person
    is most likely to be, based on risk level.

    high risk   → shadow zones activated (nervous, dark clothing)
    medium risk → shadow + medium zones
    low risk    → medium zones only (person behaving normally)

    These zones are ATTRACTORS for search, not hard blocks.
    """
    if risk_level == "high":
        raw_zones = list(_SHADOW_ZONES)
    elif risk_level == "medium":
        raw_zones = list(_SHADOW_ZONES) + list(_MEDIUM_ZONES)
    else:
        raw_zones = list(_MEDIUM_ZONES)

    # Clip zone coordinates to fit the actual grid size
    clipped = []
    for (r1, c1, r2, c2) in raw_zones:
        r1c = max(0, r1);          c1c = max(0, c1)
        r2c = min(grid_rows - 1, r2); c2c = min(grid_cols - 1, c2)
        if r1c <= r2c and c1c <= c2c:
            clipped.append((r1c, c1c, r2c, c2c))
    return clipped


# Backward-compatibility alias
generate_restricted_zones = generate_high_probability_zones


def get_zone_coverage(
    zones: list[tuple],
    grid_rows: int,
    grid_cols: int,
) -> float:
    """Return the fraction of grid cells covered by hotspot zones (deduplicated)."""
    if not zones:
        return 0.0
    covered: set[tuple] = set()
    for (r1, c1, r2, c2) in zones:
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                covered.add((r, c))
    return round(len(covered) / (grid_rows * grid_cols), 3)


# ─────────────────────────────────────────────────────────────────
# Probability grid builder — vectorised (biased by fuzzy + constraints)
# ─────────────────────────────────────────────────────────────────

def generate_probability_grid(
    grid,
    suspicion_score: float,
    risk_level: str,
    behavior: str,
    clothing: str,
    last_seen: tuple[int, int] | None = None,
    hours_since_seen: float = 0.0,
) -> np.ndarray:
    """
    Build a (rows × cols) probability matrix for the missing person's location.

    Combines (all vectorised):
      1. Base uniform prior
      2. Obstacle mask            — obstacles get probability 0
      3. Hotspot zone bias        — cells in high-prob zones get a boost
      4. Distance decay           — probability falls off from last-seen cell,
                                    amplified by hours_since_seen
      5. Behavior bias            — nervous → corners / edges preferred
      6. Clothing darkness bias   — dark clothing → lower-right quadrant preferred
      7. Normalise to [0, 1]

    Args:
        grid             : Grid object
        suspicion_score  : fuzzy output ∈ [0, 1]
        risk_level       : 'low' | 'medium' | 'high'
        behavior         : witness-reported behavior string
        clothing         : witness-reported clothing color
        last_seen        : (row, col) of last sighting, or None
        hours_since_seen : hours elapsed since person was last seen

    Returns:
        ndarray of shape (rows, cols), values ∈ [0, 1], max = 1.0
    """
    rows, cols = grid.rows, grid.cols

    # Row / column index arrays for vectorised operations
    r_idx = np.arange(rows, dtype=float)[:, None]   # shape (rows, 1)
    c_idx = np.arange(cols, dtype=float)[None, :]   # shape (1, cols)

    # ── 1. Base prior ─────────────────────────────────────────────
    prob = np.full((rows, cols), 0.1, dtype=float)

    # ── 2. Obstacle mask ──────────────────────────────────────────
    obstacle_mask = np.array(grid.grid, dtype=bool)
    prob[obstacle_mask] = 0.0

    # ── 3. Hotspot zone bias ──────────────────────────────────────
    hp_zones = generate_high_probability_zones(risk_level, rows, cols)
    for (r1, c1, r2, c2) in hp_zones:
        zone_slice = np.s_[r1:r2 + 1, c1:c2 + 1]
        free_zone  = ~obstacle_mask[zone_slice]
        prob[zone_slice][free_zone] += suspicion_score * 0.6

    # ── 4. Distance decay from last-seen location ─────────────────
    if last_seen is not None:
        lr, lc = last_seen
        # More time elapsed → decay is weaker (person spread further)
        time_factor = max(0.05, 0.20 - hours_since_seen * 0.003)
        dist = np.sqrt((r_idx - lr) ** 2 + (c_idx - lc) ** 2)
        prob *= np.exp(-time_factor * dist)

    # ── 5. Behavior bias: nervous → edges / corners ───────────────
    beh = behavior.strip().lower()
    if beh in ("nervous", "very nervous"):
        # Distance from nearest edge (0 = on the edge)
        edge_dist  = np.minimum(
            np.minimum(r_idx, (rows - 1) - r_idx),
            np.minimum(c_idx, (cols - 1) - c_idx),
        )
        half_min   = min(rows, cols) / 2.0 + 1e-9
        edge_score = 1.0 - (edge_dist / half_min)
        prob *= (1.0 + suspicion_score * edge_score * 0.4)

    # ── 6. Clothing darkness → lower-right quadrant ───────────────
    dark = {"black", "brown", "blue", "gray"}
    if clothing.strip().lower() in dark:
        # Normalise coordinates to [0, 1]; guard against 1-row/1-col grids
        r_norm = r_idx / max(rows - 1, 1)
        c_norm = c_idx / max(cols - 1, 1)
        bias   = (r_norm + c_norm) / 2.0
        prob  *= (1.0 + suspicion_score * bias * 0.3)

    # ── 7. Re-apply obstacle mask and normalise ───────────────────
    prob[obstacle_mask] = 0.0
    mx = prob.max()
    if mx > 0:
        prob /= mx

    return prob


def generate_biased_goal(
    grid,
    prob_matrix: np.ndarray,
    forbidden: tuple | None = None,
) -> tuple[int, int]:
    """
    Sample a goal cell biased by the probability matrix.

    Higher-probability free cells are more likely to be chosen as the
    simulated target location — making the goal meaningful, not random.

    Args:
        grid        : Grid object
        prob_matrix : (rows × cols) probability array
        forbidden   : cell that must not be chosen (e.g. start cell)

    Returns:
        (row, col) of the sampled goal cell.
    """
    candidates = []
    weights    = []

    for r in range(grid.rows):
        for c in range(grid.cols):
            if grid.grid[r][c] == 0:
                if forbidden and (r, c) == forbidden:
                    continue
                w = float(prob_matrix[r, c])
                if w > 0:
                    candidates.append((r, c))
                    weights.append(w)

    if not candidates:
        return grid.random_free_cell()

    weights_arr  = np.array(weights, dtype=float)
    weights_arr /= weights_arr.sum()
    idx = int(np.random.choice(len(candidates), p=weights_arr))
    return candidates[idx]