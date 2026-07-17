"""
search.py
=========
Search Algorithms + Agent Simulation — v2.

Contents:
  • manhattan()            — 4-dir heuristic
  • euclidean()            — 8-dir heuristic
  • reconstruct_path()     — backtrack parent pointers
  • bfs()                  — breadth-first search (with timeout)
  • astar()                — A* with tunable heuristic + weight
  • run_both()             — convenience wrapper; timing measured in single pass
  • MissingPersonAgent     — probabilistic walk with behaviour bias
  • run_monte_carlo()      — N Monte-Carlo simulations → density heatmap
"""

from __future__ import annotations

import heapq
import math
import time
from collections import deque
from typing import Callable

import numpy as np

from csp import is_valid_cell


# ─────────────────────────────────────────────────────────────────
# Heuristics
# ─────────────────────────────────────────────────────────────────

def manhattan(a: tuple, b: tuple) -> float:
    """City-block distance — admissible for 4-connected grids."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a: tuple, b: tuple) -> float:
    """Euclidean distance — admissible for 8-connected grids."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ─────────────────────────────────────────────────────────────────
# Path reconstruction
# ─────────────────────────────────────────────────────────────────

def reconstruct_path(
    parent: dict,
    start: tuple,
    goal: tuple,
) -> list[tuple] | None:
    """
    Trace parent pointers from goal back to start.

    Returns [start, ..., goal] or None if goal is unreachable.
    """
    if goal not in parent and goal != start:
        return None
    path, current = [], goal
    while current != start:
        if current not in parent:
            return None
        path.append(current)
        current = parent[current]
    path.append(start)
    path.reverse()
    return path


# ─────────────────────────────────────────────────────────────────
# BFS
# ─────────────────────────────────────────────────────────────────

def bfs(
    grid,
    start: tuple,
    goal: tuple,
    restricted_zones: list | None = None,
    timeout_s: float = 10.0,
) -> tuple[set, dict]:
    """
    Breadth-First Search with early termination and timeout guard.

    Args:
        grid             : Grid object
        start            : (row, col) start cell
        goal             : (row, col) target cell
        restricted_zones : hard-blocked rectangles (CSP hard constraints)
        timeout_s        : abort after this many wall-clock seconds

    Returns:
        (visited set, parent dict)
    """
    queue   = deque([start])
    visited = {start}
    parent  = {}
    t0      = time.perf_counter()

    while queue:
        if time.perf_counter() - t0 > timeout_s:
            break

        current = queue.popleft()
        if current == goal:
            return visited, parent

        for nb in grid.get_neighbors(current[0], current[1]):
            if nb in visited:
                continue
            if not is_valid_cell(grid, nb, restricted_zones):
                continue
            visited.add(nb)
            parent[nb] = current
            queue.append(nb)

    return visited, parent


# ─────────────────────────────────────────────────────────────────
# A*
# ─────────────────────────────────────────────────────────────────

def astar(
    grid,
    start: tuple,
    goal: tuple,
    restricted_zones: list | None = None,
    suspicion_weight: float = 1.0,
    heuristic: Callable = manhattan,
    timeout_s: float = 10.0,
) -> tuple[set, dict]:
    """
    A* Search: f(n) = g(n) + w·h(n).

    The g_cost dict doubles as the closed-set: a node is "settled"
    once added to g_cost with an optimal cost.  Re-visits are skipped
    efficiently without a separate visited set.

    Args:
        grid              : Grid object
        start             : (row, col) start cell
        goal              : (row, col) target cell
        restricted_zones  : hard-blocked rectangles (CSP)
        suspicion_weight  : heuristic weight w (1.0 = standard A*, >1 = weighted A*)
        heuristic         : callable(a, b) → float
        timeout_s         : abort after this many wall-clock seconds

    Returns:
        (visited set, parent dict)
    """
    open_list = [(0.0, start)]
    parent    = {}
    g_cost    = {start: 0.0}
    visited   = set()
    t0        = time.perf_counter()

    while open_list:
        if time.perf_counter() - t0 > timeout_s:
            break

        _, current = heapq.heappop(open_list)

        # Skip stale heap entries (node already settled at lower cost)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return visited, parent

        for nb in grid.get_neighbors(current[0], current[1]):
            if not is_valid_cell(grid, nb, restricted_zones):
                continue
            if nb in visited:
                continue

            # Step cost: √2 for diagonals, 1 for cardinal moves
            dr   = abs(nb[0] - current[0])
            dc   = abs(nb[1] - current[1])
            step = math.sqrt(2) if (dr + dc == 2) else 1.0

            tentative_g = g_cost[current] + step
            if nb not in g_cost or tentative_g < g_cost[nb]:
                parent[nb] = current
                g_cost[nb] = tentative_g
                h = suspicion_weight * heuristic(nb, goal)
                heapq.heappush(open_list, (tentative_g + h, nb))

    return visited, parent


# ─────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────

def run_both(
    grid,
    start: tuple,
    goal: tuple,
    restricted_zones: list | None = None,
    suspicion_weight: float = 1.0,
    timeout_s: float = 10.0,
) -> dict:
    """
    Run BFS and A* on the same grid and return a unified comparison dict.

    Timing is measured around each single execution (not re-run) so the
    results and timing always correspond to the same computation.

    Heuristic is auto-selected:
      4-dir movement → Manhattan
      8-dir movement → Euclidean
    """
    heuristic = euclidean if grid.use_diagonals else manhattan

    # ── BFS (timed in-place) ──────────────────────────────────────
    t0 = time.perf_counter()
    visited_bfs, parent_bfs = bfs(
        grid, start, goal, restricted_zones, timeout_s)
    t_bfs = round(time.perf_counter() - t0, 6)

    # ── A* (timed in-place) ───────────────────────────────────────
    t0 = time.perf_counter()
    visited_astar, parent_astar = astar(
        grid, start, goal, restricted_zones,
        suspicion_weight, heuristic, timeout_s)
    t_astar = round(time.perf_counter() - t0, 6)

    path_bfs   = reconstruct_path(parent_bfs,   start, goal)
    path_astar = reconstruct_path(parent_astar, start, goal)

    n_bfs   = len(visited_bfs)
    n_astar = len(visited_astar)
    improvement = (
        round((n_bfs - n_astar) / n_bfs * 100, 1) if n_bfs > 0 else 0.0
    )

    return {
        "bfs_visited":    visited_bfs,
        "bfs_parent":     parent_bfs,
        "bfs_path":       path_bfs,
        "bfs_nodes":      n_bfs,
        "bfs_path_len":   len(path_bfs)   if path_bfs   else None,
        "bfs_time_s":     t_bfs,

        "astar_visited":  visited_astar,
        "astar_parent":   parent_astar,
        "astar_path":     path_astar,
        "astar_nodes":    n_astar,
        "astar_path_len": len(path_astar) if path_astar else None,
        "astar_time_s":   t_astar,

        "improvement_pct": improvement,
    }


# ─────────────────────────────────────────────────────────────────
# Missing Person Agent (probabilistic behavioural walk) — v2
# ─────────────────────────────────────────────────────────────────

class MissingPersonAgent:
    """
    Simulates a missing person's movement with behavioural bias.

    Movement model:
      • At each step, candidate free neighbours are scored by the
        probability matrix (amplified by suspicion level).
      • Behavior modifier:
          nervous / very nervous → prefer edge cells (flee response)
          calm / normal          → near-uniform random walk
      • Dark clothing → slight bias toward bottom-right quadrant.
      • The probability matrix pull scales with (1 + suspicion).

    Args:
        grid         : Grid object
        start        : starting cell (row, col)
        prob_matrix  : probability matrix from generate_probability_grid
        behavior     : witness-reported behavior string
        clothing     : witness-reported clothing color
        suspicion    : fuzzy suspicion score [0, 1]
    """

    _DARK_CLOTHING = {"black", "brown", "blue", "gray"}

    def __init__(
        self,
        grid,
        start: tuple[int, int],
        prob_matrix: np.ndarray,
        behavior: str = "normal",
        clothing: str = "gray",
        suspicion: float = 0.5,
    ):
        self.grid        = grid
        self.pos         = start
        self.prob_matrix = prob_matrix
        self.behavior    = behavior.strip().lower()
        self.clothing    = clothing.strip().lower()
        self.suspicion   = suspicion
        self.trace: list[tuple] = [start]

    def step(self) -> tuple[int, int]:
        """Advance the agent by one step. Returns new position."""
        neighbors = self.grid.get_neighbors(self.pos[0], self.pos[1])
        if not neighbors:
            return self.pos  # trapped — stay in place

        rows, cols = self.grid.rows, self.grid.cols
        prob_pull  = 1.0 + self.suspicion   # stronger pull for high suspicion

        weights = []
        for (nr, nc) in neighbors:
            # Base: probability matrix score (amplified by suspicion)
            w = (float(self.prob_matrix[nr, nc]) + 0.01) * prob_pull

            # Nervous: upweight edge cells (flee response)
            if self.behavior in ("nervous", "very nervous"):
                edge_dist  = min(nr, nc, rows - 1 - nr, cols - 1 - nc)
                half_min   = min(rows, cols) / 2.0 + 1e-9
                edge_score = 1.0 - (edge_dist / half_min)
                w *= (1.0 + self.suspicion * edge_score * 0.6)

            # Dark clothing: upweight lower-right quadrant
            if self.clothing in self._DARK_CLOTHING:
                bias = ((nr / max(rows - 1, 1)) + (nc / max(cols - 1, 1))) / 2.0
                w   *= (1.0 + self.suspicion * bias * 0.5)

            weights.append(max(w, 1e-9))

        wa        = np.array(weights, dtype=float)
        wa       /= wa.sum()
        idx       = int(np.random.choice(len(neighbors), p=wa))
        self.pos  = neighbors[idx]
        self.trace.append(self.pos)
        return self.pos

    def run(self, steps: int) -> list[tuple]:
        """Run the agent for `steps` steps. Returns the full trace."""
        for _ in range(steps):
            self.step()
        return self.trace


def run_monte_carlo(
    grid,
    start: tuple[int, int],
    prob_matrix: np.ndarray,
    behavior: str,
    clothing: str,
    suspicion: float,
    n_runs: int = 50,
    steps: int = 20,
) -> np.ndarray:
    """
    Run N independent Monte-Carlo agent simulations.

    Returns a (rows × cols) density matrix where each cell's value is
    the number of times the agent visited that cell, normalised to [0, 1].

    Note: agent positions are always within grid bounds because
    get_neighbors() only returns valid in-bounds free cells.

    Args:
        grid        : Grid object
        start       : agent starting position
        prob_matrix : probability guidance matrix
        behavior    : agent behavior string
        clothing    : agent clothing string
        suspicion   : fuzzy suspicion score
        n_runs      : number of independent simulations
        steps       : steps per simulation

    Returns:
        Normalised density ndarray of shape (rows, cols).
    """
    density = np.zeros((grid.rows, grid.cols), dtype=float)

    for _ in range(n_runs):
        agent = MissingPersonAgent(
            grid, start, prob_matrix, behavior, clothing, suspicion)
        trace = agent.run(steps)
        for (r, c) in trace:
            density[r, c] += 1.0

    mx = density.max()
    if mx > 0:
        density /= mx
    return density