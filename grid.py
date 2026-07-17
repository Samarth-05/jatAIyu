"""
grid.py
=======
2-D Grid Environment — v2.

Changes from v1:
  • Support for 8-connected (diagonal) movement via `use_diagonals` flag.
  • `random_free_cell_biased()` — samples using a probability matrix.
  • `get_neighbors()` filters using CSP is_valid_cell when zones provided.
  • `is_valid_cell` imported at module level (no local import inside method).
  • `free_cells` result is cached and invalidated on mutation.
  • `(0, 0)` is always guaranteed free after obstacle placement.
  • `display()` no longer recomputes free_cells twice.
  • Cleaner `__repr__` / `display()`.
"""

from __future__ import annotations
import random
import numpy as np

# Imported at module level to avoid repeated local imports
from csp import is_valid_cell


class Grid:
    """
    A 2-D grid environment for pathfinding simulation.

    Attributes:
        rows (int)           : number of rows
        cols (int)           : number of columns
        grid (list[list])    : 2-D matrix; 0 = free, 1 = obstacle
        obstacle_count (int) : actual obstacles placed
        use_diagonals (bool) : 8-directional movement if True

    Cell conventions:
        0 → free (passable)
        1 → obstacle (impassable)
    """

    _DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    _DIR8 = [(-1, 0), (1, 0), (0, -1), (0, 1),
             (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def __init__(
        self,
        rows: int,
        cols: int,
        obstacle_ratio: float = 0.15,
        use_diagonals: bool = False,
    ):
        if not (0.0 <= obstacle_ratio < 1.0):
            raise ValueError("obstacle_ratio must be in [0, 1)")
        self.rows          = rows
        self.cols          = cols
        self.use_diagonals = use_diagonals
        self.grid          = [[0] * cols for _ in range(rows)]
        self._free_cache: list[tuple] | None = None   # lazy cache
        self._place_obstacles(obstacle_ratio)

    # ── Construction ──────────────────────────────────────────────

    def _place_obstacles(self, ratio: float) -> None:
        """
        Randomly place obstacles, guaranteeing that (0, 0) stays free.
        Invalidates the free-cell cache.
        """
        total     = self.rows * self.cols
        target    = int(total * ratio)
        # Exclude (0, 0) from obstacle candidates
        all_cells = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) != (0, 0)
        ]
        chosen = random.sample(all_cells, min(target, len(all_cells)))
        for r, c in chosen:
            self.grid[r][c] = 1
        self.obstacle_count = len(chosen)
        self._free_cache    = None  # invalidate

    def clear_cell(self, row: int, col: int) -> None:
        """Force a cell to be free (used for start / goal placement)."""
        if self.in_bounds(row, col) and self.grid[row][col] == 1:
            self.grid[row][col] = 0
            self.obstacle_count = max(0, self.obstacle_count - 1)
            self._free_cache = None  # invalidate

    # ── Query ─────────────────────────────────────────────────────

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and self.grid[r][c] == 0

    def is_valid(self, r: int, c: int) -> bool:
        """Alias for is_free."""
        return self.is_free(r, c)

    def get_neighbors(
        self,
        row: int,
        col: int,
        restricted_zones: list[tuple] | None = None,
    ) -> list[tuple]:
        """
        Return free neighbouring cells.

        Applies CSP zone filtering if `restricted_zones` is provided.
        Direction set depends on `use_diagonals` flag.
        """
        directions = self._DIR8 if self.use_diagonals else self._DIR4
        neighbors  = []
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if not self.is_free(nr, nc):
                continue
            if restricted_zones is not None:
                if not is_valid_cell(self, (nr, nc), restricted_zones):
                    continue
            neighbors.append((nr, nc))
        return neighbors

    def free_cells(self) -> list[tuple]:
        """Return list of all free (non-obstacle) cells (cached)."""
        if self._free_cache is None:
            self._free_cache = [
                (r, c)
                for r in range(self.rows)
                for c in range(self.cols)
                if self.grid[r][c] == 0
            ]
        return self._free_cache

    def free_count(self) -> int:
        """Return the number of free cells."""
        return len(self.free_cells())

    def random_free_cell(self, exclude: tuple | None = None) -> tuple[int, int]:
        """Return a uniformly random free cell (optionally excluding one)."""
        cells = [c for c in self.free_cells() if c != exclude]
        if not cells:
            raise RuntimeError("Grid has no free cells available.")
        return random.choice(cells)

    def random_free_cell_biased(
        self,
        prob_matrix: np.ndarray,
        exclude: tuple | None = None,
    ) -> tuple[int, int]:
        """
        Sample a free cell weighted by prob_matrix values.
        Falls back to uniform sampling if all weights are zero.
        """
        candidates, weights = [], []
        for r, c in self.free_cells():
            if (r, c) == exclude:
                continue
            w = float(prob_matrix[r, c])
            if w > 0:
                candidates.append((r, c))
                weights.append(w)

        if not candidates:
            return self.random_free_cell(exclude=exclude)

        wa  = np.array(weights, dtype=float)
        wa /= wa.sum()
        idx = int(np.random.choice(len(candidates), p=wa))
        return candidates[idx]

    def __len__(self) -> int:
        """Total number of cells (rows × cols)."""
        return self.rows * self.cols

    # ── Display ───────────────────────────────────────────────────

    def display(self) -> None:
        """Print grid to stdout: '.' = free, '#' = obstacle."""
        symbols = {0: ".", 1: "#"}
        for row in self.grid:
            print(" ".join(symbols[cell] for cell in row))
        print(
            f"Size: {self.rows}×{self.cols}  |  "
            f"Obstacles: {self.obstacle_count}  |  "
            f"Free: {self.free_count()}  |  "
            f"{'8-dir' if self.use_diagonals else '4-dir'} movement"
        )

    def __repr__(self) -> str:
        return (
            f"Grid(rows={self.rows}, cols={self.cols}, "
            f"obstacles={self.obstacle_count}, "
            f"diagonals={self.use_diagonals})"
        )