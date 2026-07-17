"""
fuzzy.py
========
Fuzzy Logic Engine — Mamdani-style inference with centroid defuzzification.
v2: Added age input, cleaned rule evaluation, proper 4-antecedent min-inference.

Pipeline:
  1. Fuzzification  — map crisp inputs to membership degrees via trimf / trapmf
  2. Rule Evaluation — fire all IF-THEN rules (height × age × behavior × clothing)
  3. Aggregation    — union (max) of all clipped output MFs → output fuzzy set
  4. Defuzzification — centroid of output fuzzy set → crisp suspicion score

Design note — behavior / clothing antecedents
----------------------------------------------
The behavior and clothing lookup tables (_BEHAVIOR_MAP, _CLOTHING_MAP) encode
memberships **directly in output-term space** {low, medium, high}.  A rule like
  (tall, young, very nervous, black, "high", 1.0)
uses the output term "high" to read the firing strength from BOTH the behavior
map (very nervous → high = 1.0) and the clothing map (black → high = 0.90).
The behavior_term / clothing_term columns in _RULES are kept for human
readability and consistency checks, but the actual inference reads the output-
term bucket.  This is intentional: it collapses the antecedent/consequent pair
into a single lookup, which is equivalent to a simple weighted Mamdani rule.

Output universe: suspicion ∈ [0.0, 1.0]
Output linguistic terms: LOW, MEDIUM, HIGH (overlapping MFs)
"""

from __future__ import annotations
import numpy as np

__version__ = "2.0"
__all__ = [
    "fuzzify_height",
    "fuzzify_age",
    "fuzzify_behavior",
    "fuzzify_clothing",
    "calculate_suspicion",
    "get_risk_level",
    "get_score_breakdown",
    "get_output_mfs",
    "trimf",
    "trapmf",
    "trimf_scalar",
    "trapmf_scalar",
]


# ─────────────────────────────────────────────────────────────────
# Module-level MF parameter constants
# ─────────────────────────────────────────────────────────────────

# Height (cm) MF parameters
_H_SHORT_PARAMS  = (140, 140, 155, 168)   # trapmf
_H_MEDIUM_PARAMS = (155, 168, 182)         # trimf
_H_TALL_PARAMS   = (170, 182, 210, 210)    # trapmf

# Age (years) MF parameters
_A_YOUNG_PARAMS  = (15, 15, 22, 32)        # trapmf
_A_ADULT_PARAMS  = (25, 38, 52)            # trimf
_A_SENIOR_PARAMS = (45, 58, 80, 80)        # trapmf

# Output universe MF parameters
_O_LOW_PARAMS    = (0.0, 0.0, 0.25, 0.45)  # trapmf
_O_MEDIUM_PARAMS = (0.30, 0.50, 0.70)       # trimf
_O_HIGH_PARAMS   = (0.55, 0.75, 1.0, 1.0)  # trapmf


# ─────────────────────────────────────────────────────────────────
# Core MF primitives (vectorised over numpy arrays)
# ─────────────────────────────────────────────────────────────────

def trimf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Triangular membership function — peak at b, zero at a and c."""
    y = np.zeros_like(x, dtype=float)
    left  = (x > a) & (x <= b)
    right = (x > b) & (x < c)
    if b != a:
        y[left]  = (x[left]  - a) / (b - a)
    if c != b:
        y[right] = (c - x[right]) / (c - b)
    return y


def trapmf(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Trapezoidal membership function — flat top from b to c."""
    y = np.zeros_like(x, dtype=float)
    y[(x >= b) & (x <= c)] = 1.0
    left  = (x > a) & (x < b)
    right = (x > c) & (x < d)
    if b != a:
        y[left]  = (x[left]  - a) / (b - a)
    if d != c:
        y[right] = (d - x[right]) / (d - c)
    return y


def trimf_scalar(x: float, a: float, b: float, c: float) -> float:
    """Scalar triangular MF."""
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    return (c - x) / (c - b) if c != b else 1.0


def trapmf_scalar(x: float, a: float, b: float, c: float, d: float) -> float:
    """Scalar trapezoidal MF."""
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b != a else 1.0
    return (d - x) / (d - c) if d != c else 1.0


# ─────────────────────────────────────────────────────────────────
# Output universe & output MFs
# ─────────────────────────────────────────────────────────────────

_OUT_X      = np.linspace(0.0, 1.0, 500)
_OUT_LOW    = trapmf(_OUT_X, *_O_LOW_PARAMS)
_OUT_MEDIUM = trimf (_OUT_X, *_O_MEDIUM_PARAMS)
_OUT_HIGH   = trapmf(_OUT_X, *_O_HIGH_PARAMS)


# ─────────────────────────────────────────────────────────────────
# Fuzzification: HEIGHT (cm)
# ─────────────────────────────────────────────────────────────────

def fuzzify_height(h: float) -> dict[str, float]:
    """
    Fuzzify height into three overlapping sets.
      short  : fully ≤ 155, fades out by 168
      medium : peaks at 168, spans 155–182
      tall   : fully ≥ 182, rises from 170
    """
    return {
        "short":  trapmf_scalar(h, *_H_SHORT_PARAMS),
        "medium": trimf_scalar (h, *_H_MEDIUM_PARAMS),
        "tall":   trapmf_scalar(h, *_H_TALL_PARAMS),
    }


# ─────────────────────────────────────────────────────────────────
# Fuzzification: AGE (years)
# ─────────────────────────────────────────────────────────────────

def fuzzify_age(age: int) -> dict[str, float]:
    """
    Fuzzify age into three overlapping sets.
      young  : teens / early adults — 15–30, peak at 22
      adult  : working age          — 25–50, peak at 38
      senior : older adults         — 45+,   rises from 45

    Interpretation:
      • young + nervous → slightly higher suspicion (impulsive movement)
      • senior + calm   → lower suspicion (more predictable behaviour)
    """
    return {
        "young":  trapmf_scalar(float(age), *_A_YOUNG_PARAMS),
        "adult":  trimf_scalar (float(age), *_A_ADULT_PARAMS),
        "senior": trapmf_scalar(float(age), *_A_SENIOR_PARAMS),
    }


# ─────────────────────────────────────────────────────────────────
# Fuzzification: BEHAVIOR (discrete linguistic variable)
# ─────────────────────────────────────────────────────────────────

# Maps behavior label → output-term memberships {low, medium, high}
_BEHAVIOR_MAP: dict[str, dict[str, float]] = {
    "calm":         {"low": 0.90, "medium": 0.10, "high": 0.00},
    "normal":       {"low": 0.50, "medium": 0.50, "high": 0.00},
    "nervous":      {"low": 0.00, "medium": 0.30, "high": 0.70},
    "very nervous": {"low": 0.00, "medium": 0.00, "high": 1.00},
}


def fuzzify_behavior(b: str) -> dict[str, float]:
    """Return output-term memberships for observed behavior."""
    return _BEHAVIOR_MAP.get(b.strip().lower(), _BEHAVIOR_MAP["normal"])


# ─────────────────────────────────────────────────────────────────
# Fuzzification: CLOTHING COLOR
# ─────────────────────────────────────────────────────────────────

# Maps clothing color → output-term memberships {low, medium, high}
_CLOTHING_MAP: dict[str, dict[str, float]] = {
    "black":  {"low": 0.00, "medium": 0.10, "high": 0.90},
    "brown":  {"low": 0.10, "medium": 0.30, "high": 0.60},
    "blue":   {"low": 0.15, "medium": 0.55, "high": 0.30},
    "gray":   {"low": 0.20, "medium": 0.60, "high": 0.20},
    "green":  {"low": 0.30, "medium": 0.60, "high": 0.10},
    "red":    {"low": 0.40, "medium": 0.50, "high": 0.10},
    "white":  {"low": 0.80, "medium": 0.20, "high": 0.00},
    "yellow": {"low": 0.90, "medium": 0.10, "high": 0.00},
}

# Default for unrecognised colors (neutral prior)
_CLOTHING_DEFAULT: dict[str, float] = {"low": 0.33, "medium": 0.33, "high": 0.33}


def fuzzify_clothing(c: str) -> dict[str, float]:
    """Return output-term memberships for observed clothing color."""
    return _CLOTHING_MAP.get(c.strip().lower(), _CLOTHING_DEFAULT)


# ─────────────────────────────────────────────────────────────────
# Mamdani Rule Base
# ─────────────────────────────────────────────────────────────────
# Format:
#   (height_term, age_term, behavior_label, clothing_label, output_term, weight)
#
# • height_term : "short" | "medium" | "tall"
# • age_term    : "young" | "adult" | "senior" | None (= rule ignores age)
# • behavior_label : human-readable label — used for documentation only;
#   the firing strength is read from fuzzify_behavior()[out_term]
# • clothing_label : similarly for documentation only;
#   the firing strength is read from fuzzify_clothing()[out_term]
# • output_term : "low" | "medium" | "high"
# • weight      : rule confidence ∈ (0, 1]
# ─────────────────────────────────────────────────────────────────

_RULES: list[tuple] = [
    # height    age       behavior         clothing  output    weight
    # ── High-suspicion core rules ────────────────────────────────
    ("tall",   "young",  "very nervous",  "black",  "high",   1.00),
    ("tall",   "young",  "very nervous",  "brown",  "high",   0.92),
    ("tall",   None,     "very nervous",  "black",  "high",   0.88),
    ("tall",   "young",  "nervous",       "black",  "high",   0.85),
    ("medium", "young",  "very nervous",  "black",  "high",   0.80),
    ("tall",   "adult",  "very nervous",  "black",  "high",   0.78),

    # ── Age-specific rules ────────────────────────────────────────
    ("tall",   "young",  "nervous",       "blue",   "medium", 0.75),
    ("medium", "young",  "nervous",       "black",  "medium", 0.72),
    ("tall",   "senior", "nervous",       "black",  "medium", 0.68),
    ("medium", "adult",  "very nervous",  "brown",  "high",   0.70),
    ("short",  "young",  "very nervous",  "black",  "medium", 0.62),
    ("medium", "senior", "very nervous",  "black",  "medium", 0.58),

    # ── Medium suspicion ──────────────────────────────────────────
    ("tall",   None,     "normal",        "black",  "medium", 0.65),
    ("medium", None,     "nervous",       "brown",  "medium", 0.60),
    ("tall",   None,     "nervous",       "blue",   "medium", 0.55),
    ("medium", None,     "normal",        "gray",   "medium", 0.45),
    ("short",  "adult",  "nervous",       "gray",   "medium", 0.40),
    ("tall",   None,     "normal",        "blue",   "medium", 0.50),
    ("medium", "young",  "normal",        "blue",   "medium", 0.42),

    # ── Low suspicion ─────────────────────────────────────────────
    ("tall",   "senior", "calm",          "white",  "low",    0.30),
    ("medium", None,     "calm",          "white",  "low",    0.20),
    ("short",  None,     "calm",          "yellow", "low",    0.10),
    ("short",  None,     "calm",          "white",  "low",    0.10),
    ("medium", None,     "normal",        "white",  "low",    0.25),
    ("short",  None,     "normal",        "red",    "low",    0.20),
    ("medium", "senior", "calm",          "gray",   "low",    0.22),
]

_OUT_MFS = {"low": _OUT_LOW, "medium": _OUT_MEDIUM, "high": _OUT_HIGH}


def _evaluate_rules(
    h_sets: dict[str, float],
    a_sets: dict[str, float],
    b_sets: dict[str, float],
    c_sets: dict[str, float],
) -> np.ndarray:
    """
    Fire all rules via Mamdani min-inference; aggregate with max (union).

    Each rule antecedent strength is:
      μ = min(μ_height, μ_age, μ_behavior[out_term], μ_clothing[out_term]) × weight

    Since _BEHAVIOR_MAP and _CLOTHING_MAP already encode memberships in
    output-term space {low, medium, high}, we read the output-term bucket
    directly.  This is equivalent to saying "the behavior/clothing variable
    contributes its measured affinity toward the stated output term."

    Returns the aggregated output fuzzy set over _OUT_X.
    """
    agg = np.zeros_like(_OUT_X)

    for (h_term, a_term, _b_label, _c_label, out_term, weight) in _RULES:
        mu_h = h_sets.get(h_term, 0.0)

        # age_term = None → rule fires regardless of age
        mu_a = a_sets.get(a_term, 0.0) if a_term is not None else 1.0

        # behavior and clothing contribute their output-term affinity
        mu_b = b_sets.get(out_term, 0.0)
        mu_c = c_sets.get(out_term, 0.0)

        antecedent = min(mu_h, mu_a, mu_b, mu_c) * weight

        if antecedent < 1e-9:
            continue

        # Mamdani implication: clip output MF, then union-aggregate
        agg = np.maximum(agg, np.minimum(antecedent, _OUT_MFS[out_term]))

    return agg


def _centroid_defuzz(x: np.ndarray, y: np.ndarray) -> float:
    """Centroid defuzzification. Falls back to 0.5 if the set is empty."""
    denom = np.sum(y)
    if denom < 1e-9:
        return 0.5
    return float(np.sum(x * y) / denom)


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

def calculate_suspicion(
    height_cm: int,
    behavior: str,
    clothing: str,
    age: int = 28,
) -> float:
    """
    Full Mamdani fuzzy inference → centroid defuzzification.

    Args:
        height_cm : witness-reported height in cm
        behavior  : witnessed behavior string
        clothing  : clothing color string
        age       : witness-estimated age in years

    Returns:
        Crisp suspicion score ∈ [0.0, 1.0], rounded to 2 dp.
    """
    h_sets = fuzzify_height(height_cm)
    a_sets = fuzzify_age(age)
    b_sets = fuzzify_behavior(behavior)
    c_sets = fuzzify_clothing(clothing)
    agg    = _evaluate_rules(h_sets, a_sets, b_sets, c_sets)
    score  = _centroid_defuzz(_OUT_X, agg)
    return round(float(np.clip(score, 0.0, 1.0)), 2)


def get_risk_level(score: float) -> str:
    """
    Map crisp suspicion score → linguistic risk category.
      [0.00, 0.35) → 'low'
      [0.35, 0.60) → 'medium'
      [0.60, 1.00] → 'high'
    """
    if score < 0.35:
        return "low"
    if score < 0.60:
        return "medium"
    return "high"


def get_score_breakdown(
    height_cm: int,
    behavior: str,
    clothing: str,
    age: int = 28,
) -> dict:
    """
    Full breakdown of every fuzzy intermediate value.

    Returns a dict with keys:
      height_sets, age_sets, behavior_sets, clothing_sets — fuzzified inputs
      h_score, a_score, b_score, c_score                  — scalar contributions
      agg_x, agg_y                                        — aggregated output MF
      total_score                                          — defuzzified crisp value
      risk_level                                           — linguistic risk label
    """
    h_sets = fuzzify_height(height_cm)
    a_sets = fuzzify_age(age)
    b_sets = fuzzify_behavior(behavior)
    c_sets = fuzzify_clothing(clothing)

    agg   = _evaluate_rules(h_sets, a_sets, b_sets, c_sets)
    score = round(float(np.clip(_centroid_defuzz(_OUT_X, agg), 0.0, 1.0)), 2)
    risk  = get_risk_level(score)

    # Per-feature scalar contributions (for display / bar charts)
    # Weight: short/young/low → 0.0 · medium/adult/medium → 0.5 · tall/senior/high → 1.0
    h_scalar = 0.0 * h_sets["short"]  + 0.5 * h_sets["medium"]  + 1.0 * h_sets["tall"]
    a_scalar = 0.0 * a_sets["young"]  + 0.5 * a_sets["adult"]   + 1.0 * a_sets["senior"]
    b_scalar = 0.0 * b_sets["low"]    + 0.5 * b_sets["medium"]  + 1.0 * b_sets["high"]
    c_scalar = 0.0 * c_sets["low"]    + 0.5 * c_sets["medium"]  + 1.0 * c_sets["high"]

    return {
        "height_sets":   h_sets,
        "age_sets":      a_sets,
        "behavior_sets": b_sets,
        "clothing_sets": c_sets,
        "h_score":       round(h_scalar, 3),
        "a_score":       round(a_scalar, 3),
        "b_score":       round(b_scalar, 3),
        "c_score":       round(c_scalar, 3),
        "agg_x":         _OUT_X,
        "agg_y":         agg,
        "total_score":   score,
        "risk_level":    risk,
    }


def get_output_mfs() -> dict[str, np.ndarray]:
    """Return output MFs for plotting (used in visualization.py)."""
    return {
        "x":      _OUT_X,
        "low":    _OUT_LOW,
        "medium": _OUT_MEDIUM,
        "high":   _OUT_HIGH,
    }