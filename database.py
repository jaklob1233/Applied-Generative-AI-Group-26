"""
database.py
Loads product CSVs into pandas DataFrames and exposes a filter function.
All filtering logic lives here — no LLM involved.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional

# ── Load data ─────────────────────────────────────────────────────────────────

DATA_DIR = Path("datasets")

_dataframes: Dict[str, pd.DataFrame] = {}

# Headphone CSV has columns with spaces/units. Normalize to snake_case on load
# so filter keys stay clean across categories.
_HEADPHONE_COLUMN_MAP = {
    "Brand": "brand",
    "Model": "model",
    "Type": "type",
    "Connectivity": "connectivity",
    "Freq Low (Hz)": "freq_low_hz",
    "Freq High (Hz)": "freq_high_hz",
    "Cable Length (m)": "cable_length_m",
    "Microphone": "microphone",
    "Noise Cancellation": "noise_cancellation",
    "Foldable": "foldable",
    "Battery (hrs)": "battery_hrs",
    "BT Version": "bt_version",
    "Charging Port": "charging_port",
    "Color": "color",
    "Price (USD)": "price_usd",
    "Release Year": "release_year",
    "Avg Rating": "avg_rating",
    "Form Factor": "form_factor",
}


def load_all() -> None:
    """Call once at startup to load all CSVs."""
    files = {
        "smartphone": "reduced_file_smartphone_500.csv",
        "headphones": "reduced_file_headphones_500.csv",
    }
    for category, filename in files.items():
        path = DATA_DIR / filename
        if not path.exists():
            print(f"  [WARNING] Missing {path} - add your CSV files to the datasets/ folder")
            continue
        df = pd.read_csv(path)
        if category == "headphones":
            df = df.rename(columns=_HEADPHONE_COLUMN_MAP)
        _dataframes[category] = df
        print(f"  [OK] Loaded {len(df)} {category}")

def get_categories() -> List[str]:
    return list(_dataframes.keys())

# ── Filter helpers ─────────────────────────────────────────────────────────────

def _apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply structured filters to a DataFrame.
    Supported filter keys follow the convention:
      <field>          → exact match  (e.g. brand_name="samsung")
      <field>_min      → >= threshold (e.g. battery_capacity_min=4000)
      <field>_max      → <= threshold (e.g. price_usd_max=300)
      <field>_contains → case-insensitive substring (e.g. model_contains="pro")
    Booleans: noise_cancellation=True (CSV strings "True"/"False" handled).
    """
    for key, value in filters.items():
        if value is None:
            continue

        if key.endswith("_min"):
            col = key[:-4]
            if col in df.columns:
                df = df[pd.to_numeric(df[col], errors="coerce") >= float(value)]

        elif key.endswith("_max"):
            col = key[:-4]
            if col in df.columns:
                df = df[pd.to_numeric(df[col], errors="coerce") <= float(value)]

        elif key.endswith("_contains"):
            col = key[:-9]
            if col in df.columns:
                df = df[df[col].astype(str).str.lower().str.contains(str(value).lower(), na=False)]

        else:
            # Exact match (case-insensitive for strings)
            if key in df.columns:
                col_dtype = df[key].dtype
                if col_dtype == bool:
                    df = df[df[key] == bool(value)]
                elif col_dtype == object or isinstance(value, str):
                    # Always treat as string if either the column or the value is a string
                    df = df[df[key].astype(str).str.lower() == str(value).lower()]
                else:
                    df = df[pd.to_numeric(df[key], errors="coerce") == float(value)]

    return df


def retrieve(category: str, filters: Dict[str, Any], limit: int = 10) -> List[Dict]:
    """Return up to `limit` products matching the filters for a given category."""
    if category not in _dataframes:
        return []
    df = _dataframes[category].copy()
    df = _apply_filters(df, filters)
    # Sort by price ascending. Both categories use `price_usd` after the
    # one-time INR→USD conversion on 2026-05-12 (rate 1 INR = 0.010485 USD).
    if "price_usd" in df.columns:
        df = df.sort_values("price_usd")
    return df.head(limit).to_dict("records")


# ── Dialogue: fixed question order per category ───────────────────────────────
#
# The recommender follows this fixed sequence — picking the first attribute the
# user hasn't either answered (via a filter) or explicitly skipped.

QUESTION_ORDER: Dict[str, List[str]] = {
    "smartphone": [
        "os",
        "price_usd",
        "battery_capacity",
        "primary_camera_rear",
        "ram_capacity",
        "internal_memory",
    ],
    "headphones": [
        "type",
        "form_factor",
        "noise_cancellation",
        "price_usd",
    ],
}


def base_attr(filter_key: str) -> str:
    """Strip _min / _max / _contains suffix to get the underlying attribute name."""
    for suffix in ("_min", "_max", "_contains"):
        if filter_key.endswith(suffix):
            return filter_key[: -len(suffix)]
    return filter_key


def next_question(
    category: str,
    filters: Dict[str, Any],
    asked_skipped: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Return the next attribute to ask the user about, or None if all questions
    in QUESTION_ORDER have been answered or explicitly skipped.
    """
    order = QUESTION_ORDER.get(category, [])
    answered = {base_attr(k) for k in (filters or {}).keys()}
    skipped = set(asked_skipped or [])
    for attr in order:
        if attr in answered or attr in skipped:
            continue
        return attr
    return None


# ── Scoring: weighted-feature score (0-100) for picking top recommendations ───

_SMARTPHONE_WEIGHTS = {
    # attribute              (weight, direction)
    "price_usd":            (25, "lower"),
    "primary_camera_rear":  (20, "higher"),
    "battery_capacity":     (20, "higher"),
    "ram_capacity":         (15, "higher"),
    "internal_memory":      (15, "higher"),
    "rating":               (5,  "higher"),
}

_HEADPHONES_WEIGHTS = {
    "price_usd":            (25, "lower"),
    "avg_rating":           (30, "higher"),
    "battery_hrs":          (20, "higher"),
    "freq_range":           (10, "higher"),   # derived: freq_high_hz - freq_low_hz
    "noise_cancellation":   (15, "binary"),
}

_WEIGHTS_BY_CATEGORY = {
    "smartphone": _SMARTPHONE_WEIGHTS,
    "headphones": _HEADPHONES_WEIGHTS,
}


def _is_nan(v: Any) -> bool:
    return isinstance(v, float) and v != v


def _normalize_to_score(values: List[Any], mode: str) -> List[float]:
    """Map values to a 0-100 score using min-max within the candidate set."""
    if mode == "binary":
        return [100.0 if str(v).lower() == "true" else 0.0 for v in values]

    nums: List[float] = []
    for v in values:
        if v is None or _is_nan(v):
            continue
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            continue
    if not nums:
        return [50.0] * len(values)

    vmin, vmax = min(nums), max(nums)
    if vmax == vmin:
        return [50.0] * len(values)

    out: List[float] = []
    for v in values:
        if v is None or _is_nan(v):
            out.append(50.0)
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            out.append(50.0)
            continue
        if mode == "higher":
            out.append(100.0 * (x - vmin) / (vmax - vmin))
        else:  # "lower"
            out.append(100.0 * (vmax - x) / (vmax - vmin))
    return out


def score_candidates(category: str, candidates: List[Dict]) -> List[float]:
    """Return a 0-100 weighted score per candidate (same order as input)."""
    weights = _WEIGHTS_BY_CATEGORY.get(category)
    if not weights or not candidates:
        return [0.0] * len(candidates)

    attr_scores: Dict[str, List[float]] = {}
    for attr, (_, mode) in weights.items():
        if attr == "freq_range":
            values = [
                (c.get("freq_high_hz") or 0) - (c.get("freq_low_hz") or 0)
                for c in candidates
            ]
        else:
            values = [c.get(attr) for c in candidates]
        attr_scores[attr] = _normalize_to_score(values, mode)

    total_weight = sum(w for w, _ in weights.values())
    final: List[float] = []
    for i in range(len(candidates)):
        s = sum(
            attr_scores[a][i] * weights[a][0] / total_weight
            for a in weights
        )
        final.append(s)
    return final


def top_n_by_score(category: str, candidates: List[Dict], n: int = 2) -> List[Dict]:
    """
    Pick the top-n candidates by weighted score. Ties are broken randomly.
    Each returned dict includes a '_score' key (rounded to 1 decimal).
    """
    import random

    if not candidates:
        return []
    scores = score_candidates(category, candidates)
    pairs = list(zip(candidates, scores))
    random.shuffle(pairs)               # random order for ties
    pairs.sort(key=lambda p: p[1], reverse=True)  # stable sort preserves shuffle for ties
    return [{**c, "_score": round(s, 1)} for c, s in pairs[:n]]