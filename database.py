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

def load_all() -> None:
    """Call once at startup to load all CSVs."""
    files = {
        "smartphone": "reduced_file_smartphone_500.csv",
        "headphones": "reduced_file_headphones_500.csv",
    }
    for category, filename in files.items():
        path = DATA_DIR / filename
        if path.exists():
            _dataframes[category] = pd.read_csv(path)
            print(f"  [OK] Loaded {len(_dataframes[category])} {category}s")
        else:
            print(f"  [WARNING] Missing {path} - add your CSV files to the datasets/ folder")

def get_categories() -> List[str]:
    return list(_dataframes.keys())

# ── Filter helpers ─────────────────────────────────────────────────────────────

def _apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply structured filters to a DataFrame.
    Supported filter keys follow the convention:
      <field>          → exact match  (e.g. brand="Samsung")
      <field>_min      → >= threshold (e.g. battery_mah_min=4000)
      <field>_max      → <= threshold (e.g. price_eur_max=500)
      <field>_contains → case-insensitive substring (e.g. model_contains="pro")
    Boolean fields: has_5g=True
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
    # Sort by price ascending by default
    if "price_eur" in df.columns:
        df = df.sort_values("price_eur")
    return df.head(limit).to_dict("records")


def most_discriminative_attribute(category: str, filters: Dict[str, Any]) -> Optional[str]:
    """
    Among unfiltered attributes, return the one whose values are most varied
    in the current candidate set — i.e. the best next question to ask.
    """
    if category not in _dataframes:
        return None

    df = _dataframes[category].copy()
    df = _apply_filters(df, filters)

    if df.empty:
        return None

    # Candidate attributes to ask about (categorical / low-cardinality)
    ASKABLE: Dict[str, List[str]] = {
        "smartphone":     ["brand", "os", "price_eur", "ram_gb", "storage_gb", "has_5g"],
        "washing_machine": ["brand", "capacity_kg", "energy_class", "load_type", "has_steam"],
        "laptop":         ["brand", "os", "category", "ram_gb", "gpu_type", "price_eur"],
    }
    candidates = ASKABLE.get(category, [])

    # Filter out already-constrained attributes
    already_set = set()
    for key in filters:
        base = key.replace("_min", "").replace("_max", "").replace("_contains", "")
        already_set.add(base)

    best_attr = None
    best_nunique = 0
    for attr in candidates:
        if attr in already_set:
            continue
        if attr not in df.columns:
            continue
        n = df[attr].nunique()
        if n > best_nunique:
            best_nunique = n
            best_attr = attr

    return best_attr