#!/usr/bin/env python3
"""
eval/filter_eval.py

Evaluates filter extraction quality of intent_and_extract_node against a
hand-labelled golden dataset. Measures field-level precision, recall, and F1.

Precision = fraction of extracted fields that are correct
Recall    = fraction of expected fields that were extracted
F1        = harmonic mean of precision and recall

Usage:
    python eval/filter_eval.py
    python eval/filter_eval.py --verbose       # show every case, not just failures
    python eval/filter_eval.py --plot          # save PNG to eval/filter_eval.png
    python eval/filter_eval.py --threshold 80  # custom pass threshold on F1 (default 75)
"""

import sys
import os
import argparse
import statistics
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import database
database.load_all()

from state import initial_state
import nodes
from nodes import intent_and_extract_node

# Cap max_tokens for eval runs — extraction responses are small JSON objects.
# Without this, OpenRouter pre-reserves 16 384 tokens per call which exhausts
# the key's credit limit even though the actual output is ~100 tokens.
from langchain_openai import ChatOpenAI
nodes.llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
    max_tokens=512,
)

# ── Vague-term thresholds (loaded once from the actual dataset) ────────────────
_SP = database.vague_term_thresholds("smartphone")
_HP = database.vague_term_thresholds("headphones")

SP_PRICE_P25  = int(round(_SP["price_usd"]["p25"]))   # "cheap smartphone"
SP_PRICE_P75  = int(round(_SP["price_usd"]["p75"]))   # "premium smartphone"
SP_CAM_P75    = int(round(_SP["primary_camera_rear"]["p75"]))  # "good camera"
SP_BAT_P75    = int(round(_SP["battery_capacity"]["p75"]))     # "long battery life"
SP_RAM_P75    = int(round(_SP["ram_capacity"]["p75"]))         # "lots of RAM"

HP_PRICE_P25  = int(round(_HP["price_usd"]["p25"]))   # "affordable headphones"
HP_PRICE_P75  = int(round(_HP["price_usd"]["p75"]))   # "premium headphones"


# ── Golden dataset ─────────────────────────────────────────────────────────────
# Fields:
#   id                  : unique identifier
#   utterance           : raw user message
#   expected_filters    : ground-truth filter dict (values must match exactly
#                         after normalisation — strings lowercased, ints rounded)
#   category            : session category at time of utterance (None = cold start)
#   active_filters      : filters already set before this turn (default: {})
#   last_recommend_stats: stats from previous recommendation set (for relative critiques)
#   group               : category label for reporting
#   note                : why this case is interesting

GOLDEN_DATASET = [

    # ── GROUP 1: Simple single-field filters ───────────────────────────────────
    {
        "id": "simple_001",
        "utterance": "I want an Android phone",
        "expected_filters": {"os": "android"},
        "category": None,
        "group": "simple",
        "note": "single OS filter, cold start",
    },
    {
        "id": "simple_002",
        "utterance": "I'm looking for an iPhone",
        "expected_filters": {"os": "ios"},
        "category": None,
        "group": "simple",
        "note": "brand name implies iOS",
    },
    {
        "id": "simple_003",
        "utterance": "Show me Samsung phones",
        "expected_filters": {"brand_name": "samsung"},
        "category": "smartphone",
        "group": "simple",
        "note": "brand filter",
    },
    {
        "id": "simple_004",
        "utterance": "I want wireless headphones",
        "expected_filters": {"type": "Wireless"},
        "category": None,
        "group": "simple",
        "note": "headphone type filter",
    },
    {
        "id": "simple_005",
        "utterance": "Show me wired headphones",
        "expected_filters": {"type": "Wired"},
        "category": "headphones",
        "group": "simple",
        "note": "explicit wired type",
    },
    {
        "id": "simple_006",
        "utterance": "I want over-ear headphones",
        "expected_filters": {"form_factor": "Over-Ear"},
        "category": "headphones",
        "group": "simple",
        "note": "form factor filter",
    },
    {
        "id": "simple_007",
        "utterance": "I prefer in-ear headphones",
        "expected_filters": {"form_factor": "In-Ear"},
        "category": "headphones",
        "group": "simple",
        "note": "in-ear form factor",
    },
    {
        "id": "simple_008",
        "utterance": "I need headphones with noise cancellation",
        "expected_filters": {"noise_cancellation": True},
        "category": "headphones",
        "group": "simple",
        "note": "boolean feature filter",
    },
    {
        "id": "simple_009",
        "utterance": "I need headphones with a built-in microphone",
        "expected_filters": {"microphone": True},
        "category": "headphones",
        "group": "simple",
        "note": "microphone boolean filter",
    },
    {
        "id": "simple_010",
        "utterance": "Show me Bluetooth headphones",
        "expected_filters": {"connectivity": "Bluetooth"},
        "category": "headphones",
        "group": "simple",
        "note": "connectivity filter",
    },

    # ── GROUP 2: Explicit numeric / range filters ──────────────────────────────
    {
        "id": "range_001",
        "utterance": "I want a phone under $400",
        "expected_filters": {"price_usd_max": 400},
        "category": "smartphone",
        "group": "range",
        "note": "price ceiling — explicit number",
    },
    {
        "id": "range_002",
        "utterance": "My budget is between $200 and $600",
        "expected_filters": {"price_usd_min": 200, "price_usd_max": 600},
        "category": "smartphone",
        "group": "range",
        "note": "explicit price range",
    },
    {
        "id": "range_003",
        "utterance": "I need at least 8GB of RAM",
        "expected_filters": {"ram_capacity_min": 8},
        "category": "smartphone",
        "group": "range",
        "note": "RAM floor — explicit number",
    },
    {
        "id": "range_004",
        "utterance": "I want a phone with at least 5000mAh battery",
        "expected_filters": {"battery_capacity_min": 5000},
        "category": "smartphone",
        "group": "range",
        "note": "battery floor — explicit number with unit",
    },
    {
        "id": "range_005",
        "utterance": "I want 128GB of internal storage",
        "expected_filters": {"internal_memory": 128},
        "category": "smartphone",
        "group": "range",
        "note": "exact storage value",
    },
    {
        "id": "range_006",
        "utterance": "Show me headphones under $200",
        "expected_filters": {"price_usd_max": 200},
        "category": "headphones",
        "group": "range",
        "note": "headphone price ceiling",
    },
    {
        "id": "range_007",
        "utterance": "I want headphones with at least 30 hours of battery",
        "expected_filters": {"battery_hrs_min": 30},
        "category": "headphones",
        "group": "range",
        "note": "battery hours floor",
    },
    {
        "id": "range_008",
        "utterance": "Show me Sony headphones",
        "expected_filters": {"brand": "Sony"},
        "category": "headphones",
        "group": "range",
        "note": "headphone brand (different field than smartphone brand_name)",
    },

    # ── GROUP 3: Vague terms grounded to dataset percentiles ───────────────────
    {
        "id": "vague_001",
        "utterance": "I want a cheap smartphone",
        "expected_filters": {"price_usd_max": SP_PRICE_P25},
        "category": None,
        "group": "vague",
        "note": f"'cheap' → price_usd_max = p25 = {SP_PRICE_P25}",
    },
    {
        "id": "vague_002",
        "utterance": "Show me premium smartphones",
        "expected_filters": {"price_usd_min": SP_PRICE_P75},
        "category": None,
        "group": "vague",
        "note": f"'premium' → price_usd_min = p75 = {SP_PRICE_P75}",
    },
    {
        "id": "vague_003",
        "utterance": "I'm looking for a mid-range phone",
        "expected_filters": {"price_usd_min": SP_PRICE_P25, "price_usd_max": SP_PRICE_P75},
        "category": "smartphone",
        "group": "vague",
        "note": f"'mid-range' → p25={SP_PRICE_P25} to p75={SP_PRICE_P75}",
    },
    {
        "id": "vague_004",
        "utterance": "I want a phone with a good camera",
        "expected_filters": {"primary_camera_rear_min": SP_CAM_P75},
        "category": "smartphone",
        "group": "vague",
        "note": f"'good camera' → primary_camera_rear_min = p75 = {SP_CAM_P75}",
    },
    {
        "id": "vague_005",
        "utterance": "I need a phone with a long battery life",
        "expected_filters": {"battery_capacity_min": SP_BAT_P75},
        "category": "smartphone",
        "group": "vague",
        "note": f"'long battery life' → battery_capacity_min = p75 = {SP_BAT_P75}",
    },
    {
        "id": "vague_006",
        "utterance": "Show me affordable headphones",
        "expected_filters": {"price_usd_max": HP_PRICE_P25},
        "category": None,
        "group": "vague",
        "note": f"'affordable' → price_usd_max = p25 = {HP_PRICE_P25}",
    },

    # ── GROUP 4: Multi-attribute extraction ────────────────────────────────────
    {
        "id": "multi_001",
        "utterance": "I want an Android phone under $400 with at least 8GB RAM",
        "expected_filters": {"os": "android", "price_usd_max": 400, "ram_capacity_min": 8},
        "category": None,
        "group": "multi",
        "note": "3-attribute extraction: OS + price + RAM",
    },
    {
        "id": "multi_002",
        "utterance": "Show me wireless over-ear headphones with noise cancellation",
        "expected_filters": {"type": "Wireless", "form_factor": "Over-Ear", "noise_cancellation": True},
        "category": None,
        "group": "multi",
        "note": "3-attribute headphone extraction",
    },
    {
        "id": "multi_003",
        "utterance": "Wireless over-ear headphones with noise cancellation under $200",
        "expected_filters": {"type": "Wireless", "form_factor": "Over-Ear", "noise_cancellation": True, "price_usd_max": 200},
        "category": "headphones",
        "group": "multi",
        "note": "4-attribute extraction",
    },
    {
        "id": "multi_004",
        "utterance": "I want an iPhone with 256GB storage",
        "expected_filters": {"os": "ios", "internal_memory": 256},
        "category": "smartphone",
        "group": "multi",
        "note": "brand implied OS + storage",
    },
    {
        "id": "multi_005",
        "utterance": "I need a cheap Samsung Android phone",
        "expected_filters": {"brand_name": "samsung", "os": "android", "price_usd_max": SP_PRICE_P25},
        "category": None,
        "group": "multi",
        "note": "vague term + explicit brand + OS",
    },
    {
        "id": "multi_006",
        "utterance": "Sony wireless over-ear headphones with ANC under $300",
        "expected_filters": {
            "brand": "Sony",
            "type": "Wireless",
            "form_factor": "Over-Ear",
            "noise_cancellation": True,
            "price_usd_max": 300,
        },
        "category": "headphones",
        "group": "multi",
        "note": "5-attribute extraction — hardest multi case",
    },

    # ── GROUP 5: Skip signals (should extract nothing) ─────────────────────────
    {
        "id": "skip_001",
        "utterance": "Any OS is fine with me",
        "expected_filters": {},
        "category": "smartphone",
        "last_asked_attribute": "os",
        "group": "skip",
        "note": "skip signal for OS — must not invent a filter",
    },
    {
        "id": "skip_002",
        "utterance": "I don't care about the brand",
        "expected_filters": {},
        "category": "smartphone",
        "last_asked_attribute": "brand_name",
        "group": "skip",
        "note": "explicit skip for brand",
    },
    {
        "id": "skip_003",
        "utterance": "Either wired or wireless is fine",
        "expected_filters": {},
        "category": "headphones",
        "last_asked_attribute": "type",
        "group": "skip",
        "note": "both options acceptable → no filter",
    },
    {
        "id": "skip_004",
        "utterance": "No preference on storage",
        "expected_filters": {},
        "category": "smartphone",
        "last_asked_attribute": "internal_memory",
        "group": "skip",
        "note": "no preference = no filter",
    },

    # ── GROUP 6: Relative critiques (anchored to last_recommend_stats) ─────────
    {
        "id": "relative_001",
        "utterance": "Show me cheaper ones",
        "expected_filters": {"price_usd_max": 220},
        "category": "smartphone",
        "active_filters": {"os": "android"},
        "last_recommend_stats": {
            "price_usd": {"min": 220, "median": 400, "max": 750}
        },
        "group": "relative",
        "note": "'cheaper' → price_usd_max = previous min = 220",
    },
    {
        "id": "relative_002",
        "utterance": "I want something with a better camera",
        "expected_filters": {"primary_camera_rear_min": 64},
        "category": "smartphone",
        "active_filters": {"os": "android", "price_usd_max": 400},
        "last_recommend_stats": {
            "price_usd": {"min": 150, "median": 280, "max": 400},
            "primary_camera_rear": {"min": 12, "median": 48, "max": 64},
        },
        "group": "relative",
        "note": "'better camera' → primary_camera_rear_min = previous max = 64",
    },
    {
        "id": "relative_003",
        "utterance": "Something with a bigger battery",
        "expected_filters": {"battery_capacity_min": 5000},
        "category": "smartphone",
        "active_filters": {"os": "android"},
        "last_recommend_stats": {
            "battery_capacity": {"min": 3000, "median": 4000, "max": 5000}
        },
        "group": "relative",
        "note": "'bigger battery' → battery_capacity_min = previous max = 5000",
    },
    {
        "id": "relative_004",
        "utterance": "Show me more expensive options",
        "expected_filters": {"price_usd_min": 750},
        "category": "smartphone",
        "active_filters": {"os": "android"},
        "last_recommend_stats": {
            "price_usd": {"min": 220, "median": 400, "max": 750}
        },
        "group": "relative",
        "note": "'more expensive' → price_usd_min = previous max = 750",
    },

    # ── GROUP 7: Filter removal ────────────────────────────────────────────────
    {
        "id": "removal_001",
        "utterance": "Actually forget about the brand, any brand is fine",
        "expected_filters": {"brand_name": None},
        "category": "smartphone",
        "active_filters": {"os": "android", "brand_name": "samsung"},
        "group": "removal",
        "note": "explicit filter removal via null",
    },
    {
        "id": "removal_002",
        "utterance": "I changed my mind, I don't care about noise cancellation anymore",
        "expected_filters": {"noise_cancellation": None},
        "category": "headphones",
        "active_filters": {"type": "Wireless", "noise_cancellation": True},
        "group": "removal",
        "note": "remove previously set boolean filter",
    },
]

GROUPS = ["simple", "range", "vague", "multi", "skip", "relative", "removal"]

GROUP_COLORS = {
    "simple":   "#4C72B0",
    "range":    "#55A868",
    "vague":    "#C44E52",
    "multi":    "#8172B2",
    "skip":     "#CCB974",
    "relative": "#64B5CD",
    "removal":  "#777777",
}


# ── State builder ──────────────────────────────────────────────────────────────

def _build_state(case: dict):
    state = initial_state()
    state["user_input"] = case["utterance"]
    state["category"] = case.get("category")
    state["active_filters"] = case.get("active_filters", {})
    state["messages"] = case.get("history", [])
    state["last_asked_attribute"] = case.get("last_asked_attribute")
    state["last_recommend_stats"] = case.get("last_recommend_stats")
    return state


# ── Filter comparison ──────────────────────────────────────────────────────────

def _normalise(val):
    """Normalise a filter value for comparison."""
    if isinstance(val, str):
        return val.strip().lower()
    if isinstance(val, float) and val == int(val):
        return int(val)
    return val


def _values_match(extracted_val, expected_val, tolerance: int = 5) -> bool:
    """
    Compare two filter values with optional numeric tolerance.
    tolerance=5 means ±5 on integer values (used for vague-term cases).
    """
    if expected_val is None:
        return extracted_val is None
    if extracted_val is None:
        return False
    ev = _normalise(extracted_val)
    xv = _normalise(expected_val)
    if isinstance(xv, (int, float)) and isinstance(ev, (int, float)):
        return abs(ev - xv) <= tolerance
    return ev == xv


def _compare_filters(extracted: dict, expected: dict, group: str) -> dict:
    """
    Field-level precision / recall / F1 comparison.
    Uses wider tolerance (±10) for vague/relative groups.
    """
    tolerance = 10 if group in ("vague", "relative") else 1

    # True positives: key in both, value matches
    tp = sum(
        1 for k, v in expected.items()
        if k in extracted and _values_match(extracted[k], v, tolerance)
    )
    # False positives: extracted field not in expected, OR wrong value
    fp = sum(
        1 for k, v in extracted.items()
        if k not in expected or not _values_match(v, expected.get(k), tolerance)
    )
    fn = len(expected) - tp  # expected fields not correctly extracted

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not expected else 0.0)
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    exact     = (fp == 0 and fn == 0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact,
        "tp": tp, "fp": fp, "fn": fn,
    }


# ── Eval runner ────────────────────────────────────────────────────────────────

def run_eval(verbose: bool = False) -> dict:
    results = []
    per_group = defaultdict(lambda: {"precision": [], "recall": [], "f1": [], "exact": 0, "total": 0})

    print(f"Running {len(GOLDEN_DATASET)} filter extraction cases...\n")

    for case in GOLDEN_DATASET:
        state = _build_state(case)
        try:
            output = intent_and_extract_node(state)
            extracted = output.get("extracted_filters", {})
            error = None
        except Exception as exc:
            extracted = {}
            error = str(exc)

        expected = case["expected_filters"]
        metrics = _compare_filters(extracted, expected, case["group"])

        per_group[case["group"]]["precision"].append(metrics["precision"])
        per_group[case["group"]]["recall"].append(metrics["recall"])
        per_group[case["group"]]["f1"].append(metrics["f1"])
        per_group[case["group"]]["total"] += 1
        if metrics["exact_match"]:
            per_group[case["group"]]["exact"] += 1

        result = case | {"extracted": extracted, "metrics": metrics, "error": error}
        results.append(result)

        ok = metrics["exact_match"]
        if verbose or not ok:
            mark = "OK" if ok else "FAIL"
            snippet = case["utterance"][:55]
            print(f"  {mark:<4} [{case['id']}] \"{snippet}\"")
            if not ok or verbose:
                print(f"       expected : {expected}")
                print(f"       extracted: {extracted}")
                if error:
                    print(f"       ERROR    : {error}")
                print(f"       P={metrics['precision']:.2f}  R={metrics['recall']:.2f}  F1={metrics['f1']:.2f}")

    # ── Summary ────────────────────────────────────────────────────────────────
    all_f1        = [r["metrics"]["f1"] for r in results]
    all_precision = [r["metrics"]["precision"] for r in results]
    all_recall    = [r["metrics"]["recall"] for r in results]
    exact_total   = sum(1 for r in results if r["metrics"]["exact_match"])
    mean_f1       = statistics.mean(all_f1)
    mean_p        = statistics.mean(all_precision)
    mean_r        = statistics.mean(all_recall)
    n             = len(GOLDEN_DATASET)

    print("\n" + "=" * 60)
    print(f"  OVERALL  |  Precision: {mean_p:.1%}  Recall: {mean_r:.1%}  F1: {mean_f1:.1%}")
    print(f"           |  Exact match: {exact_total}/{n} = {exact_total/n:.1%}")
    print("=" * 60)
    print(f"\n  {'Group':<10} {'P':>7} {'R':>7} {'F1':>7} {'Exact':>8}")
    print("  " + "-" * 43)
    for g in GROUPS:
        s = per_group[g]
        if not s["total"]:
            continue
        mp = statistics.mean(s["precision"])
        mr = statistics.mean(s["recall"])
        mf = statistics.mean(s["f1"])
        ex = s["exact"]
        t  = s["total"]
        print(f"  {g:<10} {mp:>6.1%} {mr:>7.1%} {mf:>7.1%} {ex:>5}/{t}")

    failures = [r for r in results if not r["metrics"]["exact_match"]]
    print(f"\n  FAILURES ({len(failures)}/{n}):")
    if not failures:
        print("  (none)")
    else:
        for f in failures:
            print(f"  [{f['id']}] \"{f['utterance'][:60]}\"")
            print(f"       expected : {f['expected_filters']}")
            print(f"       extracted: {f['extracted']}")
            print(f"       P={f['metrics']['precision']:.2f}  R={f['metrics']['recall']:.2f}  F1={f['metrics']['f1']:.2f}")

    return {
        "mean_f1": mean_f1,
        "mean_precision": mean_p,
        "mean_recall": mean_r,
        "exact_rate": exact_total / n,
        "exact_total": exact_total,
        "n": n,
        "results": results,
        "per_group": {k: dict(v) for k, v in per_group.items()},
    }


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_results(data: dict, out_path: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    results   = data["results"]
    per_group = data["per_group"]
    mean_p    = data["mean_precision"]
    mean_r    = data["mean_recall"]
    mean_f1   = data["mean_f1"]
    exact_rate = data["exact_rate"]

    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor("#FAFAFA")

    fig.text(
        0.5, 0.97,
        f"Filter Extraction Evaluation  —  F1 {mean_f1:.1%}  |  "
        f"P {mean_p:.1%}  R {mean_r:.1%}  |  Exact {exact_rate:.1%}",
        ha="center", va="top", fontsize=13, fontweight="bold", color="#222222",
    )

    gs = fig.add_gridspec(1, 2, left=0.07, right=0.97,
                          bottom=0.13, top=0.88, wspace=0.38)

    # ── Left: Per-group F1, P, R grouped bars ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#FAFAFA")

    active_groups = [g for g in GROUPS if g in per_group and per_group[g]["total"]]
    y = np.arange(len(active_groups))
    bar_h = 0.25

    for i, (metric, label, color) in enumerate([
        ("precision", "Precision", "#4C72B0"),
        ("recall",    "Recall",    "#55A868"),
        ("f1",        "F1",        "#C44E52"),
    ]):
        vals = [statistics.mean(per_group[g][metric]) for g in active_groups]
        bars = ax1.barh(y + (i - 1) * bar_h, vals, height=bar_h,
                        color=color, label=label, edgecolor="white", linewidth=0.5)

    ax1.set_yticks(y)
    ax1.set_yticklabels(active_groups, fontsize=10)
    ax1.set_xlabel("Score", fontsize=11, labelpad=8)
    ax1.set_xlim(0, 1.18)
    ax1.set_title("Per-group Precision / Recall / F1", fontsize=12, pad=10, color="#444444")
    ax1.legend(fontsize=9, loc="lower right", framealpha=0.8)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.xaxis.grid(True, linestyle="--", alpha=0.4, color="#BBBBBB")
    ax1.set_axisbelow(True)

    # ── Right: P vs R scatter, one dot per case, coloured by group ────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#FAFAFA")

    for g in active_groups:
        group_results = [r for r in results if r["group"] == g]
        xs = [r["metrics"]["recall"] for r in group_results]
        ys = [r["metrics"]["precision"] for r in group_results]
        ax2.scatter(xs, ys, color=GROUP_COLORS[g], s=60, label=g,
                    alpha=0.85, edgecolors="white", linewidth=0.5, zorder=5)

    # Overall mean marker
    ax2.scatter([mean_r], [mean_p], color="#E05C2A", s=140, marker="D",
                zorder=10, label=f"mean (F1={mean_f1:.2f})", edgecolors="white", linewidth=0.8)

    # Iso-F1 curves
    for f1_val in [0.5, 0.75, 0.9]:
        p_vals = np.linspace(f1_val / (2 - f1_val), 1.0, 200)
        r_vals = f1_val * p_vals / (2 * p_vals - f1_val)
        mask = (r_vals >= 0) & (r_vals <= 1)
        ax2.plot(r_vals[mask], p_vals[mask], color="#BBBBBB",
                 linestyle="--", linewidth=0.9, zorder=2)
        ax2.text(r_vals[mask][-1] + 0.01, p_vals[mask][-1],
                 f"F1={f1_val}", fontsize=7, color="#999999", va="center")

    ax2.set_xlabel("Recall", fontsize=11, labelpad=8)
    ax2.set_ylabel("Precision", fontsize=11, labelpad=8)
    ax2.set_title("Precision vs Recall (per case)", fontsize=12, pad=10, color="#444444")
    ax2.set_xlim(-0.05, 1.15)
    ax2.set_ylim(-0.05, 1.15)
    ax2.legend(fontsize=8, framealpha=0.8, loc="lower left")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.xaxis.grid(True, linestyle="--", alpha=0.4, color="#BBBBBB")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4, color="#BBBBBB")
    ax2.set_axisbelow(True)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  Chart saved to: {out_path}")


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter extraction evaluation")
    parser.add_argument("--verbose", action="store_true",
                        help="Print all cases, not just failures")
    parser.add_argument("--threshold", type=float, default=75.0,
                        help="Minimum mean F1 %% to exit 0 (default: 75)")
    parser.add_argument("--plot", action="store_true",
                        help="Save a PNG chart of the results")
    parser.add_argument("--output", default="eval/filter_eval.png",
                        help="Chart output path (default: eval/filter_eval.png)")
    args = parser.parse_args()

    data = run_eval(verbose=args.verbose)

    if args.plot:
        plot_results(data, out_path=args.output)

    print()
    score = data["mean_f1"] * 100
    if score >= args.threshold:
        print(f"  PASS  {score:.1f}% F1 >= {args.threshold}% threshold")
        sys.exit(0)
    else:
        print(f"  FAIL  {score:.1f}% F1 < {args.threshold}% threshold")
        sys.exit(1)
