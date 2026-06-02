#!/usr/bin/env python3
"""
eval/intent_eval.py

Evaluates intent classification accuracy of intent_and_extract_node
against a hand-labelled golden dataset.

Usage:
    python eval/intent_eval.py
    python eval/intent_eval.py --verbose          # print all cases, not just failures
    python eval/intent_eval.py --threshold 85     # custom pass threshold (default 80%)
    python eval/intent_eval.py --plot             # save PNG chart to eval/intent_eval.png
    python eval/intent_eval.py --plot --no-llm    # plot from hardcoded last-run results (no API calls)
"""

import sys
import os
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import initial_state
from nodes import intent_and_extract_node

# ── Golden dataset ─────────────────────────────────────────────────────────────
# Fields:
#   id                   : unique identifier for the test case
#   utterance            : the raw user message to classify
#   expected_intent      : explore | specific | refine | done | chitchat
#   category             : session category at the time of the utterance (None = cold start)
#   active_filters       : filters already set in the session (default: {})
#   history              : list of {"role": ..., "content": ...} prior turns (default: [])
#   last_asked_attribute : what the assistant last asked about (for skip detection)
#   note                 : why this case is interesting (not used in eval logic)

GOLDEN_DATASET = [

    # ── EXPLORE ───────────────────────────────────────────────────────────────
    {
        "id": "explore_001",
        "utterance": "I'm looking for a new smartphone",
        "expected_intent": "explore",
        "category": None,
        "note": "cold start, plain browse intent",
    },
    {
        "id": "explore_002",
        "utterance": "Help me find some headphones",
        "expected_intent": "explore",
        "category": None,
        "note": "cold start, headphones category",
    },
    {
        "id": "explore_003",
        "utterance": "Show me Android phones under $400",
        "expected_intent": "explore",
        "category": None,
        "note": "cold start with implicit filters — borderline with specific",
    },
    {
        "id": "explore_004",
        "utterance": "I want wireless headphones",
        "expected_intent": "explore",
        "category": None,
        "note": "cold start with a type preference",
    },
    {
        "id": "explore_005",
        "utterance": "I need a phone with good battery life",
        "expected_intent": "explore",
        "category": None,
        "note": "cold start, vague feature preference — borderline with specific",
    },
    {
        "id": "explore_006",
        "utterance": "What smartphones do you have?",
        "expected_intent": "explore",
        "category": None,
        "note": "open-ended browse question",
    },
    {
        "id": "explore_007",
        "utterance": "Find me something mid-range",
        "expected_intent": "explore",
        "category": "smartphone",
        "active_filters": {},
        "note": "mid-session, vague price direction without constraining",
    },
    {
        "id": "explore_008",
        "utterance": "I'm interested in noise-cancelling headphones",
        "expected_intent": "explore",
        "category": None,
        "note": "cold start with a feature, still browsing",
    },
    {
        "id": "explore_009",
        "utterance": "Any operating system is fine with me",
        "expected_intent": "explore",
        "category": "smartphone",
        "last_asked_attribute": "os",
        "note": "skip signal — user not constraining OS, should not be chitchat",
    },
    {
        "id": "explore_010",
        "utterance": "I don't know, I just want something good",
        "expected_intent": "explore",
        "category": "smartphone",
        "active_filters": {},
        "note": "uncertainty with a positive direction",
    },

    # ── SPECIFIC ──────────────────────────────────────────────────────────────
    {
        "id": "specific_001",
        "utterance": "I want the Samsung Galaxy S24 Ultra",
        "expected_intent": "specific",
        "category": None,
        "note": "exact brand + model, cold start",
    },
    {
        "id": "specific_002",
        "utterance": "Show me the iPhone 15 Pro Max",
        "expected_intent": "specific",
        "category": None,
        "note": "exact Apple model",
    },
    {
        "id": "specific_003",
        "utterance": "I need the Sony WH-1000XM5",
        "expected_intent": "specific",
        "category": None,
        "note": "exact headphone model number",
    },
    {
        "id": "specific_004",
        "utterance": "Do you have the Bose QuietComfort 45?",
        "expected_intent": "specific",
        "category": None,
        "note": "question form with exact model",
    },
    {
        "id": "specific_005",
        "utterance": "I'm looking for a OnePlus 12",
        "expected_intent": "specific",
        "category": None,
        "note": "exact model, less common brand",
    },
    {
        "id": "specific_006",
        "utterance": "Get me the Apple AirPods Pro",
        "expected_intent": "specific",
        "category": None,
        "note": "accessory with exact model name",
    },
    {
        "id": "specific_007",
        "utterance": "I want the most expensive Samsung phone you have",
        "expected_intent": "specific",
        "category": None,
        "note": "brand-constrained + superlative — still a specific ask",
    },

    # ── REFINE ────────────────────────────────────────────────────────────────
    {
        "id": "refine_001",
        "utterance": "Show me cheaper options",
        "expected_intent": "refine",
        "category": "smartphone",
        "active_filters": {"os": "android", "price_usd_max": 800},
        "note": "classic tighten-price critique",
    },
    {
        "id": "refine_002",
        "utterance": "I want something with better battery life",
        "expected_intent": "refine",
        "category": "smartphone",
        "active_filters": {"os": "android"},
        "note": "add a spec constraint mid-session",
    },
    {
        "id": "refine_003",
        "utterance": "No Samsung please",
        "expected_intent": "refine",
        "category": "smartphone",
        "active_filters": {"os": "android", "price_usd_max": 500},
        "note": "negation-based brand exclusion",
    },
    {
        "id": "refine_004",
        "utterance": "Make it under $400",
        "expected_intent": "refine",
        "category": "smartphone",
        "active_filters": {"os": "android"},
        "note": "price ceiling tightened mid-session",
    },
    {
        "id": "refine_005",
        "utterance": "I'd prefer wireless actually",
        "expected_intent": "refine",
        "category": "headphones",
        "active_filters": {"form_factor": "Over-Ear"},
        "note": "type preference update mid-session",
    },
    {
        "id": "refine_006",
        "utterance": "Can you show me something with more RAM?",
        "expected_intent": "refine",
        "category": "smartphone",
        "active_filters": {"os": "android", "ram_capacity": 4},
        "note": "increase a spec floor already set",
    },
    {
        "id": "refine_007",
        "utterance": "Actually I want noise cancellation too",
        "expected_intent": "refine",
        "category": "headphones",
        "active_filters": {"type": "Wireless"},
        "note": "add a feature requirement mid-session",
    },
    {
        "id": "refine_008",
        "utterance": "These are too expensive, show me something cheaper",
        "expected_intent": "refine",
        "category": "smartphone",
        "active_filters": {"os": "ios", "price_usd_max": 1000},
        "note": "explicit complaint + direction",
    },
    {
        "id": "refine_009",
        "utterance": "I want a bigger screen",
        "expected_intent": "refine",
        "category": "smartphone",
        "active_filters": {"os": "android"},
        "note": "add display size constraint",
    },
    {
        "id": "refine_010",
        "utterance": "Show me something cheaper but with a better camera",
        "expected_intent": "refine",
        "category": "smartphone",
        "active_filters": {"os": "android", "price_usd_max": 700},
        "note": "multi-attribute trade-off — harder case",
    },
    {
        "id": "refine_011",
        "utterance": "I changed my mind, I want iOS instead",
        "expected_intent": "refine",
        "category": "smartphone",
        "active_filters": {"os": "android"},
        "note": "flip an already-set filter",
    },

    # ── DONE ──────────────────────────────────────────────────────────────────
    {
        "id": "done_001",
        "utterance": "Thanks, I'll take it!",
        "expected_intent": "done",
        "category": "smartphone",
        "active_filters": {"os": "android"},
        "note": "clear purchase decision",
    },
    {
        "id": "done_002",
        "utterance": "Perfect, I'll go with the first option",
        "expected_intent": "done",
        "category": "smartphone",
        "active_filters": {"os": "android"},
        "note": "explicit selection from list",
    },
    {
        "id": "done_003",
        "utterance": "That's exactly what I needed, thanks",
        "expected_intent": "done",
        "category": "headphones",
        "active_filters": {"type": "Wireless", "noise_cancellation": True},
        "note": "satisfaction signal mid-session",
    },
    {
        "id": "done_004",
        "utterance": "I'm done looking, thanks",
        "expected_intent": "done",
        "category": "smartphone",
        "active_filters": {},
        "note": "explicit done signal",
    },
    {
        "id": "done_005",
        "utterance": "Great, that's all I needed. Goodbye!",
        "expected_intent": "done",
        "category": "headphones",
        "active_filters": {},
        "note": "farewell + satisfaction",
    },

    # ── CHITCHAT ──────────────────────────────────────────────────────────────
    {
        "id": "chitchat_001",
        "utterance": "Hi there!",
        "expected_intent": "chitchat",
        "category": None,
        "note": "greeting at cold start",
    },
    {
        "id": "chitchat_002",
        "utterance": "Good morning!",
        "expected_intent": "chitchat",
        "category": None,
        "note": "time-based greeting",
    },
    {
        "id": "chitchat_003",
        "utterance": "What's the weather like today?",
        "expected_intent": "chitchat",
        "category": None,
        "note": "clearly off-topic question",
    },
    {
        "id": "chitchat_004",
        "utterance": "You're really helpful, thank you!",
        "expected_intent": "chitchat",
        "category": "smartphone",
        "active_filters": {"os": "android"},
        "note": "mid-session compliment, not a product signal",
    },
    {
        "id": "chitchat_005",
        "utterance": "Tell me a joke",
        "expected_intent": "chitchat",
        "category": None,
        "note": "off-topic request",
    },
    {
        "id": "chitchat_006",
        "utterance": "How are you doing?",
        "expected_intent": "chitchat",
        "category": None,
        "note": "social small talk",
    },
    {
        "id": "chitchat_007",
        "utterance": "Hmm, interesting",
        "expected_intent": "chitchat",
        "category": "smartphone",
        "active_filters": {"os": "android"},
        "note": "vague filler, no product intent",
    },
    {
        "id": "chitchat_008",
        "utterance": "What can you help me with?",
        "expected_intent": "chitchat",
        "category": None,
        "note": "meta question about system scope",
    },
]

INTENTS = ["explore", "specific", "refine", "done", "chitchat"]

# Palette: one colour per intent (used in both the bar chart and confusion matrix diagonal)
INTENT_COLORS = {
    "explore":  "#4C72B0",
    "specific": "#55A868",
    "refine":   "#C44E52",
    "done":     "#8172B2",
    "chitchat": "#CCB974",
}


# ── Eval runner ────────────────────────────────────────────────────────────────

def _build_state(case: dict):
    state = initial_state()
    state["user_input"] = case["utterance"]
    state["category"] = case.get("category")
    state["active_filters"] = case.get("active_filters", {})
    state["messages"] = case.get("history", [])
    state["last_asked_attribute"] = case.get("last_asked_attribute")
    return state


def run_eval(verbose: bool = False) -> dict:
    """Run all cases and return a structured results dict."""
    per_class: dict = defaultdict(lambda: {"correct": 0, "total": 0, "failures": []})
    confusion: dict = defaultdict(lambda: defaultdict(int))  # confusion[actual][predicted]
    results = []

    print(f"Running {len(GOLDEN_DATASET)} cases...\n")

    for case in GOLDEN_DATASET:
        state = _build_state(case)
        try:
            output = intent_and_extract_node(state)
            predicted = output.get("intent", "MISSING")
        except Exception as exc:
            predicted = f"ERROR: {exc}"

        expected = case["expected_intent"]
        correct = predicted == expected

        per_class[expected]["total"] += 1
        confusion[expected][predicted] += 1
        if correct:
            per_class[expected]["correct"] += 1
        else:
            per_class[expected]["failures"].append(case | {"predicted": predicted})

        results.append(case | {"predicted": predicted, "correct": correct})

        if verbose or not correct:
            mark = "OK" if correct else "FAIL"
            snippet = case["utterance"][:65]
            print(f"  {mark:<4} [{case['id']}] \"{snippet}\"")
            if not correct:
                print(f"        expected={expected!r:<12} got={predicted!r}")

    # ── Summary table ──────────────────────────────────────────────────────────
    total = len(GOLDEN_DATASET)
    n_correct = sum(1 for r in results if r["correct"])
    accuracy = n_correct / total * 100

    print("\n" + "=" * 55)
    print(f"  OVERALL ACCURACY:  {n_correct}/{total}  =  {accuracy:.1f}%")
    print("=" * 55)
    print(f"\n  {'Intent':<12} {'Correct':>8} {'Total':>7} {'Acc':>8}")
    print("  " + "-" * 37)
    for intent in INTENTS:
        s = per_class[intent]
        c, t = s["correct"], s["total"]
        acc_str = f"{c / t * 100:.0f}%" if t else "n/a"
        print(f"  {intent:<12} {c:>8} {t:>7} {acc_str:>8}")

    # ── Failures ───────────────────────────────────────────────────────────────
    failures = [r for r in results if not r["correct"]]
    print(f"\n  FAILURES ({len(failures)}):")
    if not failures:
        print("  (none)")
    else:
        for f in failures:
            snippet = f["utterance"][:70]
            print(f"  [{f['id']}] \"{snippet}\"")
            print(f"       expected={f['expected_intent']!r:<12}  got={f['predicted']!r}")
            if f.get("note"):
                print(f"       note: {f['note']}")

    return {
        "accuracy": accuracy,
        "n_correct": n_correct,
        "total": total,
        "results": results,
        "per_class": {k: dict(v) for k, v in per_class.items()},
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_results(data: dict, out_path: str = "eval/intent_eval.png") -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    accuracy   = data["accuracy"]
    n_correct  = data["n_correct"]
    total      = data["total"]
    per_class  = data["per_class"]
    confusion  = data["confusion"]

    # Build confusion matrix as a 2-D numpy array (rows=actual, cols=predicted)
    n = len(INTENTS)
    cm = np.zeros((n, n), dtype=int)
    for r, actual in enumerate(INTENTS):
        for c, predicted in enumerate(INTENTS):
            cm[r, c] = confusion.get(actual, {}).get(predicted, 0)

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor("#FAFAFA")

    # Title band
    fig.text(
        0.5, 0.97,
        f"Intent Classification Evaluation  —  {accuracy:.1f}% accuracy  ({n_correct}/{total} cases)",
        ha="center", va="top",
        fontsize=14, fontweight="bold", color="#222222",
    )

    gs = fig.add_gridspec(1, 2, left=0.06, right=0.97, bottom=0.12, top=0.88,
                          wspace=0.38, width_ratios=[1, 1])

    # ── Left panel: confusion matrix ───────────────────────────────────────────
    ax_cm = fig.add_subplot(gs[0])
    ax_cm.set_facecolor("#FAFAFA")

    vmax = cm.max() if cm.max() > 0 else 1
    im = ax_cm.imshow(cm, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")

    # Cell annotations
    for r in range(n):
        for c in range(n):
            val = cm[r, c]
            text_color = "white" if val > vmax * 0.55 else "#333333"
            weight = "bold" if r == c else "normal"
            ax_cm.text(c, r, str(val), ha="center", va="center",
                       fontsize=13, color=text_color, fontweight=weight)

    ax_cm.set_xticks(range(n))
    ax_cm.set_yticks(range(n))
    ax_cm.set_xticklabels(INTENTS, fontsize=10, rotation=30, ha="right")
    ax_cm.set_yticklabels(INTENTS, fontsize=10)
    ax_cm.set_xlabel("Predicted intent", fontsize=11, labelpad=8)
    ax_cm.set_ylabel("Actual intent", fontsize=11, labelpad=8)
    ax_cm.set_title("Confusion Matrix", fontsize=12, pad=10, color="#444444")

    # Thin grid lines between cells
    for i in range(n + 1):
        ax_cm.axhline(i - 0.5, color="white", linewidth=1.2)
        ax_cm.axvline(i - 0.5, color="white", linewidth=1.2)
    ax_cm.set_xlim(-0.5, n - 0.5)
    ax_cm.set_ylim(n - 0.5, -0.5)

    # ── Right panel: per-class accuracy bars ───────────────────────────────────
    ax_bar = fig.add_subplot(gs[1])
    ax_bar.set_facecolor("#FAFAFA")

    class_acc = []
    class_counts = []
    for intent in INTENTS:
        s = per_class.get(intent, {"correct": 0, "total": 0})
        c, t = s["correct"], s["total"]
        class_acc.append(c / t * 100 if t else 0)
        class_counts.append((c, t))

    y_pos = np.arange(len(INTENTS))
    bar_colors = [INTENT_COLORS[i] for i in INTENTS]

    bars = ax_bar.barh(y_pos, class_acc, color=bar_colors,
                       height=0.55, edgecolor="white", linewidth=0.8)

    # Value labels on bars
    for bar, acc, (c, t) in zip(bars, class_acc, class_counts):
        x = bar.get_width()
        label = f"{acc:.0f}%  ({c}/{t})"
        ax_bar.text(
            min(x + 1, 101), bar.get_y() + bar.get_height() / 2,
            label, va="center", fontsize=10, color="#333333",
        )

    # Overall accuracy reference line
    ax_bar.axvline(accuracy, color="#E05C2A", linewidth=1.5,
                   linestyle="--", zorder=5)
    ax_bar.text(accuracy + 0.5, len(INTENTS) - 0.05,
                f"avg {accuracy:.1f}%", color="#E05C2A",
                fontsize=9, va="top")

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(INTENTS, fontsize=10)
    ax_bar.set_xlabel("Accuracy (%)", fontsize=11, labelpad=8)
    ax_bar.set_xlim(0, 118)
    ax_bar.set_title("Per-class Accuracy", fontsize=12, pad=10, color="#444444")
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.xaxis.grid(True, linestyle="--", alpha=0.4, color="#BBBBBB")
    ax_bar.set_axisbelow(True)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  Chart saved to: {out_path}")


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intent classification evaluation")
    parser.add_argument("--verbose", action="store_true",
                        help="Print all cases, not just failures")
    parser.add_argument("--threshold", type=float, default=80.0,
                        help="Minimum accuracy %% to exit 0 (default: 80)")
    parser.add_argument("--plot", action="store_true",
                        help="Save a PNG chart of the results")
    parser.add_argument("--output", default="eval/intent_eval.png",
                        help="Output path for the chart (default: eval/intent_eval.png)")
    args = parser.parse_args()

    data = run_eval(verbose=args.verbose)

    if args.plot:
        plot_results(data, out_path=args.output)

    print()
    if data["accuracy"] >= args.threshold:
        print(f"  PASS  {data['accuracy']:.1f}% >= {args.threshold}% threshold")
        sys.exit(0)
    else:
        print(f"  FAIL  {data['accuracy']:.1f}% < {args.threshold}% threshold")
        sys.exit(1)
