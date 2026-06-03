"""
app.py
Streamlit chat interface for the Conversational Recommender System.
Run with: streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import database
from state import initial_state
from graph import run_turn

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Product Assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main chat area */
    .main .block-container { padding-top: 2rem; }

    /* Filter tag pill */
    .filter-pill {
        display: inline-block;
        background: #e3f2fd;
        color: #1565c0;
        border-radius: 12px;
        padding: 2px 10px;
        margin: 2px;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)

# ── Load data once ────────────────────────────────────────────────────────────

@st.cache_resource
def load_database():
    database.load_all()
    return True

load_database()

# ── Session state ─────────────────────────────────────────────────────────────

if "dialogue_state" not in st.session_state:
    st.session_state.dialogue_state = initial_state()

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []  # [{role, content}]

if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True

# ── Sidebar: live dialogue state inspector ────────────────────────────────────

with st.sidebar:
    st.title("🔍 Session State")
    st.caption("Live view of what the system knows about your preferences")

    ds = st.session_state.dialogue_state

    # Category
    cat = ds.get("category") or "—"
    st.metric("Category", cat.replace("_", " ").title())

    # Turn count
    st.metric("Turn", ds.get("turn_count", 0))

    # Intent
    intent = ds.get("intent") or "—"
    st.metric("Last Intent", intent)

    # Active filters
    st.subheader("Active Filters")
    filters = ds.get("active_filters", {})
    if filters:
        PRICE_LABELS = {
            "price_usd_min": "Price min ($)",
            "price_usd_max": "Price max ($)",
        }
        for k, v in filters.items():
            label = PRICE_LABELS.get(k, k.replace("_", " ").title())
            st.markdown(f'<span class="filter-pill">**{label}**: {v}</span>', unsafe_allow_html=True)
    else:
        st.caption("No filters yet")

    # Candidates count
    n_candidates = len(ds.get("candidates", []))
    if n_candidates > 0:
        st.metric("Matching Products", n_candidates)

    # Last action
    action = ds.get("action") or "—"
    action_colors = {
        "recommend":        "🟢",
        "ask_clarification": "🟡",
        "ask_category":     "🔵",
        "no_results":       "🔴",
        "done":             "⚫",
    }
    icon = action_colors.get(action, "⚪")
    st.caption(f"Last action: {icon} {action}")

    st.divider()

    # Reset button
    if st.button("🔄 Start New Conversation", use_container_width=True):
        st.session_state.dialogue_state = initial_state()
        st.session_state.chat_messages = []
        st.session_state.show_welcome = True
        st.rerun()

    # Debug expander
    with st.expander("🛠 Raw State (debug)"):
        import json
        debug_state = {k: v for k, v in ds.items() if k != "messages"}
        st.code(json.dumps(debug_state, indent=2, default=str), language="json")

# ── Comparison table renderer ───────────────────────────────────────────────

def _render_comparison_table(products, category):
    """Render a side-by-side spec comparison of the top recommended products."""
    import pandas as pd

    if not products:
        return

    def cell(p, key, fmt=str):
        v = p.get(key)
        if v is None or (isinstance(v, float) and v != v):  # None or NaN
            return "—"
        try:
            return fmt(v)
        except Exception:
            return "—"

    def truthy(v):
        return str(v).lower() == "true"

    def freq_range(p):
        lo, hi = p.get("freq_low_hz"), p.get("freq_high_hz")
        bad = lambda v: v is None or (isinstance(v, float) and v != v)
        if bad(lo) or bad(hi):
            return "—"
        return f"{int(lo)}–{int(hi)} Hz"

    if category == "smartphone":
        rows = [
            ("Brand",        lambda p: cell(p, "brand_name", lambda v: str(v).title())),
            ("Model",        lambda p: cell(p, "model")),
            ("Price",        lambda p: cell(p, "price_usd", lambda v: f"${int(v):,}")),
            ("OS",           lambda p: cell(p, "os", lambda v: str(v).upper())),
            ("RAM",          lambda p: cell(p, "ram_capacity", lambda v: f"{int(v)} GB")),
            ("Storage",      lambda p: cell(p, "internal_memory", lambda v: f"{int(v)} GB")),
            ("Battery",      lambda p: cell(p, "battery_capacity", lambda v: f"{int(v)} mAh")),
            ("Rear Camera",  lambda p: cell(p, "primary_camera_rear", lambda v: f"{int(v)} MP")),
            ("Front Camera", lambda p: cell(p, "primary_camera_front", lambda v: f"{int(v)} MP")),
            ("Screen",       lambda p: cell(p, "screen_size", lambda v: f'{v}"')),
            ("Score",        lambda p: f"{p.get('_score', '—')} / 100"),
        ]
        def name(p):
            return f"{str(p.get('brand_name', '')).title()} {p.get('model', '')}".strip()
    elif category == "headphones":
        rows = [
            ("Brand",              lambda p: cell(p, "brand")),
            ("Model",              lambda p: cell(p, "model")),
            ("Price",              lambda p: cell(p, "price_usd", lambda v: f"${int(v)}")),
            ("Type",               lambda p: cell(p, "type")),
            ("Form Factor",        lambda p: cell(p, "form_factor")),
            ("Connectivity",       lambda p: cell(p, "connectivity")),
            ("Noise Cancellation", lambda p: "Yes" if truthy(p.get("noise_cancellation")) else "No"),
            ("Microphone",         lambda p: "Yes" if truthy(p.get("microphone")) else "No"),
            ("Foldable",           lambda p: "Yes" if truthy(p.get("foldable")) else "No"),
            ("Battery (hrs)",      lambda p: cell(p, "battery_hrs", lambda v: f"{v}") if str(p.get("type", "")).lower() == "wireless" else "—"),
            ("Freq Range",         freq_range),
            ("Avg Rating",         lambda p: cell(p, "avg_rating", lambda v: f"{v} / 5")),
            ("Release Year",       lambda p: cell(p, "release_year", lambda v: str(int(v)))),
            ("Score",              lambda p: f"{p.get('_score', '—')} / 100"),
        ]
        def name(p):
            return f"{p.get('brand', '')} {p.get('model', '')}".strip()
    else:
        return

    # Column headers from brand+model; disambiguate identical names
    headers = [name(p) or "Product" for p in products]
    if len(set(headers)) < len(headers):
        headers = [f"{h} ({i+1})" for i, h in enumerate(headers)]

    data = {col: [fn(p) for _, fn in rows] for col, p in zip(headers, products)}
    df = pd.DataFrame(data, index=[label for label, _ in rows])
    st.table(df)


def _product_header(p, category):
    """One-line summary used as the expander title for a single product."""
    score = p.get("_score", "—")
    if category == "smartphone":
        brand = str(p.get("brand_name", "")).title()
        model = p.get("model", "")
        price = p.get("price_usd")
        price_str = f"${int(price):,}" if isinstance(price, (int, float)) and price == price else "—"
    elif category == "headphones":
        brand = p.get("brand", "")
        model = p.get("model", "")
        price = p.get("price_usd")
        price_str = f"${int(price)}" if isinstance(price, (int, float)) and price == price else "—"
    else:
        brand, model, price_str = "", "", "—"
    return f"{brand} {model}  ·  {price_str}  ·  ⭐ {score} / 100".strip()


_SCORE_ATTR_LABELS = {
    "price_usd":           "💰 Price",
    "primary_camera_rear": "📸 Rear Camera",
    "battery_capacity":    "🔋 Battery",
    "ram_capacity":        "🧠 RAM",
    "internal_memory":     "💾 Storage",
    "rating":              "⭐ Rating",
    "avg_rating":          "⭐ Avg Rating",
    "battery_hrs":         "🔋 Battery Life",
    "freq_range":          "🎵 Freq Range",
    "noise_cancellation":  "🔇 Noise Cancel.",
}


def _format_breakdown_value(attr, raw, direction):
    """Pretty-print a raw value with the right unit for the breakdown table."""
    if raw is None:
        return "—"
    if direction == "binary":
        return "Yes" if raw == 1.0 else "No"
    if attr == "price_usd":
        return f"${int(raw):,}"
    if attr in ("battery_capacity",):
        return f"{int(raw)} mAh"
    if attr in ("ram_capacity", "internal_memory"):
        return f"{int(raw)} GB"
    if attr == "primary_camera_rear":
        return f"{int(raw)} MP"
    if attr == "battery_hrs":
        return f"{raw} hrs"
    if attr == "freq_range":
        return f"{int(raw)} Hz"
    if attr in ("rating", "avg_rating"):
        return f"{raw}"
    return str(raw)


def _render_score_breakdown(rows, total_count):
    """Render a per-attribute breakdown markdown table for one product."""
    if not rows:
        return
    md = ["| Factor | Value | Score (0-100) | Weight |", "|---|---|---|---|"]
    for r in rows:
        label = _SCORE_ATTR_LABELS.get(r["attr"], r["attr"])
        value = _format_breakdown_value(r["attr"], r["raw"], r["direction"])
        norm = r["norm_0_100"]
        marker = "🟢" if norm >= 75 else "🟡" if norm >= 40 else "🔴"
        md.append(f"| {label} | {value} | {marker} {norm:.0f} | {r['weight_pct']:.0f}% |")
    st.markdown("\n".join(md))
    st.caption(
        f"_Top score uses **TOPSIS** — closeness to the ideal (best specs, lowest price) "
        f"across the {total_count} matches. The Score column here shows where this product "
        f"stands in the **whole catalog** for each attribute (100 = best in the catalog, "
        f"0 = worst, 50 = median or missing)._"
    )


def _render_recommendation_list(products, category, msg_index):
    """
    Browse-and-compare view: every matching product as an expandable row,
    with a 'compare' checkbox inside each. A live side-by-side comparison
    table appears below when 2-3 items are checked.
    """
    if not products:
        return

    total_matches = len(products)
    st.caption(
        f"**{total_matches} matching {category}** — ranked by **TOPSIS** "
        "(closeness to the ideal mix of high specs and low price). "
        "Expand any item to see its score breakdown."
    )

    # Top-N selector: lets the user trim the list to the strongest few picks.
    # "All" preserves the full ranked browse. Defaults to a manageable cap
    # when the result set is large — keeps the page snappy (rendering 100+
    # expanders triggers a noticeable lag in Streamlit on every click).
    options = [n for n in (3, 5, 10, 20) if n < total_matches] + ["All"]
    default_label = "All" if total_matches <= 10 else 10
    if default_label not in options:
        default_label = options[0]
    selected_label = st.selectbox(
        "Show top",
        options,
        index=options.index(default_label),
        key=f"topn_{msg_index}",
        help="Limit the list to the strongest N picks; choose 'All' to see every match.",
    )
    if selected_label == "All":
        display_products = products
    else:
        display_products = products[: int(selected_label)]

    # Compute the per-product breakdown ONCE for the FULL set so the per-attribute
    # market-positioning scores are stable regardless of the display cap.
    breakdowns = database.score_breakdown(category, products)

    selected_indices = []
    for idx, p in enumerate(display_products):
        with st.expander(_product_header(p, category), expanded=(idx < 3)):
            # Compare checkbox at top — visible without scrolling specs
            chk_key = f"cmp_chk_{msg_index}_{idx}"
            if st.checkbox(
                "📊 Add to comparison",
                key=chk_key,
                help="Select 2-3 items to see a side-by-side comparison below.",
            ):
                selected_indices.append(idx)
            # Single-product spec table (reuses the comparison renderer)
            _render_comparison_table([p], category)
            # Score breakdown — shows exactly what drove this product's score
            st.markdown("**📊 Why this score?**")
            _render_score_breakdown(breakdowns[idx], total_count=len(products))

    # Live comparison area
    if len(selected_indices) >= 2:
        st.divider()
        # Take up to 3 if user checked more
        compare_indices = selected_indices[:3]
        if len(selected_indices) > 3:
            st.info(
                f"Comparing the first 3 of {len(selected_indices)} selected items. "
                "Uncheck some to compare a different combination."
            )
        st.caption(f"**Comparing {len(compare_indices)} items:**")
        _render_comparison_table([products[i] for i in compare_indices], category)
    elif len(selected_indices) == 1:
        st.caption("_Select 1-2 more items to compare._")


# ── Quick-reply suggestions for clarification questions ─────────────────────

# Each label is sent back through the LLM pipeline as if the user typed it.
# "Any" triggers the existing skip-detection in state_updater_node.
ATTRIBUTE_SUGGESTIONS = {
    "smartphone": {
        "os":                  ["Android", "iOS", "Other", "Any"],
        "price_usd":           ["Under $200", "$200-500", "$500-1000", "$1000+", "Any"],
        "battery_capacity":    ["3000+ mAh", "4000+ mAh", "5000+ mAh", "Any"],
        "primary_camera_rear": ["12+ MP", "48+ MP", "64+ MP", "100+ MP", "Any"],
        "ram_capacity":        ["4 GB", "6 GB", "8 GB", "12 GB+", "Any"],
        "internal_memory":     ["64 GB", "128 GB", "256 GB", "512 GB+", "Any"],
    },
    "headphones": {
        "type":                ["Wired", "Wireless", "Any"],
        "form_factor":         ["Over-Ear", "On-Ear", "In-Ear", "Any"],
        "noise_cancellation":  ["Yes", "No", "Any"],
        "price_usd":           ["Under $100", "$100-300", "$300+", "Any"],
    },
}


def _render_suggestion_buttons(category, attribute, msg_index):
    """Render a row of quick-reply buttons; return the clicked label or None."""
    options = ATTRIBUTE_SUGGESTIONS.get(category, {}).get(attribute)
    if not options:
        return None
    cols = st.columns(len(options))
    for i, opt in enumerate(options):
        if cols[i].button(opt, key=f"sugg_{msg_index}_{attribute}_{i}", use_container_width=True):
            return opt
    return None


# ── Main chat area ────────────────────────────────────────────────────────────

col1, col2 = st.columns([3, 1])

with col1:
    st.title("🛍️ Product Assistant")
    st.caption("I'll help you find the perfect smartphone or pair of headphones.")

# Welcome message
if st.session_state.show_welcome:
    with st.chat_message("assistant"):
        st.markdown(
            "👋 Welcome! I'm your personal product assistant. I can help you find the perfect **smartphone** or **headphones**.\n\n"
            "Just tell me what you're looking for, and I'll ask a few questions to narrow down the best options for you. What can I help you with today?"
        )
    st.session_state.show_welcome = False

# Render chat history (with browsable recommendation lists and quick-reply buttons)
_last_idx = len(st.session_state.chat_messages) - 1
for i, msg in enumerate(st.session_state.chat_messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Recommendation list — kept per-message so past recommendations
        # stay browsable; each message has its own checkbox state via msg_index.
        if msg.get("recommend_products"):
            st.divider()
            _render_recommendation_list(
                msg["recommend_products"],
                msg["recommend_category"],
                i,
            )
        # Quick-reply buttons: only on the most recent assistant message that's
        # awaiting an answer to a clarification question.
        if (
            i == _last_idx
            and msg["role"] == "assistant"
            and msg.get("clarification_attribute")
        ):
            clicked = _render_suggestion_buttons(
                msg.get("clarification_category"),
                msg["clarification_attribute"],
                i,
            )
            if clicked:
                st.session_state.queued_input = clicked
                st.rerun()

# ── Chat input ────────────────────────────────────────────────────────────────

# Accept either typed input or a queued quick-reply button click.
typed_input = st.chat_input("Tell me what you're looking for...")
queued_input = st.session_state.pop("queued_input", None)
user_input = typed_input or queued_input

if user_input:
    # If the previous turn ended with action="done" (user said thanks / start
    # over / etc.), wipe the conversation and product state BEFORE processing
    # the new message — so this input starts a completely fresh session.
    if st.session_state.pop("pending_reset", False):
        st.session_state.dialogue_state = initial_state()
        st.session_state.chat_messages = []
        st.session_state.show_welcome = False  # the new input is taking welcome's place
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_messages.append({"role": "user", "content": user_input})

    # Run the LangGraph pipeline. The interactive recommendation list and the
    # quick-reply buttons are intentionally NOT rendered inline — they have
    # widget keys that would collide if rendered twice. The rerun + history
    # loop below renders them once per pass.
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            new_state = run_turn(st.session_state.dialogue_state, user_input)
            st.session_state.dialogue_state = new_state
        response = new_state["response"]
        st.markdown(response)

    # Persist the assistant turn. Save the ranked recommendation list (for
    # recommend turns) or the clarification metadata (for question turns).
    chat_entry = {"role": "assistant", "content": response}
    if new_state["action"] == "recommend" and new_state["candidates"]:
        chat_entry["recommend_products"] = new_state["candidates"]
        chat_entry["recommend_category"] = new_state["category"]
    elif new_state["action"] == "ask_clarification" and new_state.get("clarification_attribute"):
        chat_entry["clarification_attribute"] = new_state["clarification_attribute"]
        chat_entry["clarification_category"] = new_state["category"]
    st.session_state.chat_messages.append(chat_entry)

    # If the user signaled they're done / want to start over, queue a full
    # reset that will fire on their next message (see top of input handler).
    if new_state["action"] == "done":
        st.session_state.pending_reset = True

    st.rerun()
