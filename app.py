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

# Render chat history (including any saved comparison tables)
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("table_products"):
            st.divider()
            st.caption("**Comparison:**")
            _render_comparison_table(msg["table_products"], msg["table_category"])

# ── Chat input ────────────────────────────────────────────────────────────────

if user_input := st.chat_input("Tell me what you're looking for..."):
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_messages.append({"role": "user", "content": user_input})

    # Run the LangGraph pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            new_state = run_turn(st.session_state.dialogue_state, user_input)
            st.session_state.dialogue_state = new_state

        response = new_state["response"]
        st.markdown(response)

        # If we just recommended, render the comparison table beneath the reply
        is_recommend = new_state["action"] == "recommend" and new_state["candidates"]
        if is_recommend:
            st.divider()
            st.caption("**Comparison:**")
            _render_comparison_table(new_state["candidates"][:2], new_state["category"])

    # Persist the assistant turn; include the comparison products so the table
    # re-renders on subsequent reruns from chat history.
    chat_entry = {"role": "assistant", "content": response}
    if is_recommend:
        chat_entry["table_products"] = new_state["candidates"][:2]
        chat_entry["table_category"] = new_state["category"]
    st.session_state.chat_messages.append(chat_entry)
    st.rerun()
