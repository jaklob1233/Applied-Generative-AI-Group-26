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

    /* Product card styling */
    .product-card {
        background: #f8f9fa;
        border-left: 4px solid #0066cc;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.9em;
    }

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
        for k, v in filters.items():
            label = k.replace("_", " ").replace(" eur", " (€)").title()
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

# ── Main chat area ────────────────────────────────────────────────────────────

col1, col2 = st.columns([3, 1])

with col1:
    st.title("🛍️ Product Assistant")
    st.caption("I'll help you find the perfect smartphone, laptop, or washing machine.")

# Welcome message
if st.session_state.show_welcome:
    with st.chat_message("assistant"):
        st.markdown(
            "👋 Welcome! I'm your personal product assistant. I can help you find the perfect **smartphone**, **laptop**, or **washing machine**.\n\n"
            "Just tell me what you're looking for, and I'll ask a few questions to narrow down the best options for you. What can I help you with today?"
        )
    st.session_state.show_welcome = False

# Render chat history
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

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

        # If recommendations were made, show product cards below the response
        if new_state["action"] == "recommend" and new_state["candidates"]:
            st.divider()
            st.caption("**Recommended products:**")
            for product in new_state["candidates"][:3]:
                _render_product_card(product, new_state["category"])

    st.session_state.chat_messages.append({"role": "assistant", "content": response})
    st.rerun()


# ── Product card renderer ─────────────────────────────────────────────────────

def _render_product_card(product: dict, category: str):
    """Render a styled product card below the chat response."""
    brand = product.get("brand", "")
    model = product.get("model", "")
    price = product.get("price_eur", "?")

    # Category-specific key specs
    if category == "smartphone":
        specs = (
            f"📱 {product.get('display_inches', '?')}\" display • "
            f"🔋 {product.get('battery_mah', '?')} mAh • "
            f"📸 {product.get('camera_mp', '?')} MP • "
            f"💾 {product.get('ram_gb', '?')} GB RAM / {product.get('storage_gb', '?')} GB • "
            f"{'5G ✓' if product.get('has_5g') else '4G'}"
        )
    elif category == "washing_machine":
        specs = (
            f"🫧 {product.get('capacity_kg', '?')} kg • "
            f"⚡ Class {product.get('energy_class', '?')} • "
            f"🔊 {product.get('noise_db', '?')} dB • "
            f"🔄 {product.get('spin_rpm', '?')} rpm • "
            f"{'Steam ✓' if product.get('has_steam') else ''}"
        )
    elif category == "laptop":
        specs = (
            f"💻 {product.get('display_inches', '?')}\" • "
            f"🧠 {product.get('ram_gb', '?')} GB RAM • "
            f"💾 {product.get('storage_gb', '?')} GB SSD • "
            f"🎮 {product.get('gpu_type', '?')} GPU • "
            f"⚖️ {product.get('weight_kg', '?')} kg"
        )
    else:
        specs = str(product)

    st.markdown(
        f"""<div class="product-card">
        <strong>{brand} {model}</strong> — <strong>€{price}</strong><br>
        <small>{specs}</small>
        </div>""",
        unsafe_allow_html=True,
    )
