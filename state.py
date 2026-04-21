"""
state.py
Central dialogue state definition shared by all LangGraph nodes.
"""

from typing import TypedDict, Optional, List, Dict, Any


class DialogueState(TypedDict):
    # ── Conversation history ──────────────────────────────────────────────────
    messages: List[Dict[str, str]]   # [{"role": "user"|"assistant", "content": "..."}]
    user_input: str                  # latest raw user message

    # ── Intent & extraction (refreshed every turn) ────────────────────────────
    intent: Optional[str]            # explore | specific | refine | done | chitchat
    extracted_filters: Dict[str, Any]  # filters extracted from THIS turn only

    # ── Persistent dialogue state (accumulated across turns) ──────────────────
    category: Optional[str]          # smartphone | washing_machine | laptop | None
    active_filters: Dict[str, Any]   # merged filters from all turns so far

    # ── Action & retrieval ────────────────────────────────────────────────────
    action: Optional[str]            # ask_category | ask_clarification | recommend | no_results | done
    candidates: List[Dict[str, Any]] # products matching current filters (up to 10)
    clarification_attribute: Optional[str]  # which attribute to ask about next

    # ── Response ──────────────────────────────────────────────────────────────
    response: str                    # final assistant message shown to the user
    turn_count: int


def initial_state() -> DialogueState:
    """Returns a blank starting state for a new session."""
    return DialogueState(
        messages=[],
        user_input="",
        intent=None,
        extracted_filters={},
        category=None,
        active_filters={},
        action=None,
        candidates=[],
        clarification_attribute=None,
        response="",
        turn_count=0,
    )
