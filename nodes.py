"""
nodes.py
All LangGraph node functions. Each takes a DialogueState and returns
an updated DialogueState (partial dict — LangGraph merges automatically).
"""

import os
import json
import re
from typing import Any, Dict
from dotenv import load_dotenv

load_dotenv()

import database
from state import DialogueState

# ── LLM factory ───────────────────────────────────────────────────────────────

def _get_llm():
    provider = os.getenv("LLM_PROVIDER", "openrouter")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    elif provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct"),
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0,
        )

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: '{provider}'. Choose from: openai, anthropic, openrouter")

llm = _get_llm()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_json_response(text: str) -> Dict[str, Any]:
    """Robustly extract a JSON object from an LLM response."""
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Find first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def _recent_history(state: DialogueState, n: int = 6) -> str:
    """Format the last n messages as a readable string for prompts."""
    lines = []
    for msg in state["messages"][-n:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines) or "(start of conversation)"


# ── Node 1: Intent Classification + Preference Extraction ────────────────────

def intent_and_extract_node(state: DialogueState) -> dict:
    """
    Single LLM call that classifies intent and extracts structured filters.
    Returns partial state update.
    """
    history = _recent_history(state)
    active = json.dumps(state["active_filters"], indent=2)

    prompt = f"""You are a preference extractor for a conversational product recommender.

Available product categories: smartphone, headphones.

Analyze the user's latest message and return a JSON object with EXACTLY these fields:

{{
  "intent": "<one of: explore | specific | refine | done | chitchat>",
  "category": "<one of: smartphone | headphones | null>",
  "extracted_filters": {{
    // Structured filters extracted from THIS message only.
    // Only include fields the user explicitly mentioned.
    // Use null to REMOVE a previously-set filter (user changed their mind).
    // String matching is case-insensitive.
    //
    // === Smartphone filters ===
    //   brand_name                              (string, e.g. "samsung", "apple", "xiaomi", "oneplus")
    //   model_contains                          (string, substring of model name)
    //   price_min, price_max                    (integer, INR — typical range 4000-240000)
    //   rating_min                              (number, 0-100 scale, typical 60-89)
    //   battery_capacity_min                    (integer, mAh)
    //   fast_charging_min                       (integer, watts)
    //   ram_capacity, ram_capacity_min          (integer, GB)
    //   internal_memory, internal_memory_min    (integer, GB)
    //   screen_size_min, screen_size_max        (number, inches)
    //   num_rear_cameras_min                    (integer)
    //   os                                      ("android" | "ios" | "other")
    //   primary_camera_rear_min                 (number, MP)
    //   primary_camera_front_min                (number, MP)
    //
    // === Headphones filters ===
    //   brand                                   (string, e.g. "Sony", "Bose", "Sennheiser")
    //   model_contains                          (string)
    //   type                                    ("Wired" | "Wireless")
    //   connectivity                            ("3.5mm" | "Bluetooth")
    //   form_factor                             ("In-Ear" | "On-Ear" | "Over-Ear")
    //   microphone                              (true | false)
    //   noise_cancellation                      (true | false)
    //   foldable                                (true | false)
    //   battery_hrs_min                         (number, hours — wireless only)
    //   price_usd_min, price_usd_max            (integer, USD — typical range 50-940)
    //   avg_rating_min                          (number, 0-5 scale)
    //   release_year_min, release_year_max      (integer)
  }}
}}

Intent definitions:
- explore: user wants to browse or get recommendations in a general direction
- specific: user wants a very specific product (exact model, brand+spec combo)
- refine: user is critiquing or narrowing previous results
- done: user is satisfied / wants to stop
- chitchat: unrelated to product search

Current active filters: {active}
Current category: {state['category'] or 'not set yet'}

Conversation so far:
{history}

User's latest message: "{state['user_input']}"

Return ONLY the JSON object, no explanation."""

    response = llm.invoke(prompt)
    parsed = _parse_json_response(response.content)

    # Determine category: keep existing if new one is null
    new_category = parsed.get("category") or state["category"]

    return {
        "intent": parsed.get("intent", "explore"),
        "extracted_filters": parsed.get("extracted_filters", {}),
        "category": new_category,
    }


# ── Node 2: Dialogue State Updater ────────────────────────────────────────────

def state_updater_node(state: DialogueState) -> dict:
    """
    Merges newly extracted filters into the persistent active_filters.
    Pure Python — no LLM call.
    - Non-null values overwrite existing filters
    - Null values explicitly REMOVE a filter (user changed their mind)
    """
    updated = dict(state["active_filters"])

    for key, value in state["extracted_filters"].items():
        if value is None:
            # Explicit removal
            updated.pop(key, None)
        else:
            updated[key] = value

    return {
        "active_filters": updated,
        "turn_count": state["turn_count"] + 1,
    }


# ── Node 3: Product Retriever + Action Selector ───────────────────────────────

def retrieve_and_act_node(state: DialogueState) -> dict:
    """
    Queries the database with current filters and decides the next action.
    Pure Python — no LLM call.
    """
    # No category yet → must ask
    if not state["category"]:
        return {
            "action": "ask_category",
            "candidates": [],
            "clarification_attribute": None,
        }

    # User is done
    if state["intent"] == "done":
        return {
            "action": "done",
            "candidates": [],
            "clarification_attribute": None,
        }

    candidates = database.retrieve(state["category"], state["active_filters"], limit=10)

    if len(candidates) == 0:
        action = "no_results"
        clarification_attribute = None
    elif len(candidates) <= 3:
        action = "recommend"
        clarification_attribute = None
    else:
        # Too many results → ask the most discriminative question
        action = "ask_clarification"
        clarification_attribute = database.most_discriminative_attribute(
            state["category"], state["active_filters"]
        )

    return {
        "action": action,
        "candidates": candidates,
        "clarification_attribute": clarification_attribute,
    }


# ── Node 4: Response Generator ────────────────────────────────────────────────

def response_generator_node(state: DialogueState) -> dict:
    """
    Generates the final natural-language assistant reply.
    One LLM call, guided by the action and current state.
    """
    action = state["action"]
    category = state["category"] or "products"
    # Human-readable plural for the assistant prose (avoids "headphoness").
    CATEGORY_LABEL = {
        "smartphone": "smartphones",
        "headphones": "headphones",
    }
    category_label = CATEGORY_LABEL.get(category, category)
    filters_summary = json.dumps(state["active_filters"], indent=2) if state["active_filters"] else "none"

    # Format top candidates for the prompt
    top_candidates = state["candidates"][:3]
    candidates_text = ""
    if top_candidates:
        lines = []
        for p in top_candidates:
            # Show most important fields, not the whole row
            key_fields = {k: v for k, v in p.items() if k not in ("id",)}
            lines.append(json.dumps(key_fields))
        candidates_text = "\n".join(lines)

    # Build action-specific instructions
    action_instructions = {
        "ask_category": (
            "Greet the user warmly and ask which product category they are interested in. "
            "Mention the available categories: smartphones or headphones."
        ),
        "ask_clarification": (
            f"You have {len(state['candidates'])} products matching the current filters. "
            f"Ask ONE focused question to narrow this down. "
            f"The best attribute to ask about next is: '{state['clarification_attribute']}'. "
            "Make the question feel natural, not like a form. "
            "Optionally mention 2-3 example values for that attribute."
        ),
        "recommend": (
            f"You have found {len(top_candidates)} product(s) that match perfectly. "
            "Present them clearly: for each product, give the model name, price, and 3-4 key specs. "
            "End with a brief friendly note (e.g. ask if they want more details or to search differently)."
        ),
        "no_results": (
            "No products match the current filters. Apologize briefly. "
            "Identify the most restrictive filter and suggest relaxing it. "
            "Ask if they'd like to adjust their preferences."
        ),
        "done": (
            "The user is satisfied. Wish them well and offer to help with another search."
        ),
    }.get(action, "Respond helpfully to the user.")

    prompt = f"""You are a friendly, knowledgeable shop assistant helping a customer find {category_label}.

Your task: {action_instructions}

Current session context:
- Category: {category}
- Active filters: {filters_summary}
- Turn number: {state['turn_count']}

{"Products to present:" if top_candidates else ""}
{candidates_text}

Rules:
- Be concise (3-6 sentences max unless presenting products)
- Never mention "filters", "database", "JSON", or technical internals
- Sound like a real shop assistant, not a chatbot
- Do not repeat the user's exact words back to them
- If presenting products, use a clear format: **Brand Model** — €price — key specs

Write the assistant reply only, no preamble."""

    response = llm.invoke(prompt)

    return {"response": response.content.strip()}
