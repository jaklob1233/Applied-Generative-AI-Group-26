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


# ── Per-attribute explanations (woven into clarification questions) ──────────

ATTRIBUTE_HINTS: Dict[str, Dict[str, str]] = {
    "smartphone": {
        "os": "Operating system — Android (open, wide hardware choice) or iOS (tight integration with Apple devices). 'Other' covers less common OSes.",
        "price_usd": "Budget in US Dollars. Dataset range is roughly $40 to $2,500.",
        "battery_capacity": "Battery in mAh — higher = longer life. 4,000-5,000 mAh comfortably lasts a day.",
        "primary_camera_rear": "Main rear camera in megapixels. 48-64 MP is mid-range, 100+ MP is flagship territory.",
        "ram_capacity": "RAM in GB — affects multitasking smoothness. 4 GB is okay for casual use, 8-12 GB for heavy multitasking and gaming.",
        "internal_memory": "Internal storage in GB. 64 GB is tight for media, 128 GB is comfortable, 256 GB+ is generous.",
    },
    "headphones": {
        "type": "Wired or Wireless? Wireless gives mobility; wired avoids batteries and pairing.",
        "form_factor": "Over-Ear (most immersive), On-Ear (compact and lighter), or In-Ear (most portable, gym-friendly).",
        "noise_cancellation": "Active noise cancellation blocks ambient noise — great for flights and commutes.",
        "price_usd": "Budget in US Dollars. Dataset range is $50 to $940.",
    },
}


# ── Node 1: Intent Classification + Preference Extraction ────────────────────

def intent_and_extract_node(state: DialogueState) -> dict:
    """
    Single LLM call: classify intent, extract filters, identify category.
    Does NOT update the persistent `category` field — that's the state_updater's
    job (so it can detect category switches by comparing old vs new).
    """
    history = _recent_history(state)
    active = json.dumps(state["active_filters"], indent=2)
    last_asked = state.get("last_asked_attribute") or "nothing in particular"

    prompt = f"""You are a preference extractor for a conversational product recommender.

Available product categories: smartphone, headphones.

Analyze the user's LATEST message and return a JSON object with EXACTLY these fields:

{{
  "intent": "<one of: explore | specific | refine | done | chitchat>",
  "category": "<one of: smartphone | headphones | null>",
  "extracted_filters": {{
    // Structured filters extracted from THIS message only.
    // Only include fields the user explicitly mentioned in this message.
    // Use null as a VALUE to REMOVE a previously-set filter.
    // String matching is case-insensitive.
    //
    // === Smartphone filters ===
    //   brand_name                              (string, e.g. "samsung", "apple", "xiaomi", "oneplus")
    //   model_contains                          (string, substring of model name)
    //   price_usd_min, price_usd_max            (integer, USD — typical range $40-$2,500)
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

CATEGORY SWITCHING:
If the user pivots to a DIFFERENT product category mid-conversation (e.g. they
were looking at smartphones and now say "now show me headphones" or "actually
I want headphones"), return the NEW category in the "category" field. The
system will reset all previous filters automatically. If the user is staying
on the same category, you may return null for "category".

SKIPPING A QUESTION:
The assistant's previous question was about: "{last_asked}".
If the user declines to specify (e.g. "any", "doesn't matter", "I don't care",
"skip", "whatever", "either is fine"), simply DO NOT include a filter for that
attribute. Don't invent a value.

Current active filters: {active}
Current category: {state['category'] or 'not set yet'}

Conversation so far:
{history}

User's latest message: "{state['user_input']}"

Return ONLY the JSON object, no explanation."""

    response = llm.invoke(prompt)
    parsed = _parse_json_response(response.content)

    return {
        "intent": parsed.get("intent", "explore"),
        "extracted_filters": parsed.get("extracted_filters", {}),
        "extracted_category": parsed.get("category"),  # raw; state_updater decides what to do
    }


# ── Node 2: Dialogue State Updater ────────────────────────────────────────────

def state_updater_node(state: DialogueState) -> dict:
    """
    Merge newly extracted filters into active_filters, with two extra behaviors:
      1. CATEGORY SWITCH: if the user pivots to a different non-null category,
         reset active_filters and asked_skipped (start fresh).
      2. SKIP DETECTION: if the user was asked about an attribute last turn
         and didn't provide a filter for it (and isn't chitchatting), mark
         that attribute as 'skipped' so we don't ask again.

    Pure Python — no LLM call.
    """
    extracted = state["extracted_filters"]
    extracted_category = state.get("extracted_category")
    current_category = state["category"]
    intent = state["intent"]
    last_asked = state.get("last_asked_attribute")

    # 1. Category switch → wipe filters + skipped list, keep only THIS turn's filters
    if (
        extracted_category
        and current_category
        and extracted_category != current_category
    ):
        return {
            "category": extracted_category,
            "active_filters": {k: v for k, v in extracted.items() if v is not None},
            "asked_skipped": [],
            "candidates": [],
            "turn_count": state["turn_count"] + 1,
        }

    # 2. Otherwise: decide effective category (set first time, or keep current)
    new_category = extracted_category or current_category

    # 3. Merge filters (None removes a prior filter)
    updated = dict(state["active_filters"])
    for key, value in extracted.items():
        if value is None:
            updated.pop(key, None)
        else:
            updated[key] = value

    # 4. Skip detection — only when user engaged with the search and didn't
    #    answer the previous question.
    updated_skipped = list(state.get("asked_skipped", []))
    if (
        last_asked
        and intent not in (None, "chitchat", "done")
        and last_asked not in updated_skipped
        and not any(database.base_attr(k) == last_asked for k in updated)
    ):
        updated_skipped.append(last_asked)

    return {
        "category": new_category,
        "active_filters": updated,
        "asked_skipped": updated_skipped,
        "turn_count": state["turn_count"] + 1,
    }


# ── Node 3: Product Retriever + Action Selector ───────────────────────────────

def retrieve_and_act_node(state: DialogueState) -> dict:
    """
    Query the database, decide the next action via the fixed question order,
    and (when recommending) pick the top 2 products by weighted score.
    Pure Python — no LLM call.
    """
    # No category yet → must ask
    if not state["category"]:
        return {
            "action": "ask_category",
            "candidates": [],
            "clarification_attribute": None,
            "last_asked_attribute": None,
        }

    # User signaled they're done
    if state["intent"] == "done":
        return {
            "action": "done",
            "candidates": [],
            "clarification_attribute": None,
            "last_asked_attribute": None,
        }

    # Pull a generous candidate pool so the scorer has a real population
    candidates = database.retrieve(state["category"], state["active_filters"], limit=50)

    if len(candidates) == 0:
        return {
            "action": "no_results",
            "candidates": [],
            "clarification_attribute": None,
            "last_asked_attribute": None,
        }

    next_q = database.next_question(
        state["category"],
        state["active_filters"],
        state.get("asked_skipped", []),
    )

    # Recommend when either we've already narrowed enough OR there are no more
    # questions to ask. Always present the top 2 by weighted score.
    if len(candidates) <= 2 or next_q is None:
        top = database.top_n_by_score(state["category"], candidates, n=2)
        return {
            "action": "recommend",
            "candidates": top,
            "clarification_attribute": None,
            "last_asked_attribute": None,
        }

    # Still narrowing → ask the next attribute from the fixed sequence
    return {
        "action": "ask_clarification",
        "candidates": candidates,
        "clarification_attribute": next_q,
        "last_asked_attribute": next_q,
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

    # Format top candidates for the prompt (recommend action picks 2)
    top_candidates = state["candidates"][:2]
    candidates_text = ""
    if top_candidates:
        lines = []
        for p in top_candidates:
            key_fields = {k: v for k, v in p.items() if k != "id"}
            lines.append(json.dumps(key_fields, default=str))
        candidates_text = "\n".join(lines)

    # Look up the explanation for the attribute we're about to ask about
    attr = state.get("clarification_attribute")
    attr_hint = ATTRIBUTE_HINTS.get(category, {}).get(attr, "") if attr else ""

    # Build action-specific instructions
    action_instructions = {
        "ask_category": (
            "Greet the user warmly and ask which product category they are interested in. "
            "Mention the available categories: smartphones or headphones."
        ),
        "ask_clarification": (
            f"Ask the user about ONE attribute: '{attr}'.\n"
            f"Context to weave in (briefly): {attr_hint}\n"
            "Phrase a single natural question — 2 to 3 sentences total. Briefly explain the trade-off using the context above, "
            "but don't over-list options. Mention casually that the user can say 'any' or 'skip' to move on."
        ),
        "recommend": (
            "You've selected the top 2 products by an internal score. A comparison TABLE will be shown automatically below your reply, "
            "so DO NOT list specs, prices, or scores in your text. "
            "Write 1-2 sentences introducing the two picks (use brand + model only) and the single biggest reason each one stands out. "
            "End with a friendly closing line offering to adjust the search."
        ),
        "no_results": (
            "No products match the current preferences. Apologize briefly. "
            "Identify the most restrictive preference and suggest relaxing it. "
            "Ask if they'd like to adjust."
        ),
        "done": (
            "The user is satisfied. Wish them well and offer to help with another search."
        ),
    }.get(action, "Respond helpfully to the user.")

    prompt = f"""You are a friendly, knowledgeable shop assistant helping a customer find {category_label}.

Your task: {action_instructions}

Current session context:
- Category: {category}
- Active preferences: {filters_summary}
- Turn number: {state['turn_count']}

{"Selected products (for your awareness — DO NOT list specs in your reply):" if top_candidates else ""}
{candidates_text}

Rules:
- Be concise. For ask_clarification: 2-3 sentences. For recommend: 1-2 sentences plus a closing line.
- Never mention "filters", "database", "JSON", "score", or technical internals.
- Sound like a real shop assistant, not a chatbot.
- Do not repeat the user's exact words back to them.

Write the assistant reply only, no preamble."""

    response = llm.invoke(prompt)

    return {"response": response.content.strip()}
