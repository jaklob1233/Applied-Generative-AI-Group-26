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


# ── Vague-term hint block (injected into the intent extraction prompt) ───────

def _vague_terms_block() -> str:
    """
    Build a compact percentile reference + mapping rules so the LLM can
    translate terms like 'cheap', 'premium', 'big battery', 'good camera'
    into concrete filter thresholds grounded in the actual dataset.
    """
    rows = []
    for category in database.get_categories():
        thresholds = database.vague_term_thresholds(category)
        if not thresholds:
            continue
        bits = []
        for attr, p in thresholds.items():
            if attr.startswith("price"):
                bits.append(f"{attr}=${p['p25']:.0f}/${p['p50']:.0f}/${p['p75']:.0f}")
            elif attr in ("avg_rating",):
                bits.append(f"{attr}={p['p25']:.1f}/{p['p50']:.1f}/{p['p75']:.1f}")
            else:
                bits.append(f"{attr}={p['p25']:.0f}/{p['p50']:.0f}/{p['p75']:.0f}")
        rows.append(f"  [{category}] " + ", ".join(bits))
    if not rows:
        return ""
    percentile_lines = "\n".join(rows)
    return (
        "VAGUE TERMS — translate to concrete filter thresholds grounded in these dataset percentiles (p25/median/p75):\n"
        f"{percentile_lines}\n"
        "Mapping rules:\n"
        "  - 'cheap' / 'budget' / 'affordable' → <price>_max = p25 of that category\n"
        "  - 'premium' / 'high-end' / 'expensive' / 'flagship' → <price>_min = p75\n"
        "  - 'mid-range' / 'moderate' → <price>_min = p25 AND <price>_max = p75\n"
        "  - Quality superlatives ('good camera', 'big battery', 'lots of RAM', 'highly rated', 'long battery life') → corresponding <attribute>_min = p75\n"
        "  - Negative qualifiers ('small battery', 'low rated', 'weak') → corresponding <attribute>_max = p25\n"
        "  - Always use the actual numbers above; do NOT invent thresholds."
    )


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
    vague_block = _vague_terms_block()

    # Relative-critique anchor: if the LAST turn ended in a recommendation,
    # tell the LLM the min/median/max of those products so terms like
    # "cheaper" / "bigger battery" / "better camera" can be grounded in
    # what the user actually just saw.
    prev_stats = state.get("last_recommend_stats")
    if prev_stats:
        stat_lines = []
        for attr, s in prev_stats.items():
            prefix = "$" if attr.startswith("price") else ""
            stat_lines.append(
                f"  {attr}: min={prefix}{s['min']:.0f}, median={prefix}{s['median']:.0f}, max={prefix}{s['max']:.0f}"
            )
        critique_block = (
            "PREVIOUS RECOMMENDATIONS — these products were just shown to the user:\n"
            + "\n".join(stat_lines)
            + "\nWhen the user uses RELATIVE terms (refine intent), anchor against these:\n"
            "  - 'cheaper' / 'less expensive' → <price>_max = previous min (strictly cheaper than anything shown)\n"
            "  - 'more expensive' / 'pricier' → <price>_min = previous max\n"
            "  - 'bigger battery' / 'longer battery' → <battery>_min = previous max\n"
            "  - 'smaller' / 'lower X' → corresponding <X>_max = previous min\n"
            "  - 'better camera' / 'better X' → <X>_min = previous max\n\n"
            "CRITICAL FOR REFINE INTENT: ONLY return the NEW filter(s) the user is adding. "
            "DO NOT include the existing filters in extracted_filters — they are already in "
            "active_filters and will be preserved by the state updater. NEVER set an existing "
            "filter to null (that removes it!) unless the user explicitly says 'forget about X', "
            "'change X', 'I don't care about X anymore'. By default, refine = ADD a constraint, "
            "not replace the whole filter set."
        )
    else:
        critique_block = ""

    prompt = f"""You are a preference extractor for a conversational product recommender.

Available product categories: smartphone, headphones.

Analyze the user's LATEST message and return a JSON object with EXACTLY these fields:

{{
  "intent": "<one of: explore | specific | refine | summarize | done | chitchat>",
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

Intent definitions — choose carefully, this drives system behavior:

- specific: user has clear criteria identifying a product or a small set (examples:
  "I want a Samsung Galaxy S22", "show me the cheapest Apple phone", "an Android
  phone with at least 8 GB RAM and 5000 mAh battery", "wireless over-ear with noise
  cancellation under $200"). Trigger when the user gives one or more concrete
  filters/specifications. The system will recommend immediately rather than asking
  follow-up questions.

- explore: user is in the early stages with NO concrete criteria yet
  (examples: "I want a phone", "recommend me headphones", "help me find a smartphone").
  Trigger only when the message is essentially category-only with no spec/feature
  mentioned. The system will guide them through clarification questions.

- refine: user is reacting to PREVIOUS recommendations and wants to narrow,
  change, or critique them (examples: "show me cheaper ones", "anything with a
  better camera", "I don't like Samsung", "actually I prefer iPhone"). Trigger
  whenever the conversation history already contains a recommendation and the
  user is adjusting. The system will re-recommend with the updated filters
  rather than asking new questions.

- summarize: user wants a recap of preferences gathered so far (examples:
  "what do you have so far?", "recap my preferences", "summarize", "what did
  I tell you?"). Trigger on explicit recap requests. The system will state
  what it understood without making changes.

- done: user signals the conversation should END or RESTART (examples:
  "I'm done", "that's all", "thanks, bye", "let's start over", "start again",
  "reset", "new search", "begin again", "restart"). The conversation will be
  cleared on the next message.

- chitchat: greetings ("hi", "good morning"), generic thanks (not signaling
  done), jokes, questions about you, or anything else NOT about finding a
  product. When intent is chitchat, extracted_filters MUST be empty and
  category MUST be null (do not invent values).

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

{vague_block}

{critique_block}

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

    # 0. Chitchat short-circuit. Don't touch product state at all — even if the
    #    LLM hallucinated a category or a filter from a greeting like "good
    #    morning", we ignore it. Just advance the turn counter.
    if intent == "chitchat":
        return {"turn_count": state["turn_count"] + 1}

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
    Decide the next system action based on intent + dialogue state.

    Intent-driven branching:
      - specific (with filters): user has clear criteria → recommend immediately.
      - refine: user is critiquing prior results → re-recommend with new filters.
      - summarize: user wants a recap → no DB action, response_generator does the talk.
      - explore (or specific with no filters yet): guided question flow.
      - done / chitchat / no category: handled before retrieval.

    Pure Python — no LLM call.
    """
    intent = state["intent"]
    category = state["category"]

    # No category yet → must ask
    if not category:
        return {
            "action": "ask_category",
            "candidates": [],
            "clarification_attribute": None,
            "last_asked_attribute": None,
        }

    # User signaled they're done / restarting
    if intent == "done":
        return {
            "action": "done",
            "candidates": [],
            "clarification_attribute": None,
            "last_asked_attribute": None,
        }

    # Off-topic / small-talk — preserve candidates + pending question.
    if intent == "chitchat":
        return {"action": "chitchat"}

    # Recap request — keep candidates + clarification context so the response
    # generator can summarize the current preferences without changing them.
    if intent == "summarize":
        return {"action": "summarize"}

    filters = state["active_filters"]
    candidates = database.retrieve(category, filters, limit=None)

    if len(candidates) == 0:
        return {
            "action": "no_results",
            "candidates": [],
            "clarification_attribute": None,
            "last_asked_attribute": None,
        }

    has_filters = bool(filters)

    def _recommend_return(ranked):
        # Save stats of THIS recommendation set so the next turn can anchor
        # relative critiques ('cheaper', 'bigger battery') against it.
        return {
            "action": "recommend",
            "candidates": ranked,
            "clarification_attribute": None,
            "last_asked_attribute": None,
            "last_recommend_stats": database.candidate_stats(category, ranked),
        }

    # SPECIFIC intent + concrete filters → user knows what they want; skip the
    # question chain and recommend immediately. Falls through to explore-style
    # guided flow when the LLM tagged "specific" but no filters were actually
    # extracted (e.g. "I want a phone" mistakenly labeled specific).
    if intent == "specific" and has_filters:
        ranked = database.top_n_by_score(category, candidates, n=len(candidates))
        return _recommend_return(ranked)

    # REFINE intent → user critiquing previous recommendations; apply the new
    # filter delta and re-recommend without asking more questions.
    if intent == "refine":
        ranked = database.top_n_by_score(category, candidates, n=len(candidates))
        return _recommend_return(ranked)

    # EXPLORE flow (default): guide through the fixed question sequence.
    next_q = database.next_question(
        category, filters, state.get("asked_skipped", [])
    )
    if len(candidates) <= 2 or next_q is None:
        ranked = database.top_n_by_score(category, candidates, n=len(candidates))
        return _recommend_return(ranked)
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

    # For the LLM prompt, only show the top 2 (it doesn't need to know about
    # all 50). The full ranked list is in state["candidates"] for the UI.
    top_candidates = state["candidates"][:2]
    total_matches = len(state["candidates"])
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

    # For chitchat: optionally reference the pending clarification question
    pending_q = state.get("last_asked_attribute")
    chitchat_reprompt = (
        f" They were in the middle of choosing — you'd just asked about '{pending_q}'. Gently invite them back to that question."
        if pending_q else
        " Invite them to tell you what they're looking for."
    )

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
            f"You found {total_matches} matching {category_label}, sorted by an internal match score (best first). "
            "A browsable list will appear below your reply — each item expands to show full specs, and the user can check 2-3 boxes to see a side-by-side comparison automatically. "
            "DO NOT list specs, prices, or scores in your text. "
            f"Write 1-2 sentences acknowledging the {total_matches} options, optionally name the very top pick (brand + model only) as a starting suggestion, and invite the user to browse the list and pick items to compare. "
            "End with a friendly note offering to refine the search if needed."
        ),
        "no_results": (
            "No products match the current preferences. Apologize briefly. "
            "Identify the most restrictive preference and suggest relaxing it. "
            "Ask if they'd like to adjust."
        ),
        "summarize": (
            f"The user asked you to recap what you understand so far. "
            f"Give a clear plain-English summary of the {category_label} category "
            f"and the active preferences (listed in the session context). "
            f"Format: 1-2 sentences naming the criteria. End with 'Is that right, "
            f"or would you like to change anything?'. Do not list raw filter keys "
            f"like 'price_usd_max' — translate to natural phrasing (e.g. "
            f"'under $300', 'at least 8 GB RAM')."
        ),
        "done": (
            "The user wants to wrap up or start over. Reply warmly in ONE or TWO short sentences — "
            "thank them and mention they can begin a fresh search any time. "
            "The conversation history will be cleared automatically on their next message, "
            "but DON'T mention that explicitly (no 'I'll reset' or technical talk). "
            "Just keep it friendly and inviting."
        ),
        "chitchat": (
            "The user said something off-topic — a greeting, thanks, or small talk. "
            "Reply warmly in ONE short sentence acknowledging what they said. "
            "Then in ONE sentence politely remind them that you're here to help find a smartphone or a pair of headphones."
            + chitchat_reprompt
            + " Keep the whole reply to 2-3 sentences. Do NOT pretend to have product results — there are none to present."
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
