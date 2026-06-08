# Conversational Recommender System

A multi-turn LLM **shop assistant** for smartphones & headphones. It began as a structured
LangGraph pipeline and was **re-architected into a hybrid tool-calling agent**, validated by a
purpose-built evaluation harness. The agent is the **default engine**; the pipeline is the fallback.

## Architecture v2 — Hybrid tool-calling agent (default)

The dialogue brain is an LLM that **plans and calls deterministic tools** (`tools.py`) — `search`,
`details`, `compare`, `explain`, `top-picks`, `catalog` — over the same retrieval/ranking core.
Grounding lives in the tools (they return only real catalog data, validate every filter, honour an
explicit count `n`, and report out-of-catalog names honestly); the LLM only decides *which* tools to
call and composes the reply. Deterministic guardrails handle the precise bits (undo, brand exclusion).

**Why we re-architected.** The v1 intent pipeline was whack-a-mole — every new phrasing needed a new
intent/regex, and fixing one scenario broke another (e.g. "suggest me **2** best phones" ignored the
count). The agent makes open-ended and compound requests *just work* via tool composition, with **no
per-scenario code**.

**Proven on an evaluation gate.** `eval_harness.py` (persona conversation simulation + LLM-judge + 28
adversarial/safety checks + latency/cost) A/B-tests both engines on the *same* engine-agnostic gate:

| Metric | Pipeline | **Agent** |
|---|---|---|
| Persona conversation quality (LLM-judge) | 0.41 | **0.81** |
| Adversarial & safety | 82% | **100%** |
| count / compound requests | 1/6 | **6/6** |
| Latency / turn (avg) | 7.2s | **4.6s** |
| Input tokens cached | — | **~52%** |

```bash
streamlit run app.py                     # agent (default)
ENGINE=pipeline streamlit run app.py     # legacy pipeline (fallback)

python eval_harness.py                   # evaluate the agent     -> eval_agent.json
ENGINE=pipeline python eval_harness.py   # evaluate the pipeline  -> eval_baseline.json
python test_tools.py                     # fast deterministic tool unit-tests
```

**v2 modules:** `agent.py` (tool-calling loop + policy + guardrails), `tools.py` (validated tools +
OpenAI function-calling schemas + dispatch), `eval_harness.py` (the gate), `test_tools.py`.

> The sections below document the **v1 structured pipeline** — now the *deterministic core* the agent
> calls, and the engine you get with `ENGINE=pipeline`.

## Project Structure

```
crs_project/
├── requirements.txt      # Python dependencies
├── .env                  # API keys + optional LangSmith tracing
│
│  ── Engine ──
├── agent.py              # v2: tool-calling AGENT loop + policy + guardrails (DEFAULT)
├── tools.py              # validated tools (search/details/compare/explain/catalog) + schemas
├── graph.py              # run_turn() dispatch: ENGINE=agent (default) | pipeline (fallback)
├── state.py              # DialogueState TypedDict
├── nodes.py              # v1 pipeline: 4 nodes (NLU → state update → retrieve/act → respond)
│
│  ── NLU stack (de-risked: the prompt no longer does everything) ──
├── llm_client.py         # Shared LLM client + JSON parsing
├── nlu.py                # Split router (intent/confidence) + slot extractor
├── schema.py             # Slot schema + validation (drops unknown/invalid slots)
├── resolver.py           # Entity resolution: brand aliases + difflib fuzzy match
│
│  ── Retrieval & ranking ──
├── database.py           # CSV loader + structured filter engine + TOPSIS
├── semantic.py           # numpy TF-IDF hybrid retrieval ("good for travel")
├── ranking.py            # Pluggable ranker (learned ↔ TOPSIS) + feedback flywheel
│
│  ── Platform ──
├── memory.py             # Persistent per-user profiles (cross-session)
├── observability.py      # Structured turn logs + analytics + LangSmith status
│
├── app.py                # Streamlit chat UI  ← main entry point
│
│  ── Evaluation ──
├── eval_harness.py       # persona sim + LLM-judge + 28 adversarial/safety checks (eval_*.json)
├── test_tools.py         # deterministic tool unit-tests
├── evaluate.py           # deterministic suites (resolution / validation / intent)
├── test_cli.py           # CLI test (no UI needed)
│
├── datasets/             # Product CSVs (smartphones, headphones — both USD)
├── logs/                 # turns.jsonl, feedback.jsonl (auto-created)
├── profiles/             # per-user JSON profiles (auto-created)
└── models/               # trained ranker_<category>.json (auto-created)
```

## Architecture (deployment-ready)

Each turn flows through a 4-node LangGraph pipeline, but the heavy lifting is
now split into focused, independently-testable modules:

```
user message
   │
   ▼  nlu.route()         small classifier → intent + category + wants_results
   │                      + sort_preference + CONFIDENCE   (low → ask to clarify)
   ▼  nlu.extract_slots() given the category, pull ONLY filters
   │
   ▼  schema.validate_filters()  coerce types · resolve brands/categoricals via
   │                      resolver · range-check · DROP unknown/invalid slots
   ▼  state_updater       merge filters · category switch · skip detection ·
   │                      mixed-initiative UNDO ("ignore that")
   ▼  retrieve_and_act    structured filter → ranking.rank()  (learned model if
   │                      trained, else TOPSIS; blends semantic.py for vibe
   │                      queries; honors "cheapest" sort)
   ▼  respond             natural-language reply
```

**Why this shape (CTO notes):**

- **Validation, not trust.** Every LLM-proposed slot is checked against a schema
  (`schema.py`). Misplaced control fields, wrong types, hallucinated brands, and
  out-of-range numbers are dropped with a logged reason — they can't reach the
  query engine. This replaced an endless cycle of patching the prompt.
- **Entity resolution as a layer** (`resolver.py`): "iphone"→apple, "over ear"→
  Over-Ear, "samsng"→samsung (fuzzy) via alias dicts + stdlib difflib — not prompt
  rules.
- **Confidence-gated clarification.** When the router is unsure and nothing is
  actionable, the system asks instead of guessing.
- **Observability** (`observability.py`): every turn is one JSON line
  (utterance → intent → confidence → raw/validated/dropped slots → action →
  latency). The sidebar **📊 Analytics** panel reads these live. Set
  `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` in `.env` for full LangSmith
  traces.
- **Hybrid retrieval** (`semantic.py`): structured filters can't express "good for
  travel" or "punchy bass". Spec-derived product descriptions are embedded with
  **dense static embeddings** (`model2vec`, ~30 MB, no torch) so "long airplane
  trips" matches noise-cancelling headphones with zero shared words — then blended
  into ranking. Auto-falls back to a numpy TF-IDF index if the model can't load
  (offline). `CRS_SEMANTIC_BACKEND=embeddings|tfidf|auto`.
- **Confidence-gated clarification**: the router emits an `ambiguous` intent (and a
  confidence score); genuinely unclear messages ("hmm idk something") get a
  clarifying question instead of a wrong guess.
- **Multi-intent**: a second product type in one message ("cheap Samsung phones
  *and also headphones*") is captured as `also_category` — handled one at a time,
  with the second offered in the reply rather than dropped.
- **Drop-off funnel** (`observability.analytics()['funnel']`): conversion rate
  (% conversations reaching a recommendation), abandonment at the question stage,
  and average turns-to-recommend — surfaced in the sidebar.
- **Capability boundaries** (`out_of_scope` intent): requests the system genuinely
  can't fulfill — purchasing/checkout, stock, warranty/returns, written reviews —
  are answered honestly ("I can't do that, but I can share specs / compare / filter")
  instead of silently misfiring.
- **Product images** (`images.py` + `enrich_images.py`): images are sourced
  **offline into the catalog**, never fetched live per request (the right pattern for
  latency/reliability/licensing). `enrich_images.py` populates
  `datasets/image_cache.json` via a **pluggable, tiered resolver**:
  **GSMArena** (phones only — best coverage incl. budget brands) → SerpAPI / Google
  Custom Search (broad, needs a free key) → Wikipedia / Openverse (Creative-Commons,
  free, partial) → honest placeholder (never a fake/AI photo). `database.load_all()`
  joins the cache into an `image_url` column; the recommendation cards render it.
  Run once: `python enrich_images.py` (add `--force` to re-fetch all phones as
  uniform GSMArena shots; set `SERPAPI_API_KEY` for full headphone coverage).
  Toggle GSMArena with `CRS_ENABLE_GSMARENA=0`.
  **Caveat:** GSMArena has no public API — this scrapes it politely (offline, cached,
  rate-limited), which is fine for an academic demo but against their ToS for
  commercial use. Production swaps the tier-1 source for a licensed feed (Icecat /
  Best Buy / Amazon) by editing one function — nothing else changes.
- **Memory** (`memory.py`): per-user profiles persist preferred brand/OS, typical
  budget, and last filters across sessions. Returning users get a personalized
  greeting and a **⭐ Use my usual preferences** shortcut.
- **The flywheel** (`ranking.py`): the **👍 Pick this** button logs which product
  a user chose among those shown. `ranking.train(category)` fits a logistic model
  over those choices, replacing the hand-tuned TOPSIS weights with learned ones.
  Until enough data exists, it falls back to TOPSIS automatically.

## Setup

# 1. Create and activate a virtual environment
python -m venv .venv

# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

# 2. Install dependencies (now inside the venv)
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env        # then edit .env and add your keys

Get your keys:
- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/
- **LangSmith** (free for students): https://smith.langchain.com/

### 4. Run the app

**Streamlit UI (recommended):**
```bash
streamlit run app.py
```

**CLI test (faster for debugging):**
```bash
python test_cli.py
```

Note: The app expects product CSVs under `datasets/` (see `database.py:load_all()` for filenames).

---

## How It Works

Each user message flows through a 4-node LangGraph pipeline:

```
User message
    ↓
[intent_extract]   — LLM call: classifies intent + extracts structured filters as JSON
    ↓
[state_update]     — Pure Python: merges new filters into persistent active_filters
    ↓
[retrieve_act]     — Pure Python: queries pandas DataFrame, selects next action
    ↓
[respond]          — LLM call: generates natural-language reply
    ↓
Assistant reply
```

### Dialogue State
The `DialogueState` TypedDict persists across turns and tracks:
- `category`: smartphone | headphones
- `active_filters`: merged structured preferences (e.g. `{price_usd_max: 300, brand_name: "samsung"}`)
- `asked_skipped`: attributes the user explicitly declined to filter on
- `last_asked_attribute`: which attribute the assistant asked about last turn (for skip detection)
- `candidates`: current matching products
- `action`: what the system decided to do this turn
- `messages`: full conversation history

### Conversation Behavior

- **Fixed question order.** Each category has a predefined sequence of clarification questions (see `QUESTION_ORDER` in [database.py](database.py)):
  - Smartphones: `os → price_usd → battery_capacity → primary_camera_rear → ram_capacity → internal_memory`
  - Headphones: `type → form_factor → noise_cancellation → price_usd`
  The system asks them in order, skipping any the user has already answered.

- **Skipping.** If the user says "any", "skip", "I don't care", etc. — or clicks the "Any" quick-reply button — the attribute is added to `asked_skipped` and the system moves on instead of nagging.

- **Quick-reply buttons.** Below every clarification question, the UI shows clickable suggestion chips (e.g. *Android · iOS · Other · Any*) generated from `ATTRIBUTE_SUGGESTIONS` in `app.py`. Clicking sends the label through the LLM pipeline exactly like typed input.

- **Category switching mid-conversation.** Saying "actually, show me headphones" while shopping for smartphones resets `active_filters` and `asked_skipped` automatically — no need to press the sidebar reset button.

- **Chitchat handling.** Greetings ("good morning"), thanks, jokes, or other off-topic messages don't touch the product state; the assistant acknowledges warmly, reminds you of its scope, and (if a question was pending) gently re-asks it.

- **Recommendation.** Once all questions are answered/skipped (or candidates ≤ 2), the system runs **TOPSIS** (Hwang & Yoon, 1981 — Technique for Order of Preference by Similarity to Ideal Solution) on every matching product and shows them all as a browsable list, sorted best-fit first. Closeness coefficient = distance to the ideal (top specs, lowest price) divided by total distance to ideal and worst, scaled to 0-100. Weights live in `database.py` (`_SMARTPHONE_WEIGHTS` / `_HEADPHONES_WEIGHTS`).

- **Browse and compare.** The recommendation list is a list of expanders, top 3 auto-expanded. Each expander shows the product's spec sheet plus a **score breakdown** (per-attribute position vs the whole catalog, 0-100, weight). A checkbox inside each lets the user select 2-3 products; a live side-by-side comparison table renders as soon as 2+ items are checked.

### Supported Filter Keys

**Smartphones** (price in USD):

| Key | Type | Example |
|-----|------|---------|
| `price_usd_min` / `price_usd_max` | int | 300 |
| `brand_name` | str | "samsung" |
| `os` | str | "android" / "ios" / "other" |
| `ram_capacity` / `ram_capacity_min` | int | 8 |
| `internal_memory` / `internal_memory_min` | int | 256 |
| `screen_size_min` / `screen_size_max` | float | 6.5 |
| `battery_capacity_min` | int | 4000 |
| `fast_charging_min` | int | 25 |
| `num_rear_cameras_min` | int | 3 |
| `primary_camera_rear_min` | int | 50 |
| `primary_camera_front_min` | int | 12 |
| `rating_min` | int | 80 |
| `model_contains` | str | "pro" |

**Headphones** (price in USD):

| Key | Type | Example |
|-----|------|---------|
| `price_usd_min` / `price_usd_max` | int | 200 |
| `brand` | str | "Sony" |
| `type` | str | "Wired" / "Wireless" |
| `connectivity` | str | "Bluetooth" / "3.5mm" |
| `form_factor` | str | "Over-Ear" / "On-Ear" / "In-Ear" |
| `noise_cancellation` | bool | true |
| `microphone` | bool | true |
| `foldable` | bool | true |
| `battery_hrs_min` | float | 30 |
| `avg_rating_min` | float | 4.0 |
| `release_year_min` / `release_year_max` | int | 2022 |
| `model_contains` | str | "wh-1000" |

---

## Sample Use Cases

The system is designed around four kinds of user interaction. For each, the table below shows what information the recommender needs to act, and how it gets it.

### 1. Specific lookup (user knows what they want)
```
User:      I want a Samsung Android phone with at least 8GB RAM
System:    [intent=specific, filters={brand_name: samsung, os: android, ram_capacity_min: 8}]
           → recommends 28 matches immediately, ranked by TOPSIS (no follow-up questions)
```
**Information needed:** brand, OS, hard spec floors. The user provides them upfront — the system extracts and recommends without prompting.

### 2. Exploration (user has only a category in mind)
```
User:      I want a smartphone
System:    Which OS do you prefer — Android or iOS?
User:      Android
System:    What's your budget?
User:      Under $400
System:    [keeps asking through the fixed question chain: battery → camera → RAM → storage]
System:    → recommends top-ranked Android phones under $400, all attributes considered
```
**Information needed:** sequence of attribute preferences. The system asks them in priority order (`QUESTION_ORDER`), respecting any "any" / "skip" answers.

### 3. Refinement / critique (user reacts to recommendations)
```
User:      [recommended Samsung Android 8GB phones, $220-1719]
User:      show me cheaper ones
System:    [intent=refine, relative anchor = previous min = $220]
           → keeps brand+os+ram filters, adds price_usd_max=220
User:      and bigger battery
System:    → keeps all prior filters, adds battery_capacity_min=5000 (previous max)
User:      forget about Samsung, any brand
System:    → removes brand filter (explicit removal), keeps the rest
```
**Information needed:** the previous recommendation's stats (min/max/median per attribute), stored in `last_recommend_stats`. The system anchors relative terms against those values and treats refines as ADDITIVE unless the user says "forget", "change", "I don't care about X".

### 4. Vague language (user uses qualitative terms)
```
User:      I want a cheap Android phone
System:    [maps "cheap" → price_usd_max = p25 of smartphone catalog = $147]
           → recommends 138 candidates under $147

User:      I want a premium iPhone with a great camera
System:    [maps "premium" → price_usd_min = p75 = $411; "great camera" → primary_camera_rear_min = p75 = 64 MP]
           → applies both filters
```
**Information needed:** the dataset's distribution percentiles. Computed once on startup by `database.vague_term_thresholds(category)` and injected into the intent-extraction prompt, so the LLM never invents thresholds.

### Cross-cutting capabilities
- **Category switching mid-conversation** ("actually, show me headphones") → wipes filters, restarts.
- **Recap on demand** ("what do you have so far?") → bot summarizes preferences in natural English.
- **Skip handling** ("any" / "doesn't matter") → attribute marked skipped, advances to next question.
- **Chitchat** ("good morning", "thanks") → warm acknowledgement + scope reminder, preserves all product state.
- **Done / restart** ("thanks, bye", "let's start over") → graceful close + auto-reset on next message.

### What information does the system need? (spec question)
At minimum:
1. **The product category** (smartphone / headphones) — gates schema selection.
2. **A set of structured filter values**, either directly extracted from user utterances (specific) or elicited via questions (explore).
3. **An anchor for relative critiques** — the previous recommendation's distribution, needed to ground "cheaper", "bigger", "better".
4. **A scoring function over multiple attributes** — used when many products match, to pick the top-N. Implemented as TOPSIS with per-category weights.
5. **Dataset-derived percentile thresholds** — needed to translate vague qualitative terms ("cheap", "premium") into concrete filter bounds.

Everything else (which questions to ask in what order, when to recommend vs ask, how to present results) is downstream of these.

---

## Extending the System

- **Add a product category**: Add a CSV to `datasets/`, register it in `database.py:load_all()` (and add a column rename map if column names need normalising), extend `QUESTION_ORDER` with the desired clarification sequence, define a weight map for `top_n_by_score`, add per-attribute hints to `ATTRIBUTE_HINTS` in `nodes.py`, add the new filter keys to the extraction prompt, and add a renderer branch in `app.py:_render_comparison_table`.
- **Add a new intent**: Add it to the intent list in `nodes.py` and handle it in `retrieve_and_act_node`.
- **Add a new node**: Register it in `graph.py` and wire edges accordingly.
- **LangSmith traces**: Every `graph.invoke()` call is automatically traced at https://smith.langchain.com/
