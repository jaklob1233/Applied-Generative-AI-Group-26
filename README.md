# Conversational Recommender System

A LangGraph-powered chatbot that acts as a shop assistant for tech products.

## Project Structure

```
crs_project/
├── requirements.txt      # Python dependencies
├── .env.example          # Copy to .env and fill in your keys
│
├── state.py              # DialogueState TypedDict definition
├── database.py           # Product CSV loader + filter engine
├── nodes.py              # LangGraph node functions
├── graph.py              # Graph assembly + run_turn() API
│
├── app.py                # Streamlit chat UI  ← main entry point
├── test_cli.py           # CLI test (no UI needed)
│
├── datasets/             # Product CSVs
│   ├── reduced_file_smartphone_500.csv   # 500 smartphones (price in USD)
│   └── reduced_file_headphones_500.csv   # 500 headphones (price in USD)
│
└── eval/                 # Evaluation suite
    └── intent_eval.py    # Intent classification accuracy (41 labelled cases)
```

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

## Evaluation

The `eval/` directory contains automated evaluation scripts that measure system quality against hand-labelled golden datasets.

### Intent Classification (`eval/intent_eval.py`)

Runs 41 labelled utterances through `intent_and_extract_node` and reports per-class accuracy and a confusion matrix. Cases cover all five intents (`explore`, `specific`, `refine`, `done`, `chitchat`) across cold-start and mid-session contexts, including edge cases such as negation, skip signals, and multi-attribute refinements.

```bash
# Text report (failures only)
python eval/intent_eval.py

# Text report + save PNG chart to eval/intent_eval.png
python eval/intent_eval.py --plot

# Show every case, not just failures
python eval/intent_eval.py --verbose

# Custom pass threshold (default 80%)
python eval/intent_eval.py --threshold 85
```

Exits with code 0 if accuracy ≥ threshold, 1 otherwise — suitable for CI.

**Baseline results** (GPT-4o-mini via OpenRouter):

| Intent | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| explore | 8 | 10 | 80% |
| specific | 7 | 7 | 100% |
| refine | 11 | 11 | 100% |
| done | 5 | 5 | 100% |
| chitchat | 8 | 8 | 100% |
| **overall** | **39** | **41** | **95.1%** |

The two `explore` misclassifications are borderline cases where the user's utterance contains concrete attributes (OS, price range) that the model interprets as `specific`. They highlight a genuine boundary ambiguity in the intent definitions rather than a clear model error.

---

## Extending the System

- **Add a product category**: Add a CSV to `datasets/`, register it in `database.py:load_all()` (and add a column rename map if column names need normalising), extend `QUESTION_ORDER` with the desired clarification sequence, define a weight map for `top_n_by_score`, add per-attribute hints to `ATTRIBUTE_HINTS` in `nodes.py`, add the new filter keys to the extraction prompt, and add a renderer branch in `app.py:_render_comparison_table`.
- **Add a new intent**: Add it to the intent list in `nodes.py` and handle it in `retrieve_and_act_node`.
- **Add a new node**: Register it in `graph.py` and wire edges accordingly.
- **LangSmith traces**: Every `graph.invoke()` call is automatically traced at https://smith.langchain.com/
