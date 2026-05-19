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
└── datasets/             # Product CSVs
    ├── reduced_file_smartphone_500.csv   # 500 smartphones (price in USD)
    └── reduced_file_headphones_500.csv   # 500 headphones (price in USD)
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

## Example Conversations

**Exploring:**
> "I'm looking for a mid-range Android phone with a good camera"

**Specific:**
> "I need the cheapest Samsung with at least 8GB RAM"

**Refining:**
> "Actually, show me something cheaper" or "I'd prefer over-ear headphones with noise cancellation"

---

## Extending the System

- **Add a product category**: Add a CSV to `datasets/`, register it in `database.py:load_all()` (and add a column rename map if column names need normalising), extend `QUESTION_ORDER` with the desired clarification sequence, define a weight map for `top_n_by_score`, add per-attribute hints to `ATTRIBUTE_HINTS` in `nodes.py`, add the new filter keys to the extraction prompt, and add a renderer branch in `app.py:_render_comparison_table`.
- **Add a new intent**: Add it to the intent list in `nodes.py` and handle it in `retrieve_and_act_node`.
- **Add a new node**: Register it in `graph.py` and wire edges accordingly.
- **LangSmith traces**: Every `graph.invoke()` call is automatically traced at https://smith.langchain.com/
