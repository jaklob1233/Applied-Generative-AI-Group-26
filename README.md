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
    ├── reduced_file_smartphone_500.csv   # 500 smartphones (price in INR)
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
- `active_filters`: merged structured preferences (e.g. `{price_max: 30000, brand_name: "samsung"}`)
- `candidates`: current matching products
- `action`: what the system decided to do this turn
- `messages`: full conversation history

### Supported Filter Keys

**Smartphones** (price in INR):

| Key | Type | Example |
|-----|------|---------|
| `price_min` / `price_max` | int | 30000 |
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

- **Add a product category**: Add a CSV to `datasets/`, register it in `database.py:load_all()` (and add a column rename map if column names need normalising), extend the `ASKABLE` map, and add the new filter keys to `nodes.py`'s extraction prompt.
- **Add a new intent**: Add it to the intent list in `nodes.py` and handle it in `retrieve_and_act_node`.
- **Add a new node**: Register it in `graph.py` and wire edges accordingly.
- **LangSmith traces**: Every `graph.invoke()` call is automatically traced at https://smith.langchain.com/
