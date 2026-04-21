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
├── generate_data.py      # Generate synthetic product CSVs
│
└── products/             # Auto-created by generate_data.py
    ├── smartphones.csv
    ├── washing_machines.csv
    └── laptops.csv
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

### 3. Generate product data
```bash
python generate_data.py
```
This calls the LLM to generate ~210 synthetic products across 3 categories and saves them to `products/`.

### 4. Run the app

**Streamlit UI (recommended):**
```bash
streamlit run app.py
```

**CLI test (faster for debugging):**
```bash
python test_cli.py
```

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
- `category`: smartphone | washing_machine | laptop
- `active_filters`: merged structured preferences (e.g. `{price_eur_max: 500, brand: "Samsung"}`)
- `candidates`: current matching products
- `action`: what the system decided to do this turn
- `messages`: full conversation history

### Supported Filter Keys
| Key | Type | Example |
|-----|------|---------|
| `price_eur_max` / `price_eur_min` | int | 500 |
| `brand` | str | "Samsung" |
| `os` | str | "Android" |
| `ram_gb` | int | 8 |
| `storage_gb` | int | 256 |
| `battery_mah_min` | int | 4000 |
| `has_5g` | bool | true |
| `capacity_kg` | int | 8 (washing machines) |
| `energy_class` | str | "A" (washing machines) |
| `load_type` | str | "front" (washing machines) |
| `gpu_type` | str | "dedicated" (laptops) |
| `weight_kg_max` | float | 1.5 (laptops) |

---

## Example Conversations

**Exploring:**
> "I'm looking for a mid-range Android phone with a good camera"

**Specific:**
> "I need the cheapest Samsung with at least 8GB RAM"

**Refining:**
> "Actually, show me something cheaper" or "I prefer front-loading machines"

---

## Extending the System

- **Add a product category**: Add a CSV to `products/`, register it in `database.py:load_all()`, and add filter keys to `nodes.py`'s extraction prompt.
- **Add a new intent**: Add it to the intent list in `nodes.py` and handle it in `retrieve_and_act_node`.
- **Add a new node**: Register it in `graph.py` and wire edges accordingly.
- **LangSmith traces**: Every `graph.invoke()` call is automatically traced at https://smith.langchain.com/
