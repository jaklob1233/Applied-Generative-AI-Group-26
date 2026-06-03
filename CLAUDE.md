# Applied Generative AI — Group 26: Conversational Recommender

A LangGraph-based conversational product recommender built as a university project. Users chat with an LLM-powered shop assistant to find smartphones or headphones through guided dialogue and TOPSIS-ranked results.

## Running the app

```bash
# Install dependencies (once)
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Configure API key in .env
OPENROUTER_API_KEY=sk-or-v1-...
LLM_PROVIDER=openrouter
OPENROUTER_MODEL=openai/gpt-4o-mini

# Launch Streamlit UI
streamlit run app.py           # http://localhost:8501

# Or headless CLI for quick debugging
python test_cli.py
```

## Architecture

The system uses a **4-node linear LangGraph pipeline** executed once per user turn. The design principle: LLM handles natural language, deterministic Python handles all business logic.

```
user message
    → intent_and_extract_node   (LLM)  — classify intent + extract filters
    → state_updater_node        (pure Python) — merge filters, handle category switch
    → retrieve_and_act_node     (pure Python) — query DB, decide next action
    → response_generator_node   (LLM)  — generate natural-language reply
    → response displayed in Streamlit
```

### Key files

| File | Role |
|------|------|
| `app.py` | Streamlit UI — chat, sidebar debug panel, product cards, comparison table, quick-reply buttons |
| `graph.py` | Builds LangGraph `StateGraph`; `run_turn(state, msg)` is the public API |
| `nodes.py` | Implements the four nodes; holds `ATTRIBUTE_HINTS` and prompt templates |
| `state.py` | `DialogueState` TypedDict + `initial_state()` factory |
| `database.py` | CSV loading, `retrieve()` filter engine, TOPSIS scoring (`score_candidates`, `top_n_by_score`, `score_breakdown`) |
| `datasets/` | `reduced_file_smartphone_500.csv` and `reduced_file_headphones_500.csv` (~500 products each) |
| `eval/intent_eval.py` | Intent classification evaluation — 41 labelled cases, accuracy report + PNG chart |

## Dialogue state (`state.py`)

`DialogueState` carries all per-session data:

- `messages` — full conversation history
- `intent` — last classified intent: `explore | specific | refine | done | chitchat`
- `category` — `smartphones | headphones | None`
- `active_filters` — merged dict of all confirmed filters (persists across turns)
- `extracted_filters` — filters from the most recent turn only
- `action` — decision output of `retrieve_and_act_node`: `ask_category | ask_clarification | recommend | no_results | done | chitchat`
- `candidates` — list of matching products (added `_score` by TOPSIS)
- `clarification_attribute` — which attribute to ask about next
- `asked_skipped` — set of attributes the user skipped

## Filter engine (`database.py`)

`retrieve(category, filters, limit)` applies structured filters to the pandas DataFrame:

- Exact match: `{"os": "Android"}`
- Range: `{"price_usd_min": 200, "price_usd_max": 600}`
- Substring: `{"brand_name_contains": "samsung"}`

Fixed question order (`QUESTION_ORDER`) drives clarification — no dynamic selection:

- **Smartphones:** os → price_usd → battery_capacity → primary_camera_rear → ram_capacity → internal_memory
- **Headphones:** type → form_factor → noise_cancellation → price_usd

Headphone CSV columns are normalized on load via `_HEADPHONE_COLUMN_MAP`.

## TOPSIS scoring

`score_candidates(df, category)` ranks products by multi-attribute closeness to an ideal (high specs, low price). Weights are defined in `TOPSIS_WEIGHTS` per category. `score_breakdown()` returns a per-attribute transparency table for display in the UI.

## Action policy

`retrieve_and_act_node` decides the action purely from state (no LLM):

1. No category → `ask_category`
2. No products match → `no_results`
3. ≤ 2 matching products, or all questions asked → `recommend`
4. Otherwise → `ask_clarification` (next unasked, unskipped attribute)

## UI features (`app.py`)

- **Sidebar:** live category, turn count, intent, active filters, matching product count, raw JSON debug
- **Product cards:** top-3 auto-expanded expanders with specs + TOPSIS score breakdown + compare checkboxes
- **Comparison table:** side-by-side spec view for selected products
- **Quick-reply buttons:** per-attribute clickable suggestions (e.g. `Android · iOS · Any`)

## LLM configuration

Configured via `.env`. Currently defaults to OpenRouter → GPT-4o-mini. All providers use the `ChatOpenAI` interface:

```
LLM_PROVIDER=openrouter | openai | anthropic
OPENROUTER_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY
OPENROUTER_MODEL=<model slug>
```

## Evaluation (`eval/`)

### Intent classification — `eval/intent_eval.py`

41 hand-labelled utterances across all 5 intents, covering cold-start, mid-session, and edge cases (negation, skip signals, multi-attribute refinements). Baseline with GPT-4o-mini: **95.1% accuracy** (39/41).

```bash
python eval/intent_eval.py                # text report, failures only
python eval/intent_eval.py --verbose      # show all 41 cases
python eval/intent_eval.py --plot         # save PNG chart to eval/intent_eval.png
python eval/intent_eval.py --threshold 85 # custom pass threshold (default 80%)
```

Exits non-zero if accuracy < threshold. The 2 failures are borderline `explore` cases the model classifies as `specific` (utterances with concrete attributes but no exact model name) — a genuine intent boundary ambiguity, not a clear model error.

### Filter extraction — `eval/filter_eval.py`

40 hand-labelled `(utterance, expected_filters)` pairs across 7 groups: simple single-field, explicit ranges, vague terms, multi-attribute, skip signals, relative critiques, and filter removal. Measures field-level precision, recall, and F1. Baseline on `24-real-crs` with GPT-4o-mini: **F1 72.1%, exact match 67.5%**.

```bash
python eval/filter_eval.py                # text report, failures only
python eval/filter_eval.py --verbose      # show all 40 cases
python eval/filter_eval.py --plot         # save PNG chart to eval/filter_eval.png
python eval/filter_eval.py --threshold 80 # custom pass threshold (default 75%)
```

**Bugs found in `24-real-crs` by this eval:**

1. **Null field flooding** — for simple utterances the LLM returns every possible filter field set to `null` (e.g. `"I want Android"` → 17 fields, only `os` is non-null). The state updater treats null as "remove this filter", so filters set in a previous turn get silently wiped on the next turn. Fix: instruct the prompt to only return fields the user explicitly mentioned, never null-pad the output.

2. **Explicit prices overridden by vague-term block** — `"under $400"` → `price_usd_max: 411` (dataset p75) and `"headphones under $200"` → `price_usd_max: 140` (dataset p25). The vague-term grounding block is overriding literal dollar amounts. Fix: add a rule that explicit numeric values always take precedence over percentile mappings.

3. **Skip signals output `{field: null}` instead of `{}`** — `"Any OS is fine"` → `{os: null}` instead of an empty dict. In context this is a no-op if the field was never set, but it removes the filter if it was. Fix: clarify in the prompt that skipping means omitting the field entirely, not setting it to null.

4. **iPhone → `brand_name: 'iphone'` not `os: 'ios'`** — the model treats "iPhone" as a brand/model substring rather than implying iOS. Minor — arguably a labelling ambiguity in the test.

### Next evaluation dimensions (not yet implemented)

- **End-to-end dialogue / task success** — simulate scripted personas across full multi-turn conversations; measure whether the final recommendation matches the persona's hidden constraints and how many turns it took

## Not yet implemented (product features)

- Semantic/embedding-based fuzzy filter grounding (described in `proposal.md`)
- Persistent user sessions across page reloads
- Product images
- Additional categories beyond smartphones and headphones

## Branch conventions

- `main` — stable releases
- `dev` — integration branch; merge feature branches here first
- `23-evaluation` — current branch; evaluation suite work
