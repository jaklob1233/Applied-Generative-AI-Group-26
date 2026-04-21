Conversational Recommender System
Revised Project Documentation v2.0

Course: Applied Generative AI
Change Log from v1.0: Semantic retrieval layer added; Intent Classifier separated from Agent Node; dynamic preference weighting added; dataset expanded; thought traces demoted to debug-only; hybrid recommendation pipeline introduced.
Table of Contents

    Project Overview
    System Requirements & Use Cases
    Revised Architecture Design
    Data Design
    Semantic Layer Design
    Dialogue System Design
    Component Specifications
    LangGraph Workflow Specification
    Evaluation Framework
    Implementation Plan
    Risk Assessment
    Appendices

1. Project Overview
1.1 Problem Statement

Traditional recommender systems operate silently in the background, presenting ranked lists based on historical behaviour or collaborative filtering. This passive approach fails when users have vague, evolving, or hard-to-articulate preferences — a situation especially common in technical product domains.

A Conversational Recommender System (CRS) addresses this by engaging users in structured, multi-turn dialogue that progressively clarifies needs, maps them to structured product attributes, and returns targeted recommendations that can be further refined through natural conversation.

This project builds a hybrid CRS that combines:

    Symbolic reasoning — structured dialogue state, rule-based filtering, deterministic scoring
    Neural semantic understanding — product embeddings, semantic similarity search, LLM-powered reasoning

The system operates across three product categories: smartphones, laptops, and washing machines.
1.2 System Philosophy

The system follows a constrained hybrid architecture:

text

┌─────────────────────────────────────────────────────────┐
│              WHAT THE LLM IS RESPONSIBLE FOR            │
│                                                         │
│  • Classifying user intent                              │
│  • Extracting structured preferences from natural text  │
│  • Interpreting vague/fuzzy user expressions            │
│  • Generating natural language responses                │
│  • Semantic matching of user needs to products          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│         WHAT DETERMINISTIC LOGIC IS RESPONSIBLE FOR     │
│                                                         │
│  • Enforcing hard constraints (never delegated to LLM)  │
│  • Scoring and ranking products                         │
│  • Managing dialogue state transitions                  │
│  • Selecting next questions                             │
│  • Constraint relaxation decisions                      │
│  • Session lifecycle management                         │
└─────────────────────────────────────────────────────────┘

This separation is not a weakness — it is correct production GenAI architecture. The LLM handles what it does best (language understanding and generation); deterministic logic handles what requires reliability and explainability (filtering, ranking, state management).
1.3 What Makes This a GenAI System

The following components distinguish this from a classical rule-based recommender:

    Semantic preference interpretation — "good camera" is not mapped to camera_mp > threshold by a rule; it is resolved via embedding similarity against product descriptions
    Fuzzy linguistic grounding — "cheap", "lightweight", "powerful" are interpreted relative to the current product space using LLM reasoning, not hardcoded thresholds
    LLM-powered intent classification — handles the full diversity of natural language expressions for the same intent
    Semantic re-ranking — products are re-ranked by embedding similarity to the user's stated needs, not just by structured attribute scores
    Natural language explanation generation — response generation uses LLM to produce contextually appropriate, varied explanations grounded in product data

1.4 Scope & Constraints

In scope:

    Three product categories: smartphones, laptops, washing machines
    Text-based conversational interface with optional UI helper elements
    Product database of approximately 50 items per category
    Product embeddings stored in a lightweight local vector store
    Hybrid recommendation: structured filtering + semantic re-ranking
    Single-session dialogue (no cross-session memory)
    Quantitative evaluation on scripted test dialogues

Out of scope:

    Real-time product price feeds or e-commerce integration
    Cross-session user profiling or persistent personalisation
    Multi-language support
    Voice interface
    Online learning or model fine-tuning
    Full neural end-to-end recommendation without structured filtering

2. System Requirements & Use Cases
2.1 Functional Requirements
ID	Requirement	Priority
FR-01	The system shall classify each user utterance into one of five defined intent classes	Must Have
FR-02	The system shall extract structured product attributes from natural language utterances	Must Have
FR-03	The system shall interpret fuzzy linguistic expressions relative to the current product space	Must Have
FR-04	The system shall maintain a structured dialogue state across all turns of a session	Must Have
FR-05	The system shall select a system action at each turn based on intent and dialogue state	Must Have
FR-06	The system shall recommend up to three ranked products with explanations	Must Have
FR-07	The system shall use both structured filtering and semantic similarity in ranking	Must Have
FR-08	The system shall ask targeted follow-up questions when preferences are insufficient	Must Have
FR-09	The system shall update recommendations in response to refinement requests	Must Have
FR-10	The system shall relax soft constraints and notify the user when no products match	Must Have
FR-11	The system shall detect and resolve contradictory preferences across turns	Should Have
FR-12	The system shall dynamically adjust scoring weights based on user-expressed priorities	Should Have
FR-13	The system shall allow users to reset a session and start over	Should Have
FR-14	The system shall log all state transitions and LLM calls for observability	Should Have
FR-15	The system shall support quick-action buttons in addition to free text	Could Have
FR-16	The system shall display product images and key specifications in the UI	Could Have
2.2 Non-Functional Requirements
ID	Requirement	Target
NFR-01	Response latency per turn	< 6 seconds
NFR-02	Intent classification accuracy on test set	≥ 80%
NFR-03	Preference extraction slot accuracy on test set	≥ 75%
NFR-04	Semantic retrieval relevance (human-judged)	≥ 70%
NFR-05	System must remain functional when 0 products match filters	Always
NFR-06	All hard constraints must be satisfied in every recommendation	Always
NFR-07	Product database must have minimum 50 items per category	Always
NFR-08	LLM reasoning traces are for debugging only; not used in evaluation	Always
2.3 Use Cases
Use Case 1: Direct Product Search

Actor: User who knows exactly what they want
Trigger: User names a specific product or very specific configuration


User:    "I want the Samsung Galaxy S24 Ultra"
System:  "Here is the Samsung Galaxy S24 Ultra:
          Price: €1,299 | Camera: 200MP | Battery: 5,000mAh
          ✓ Matches your request exactly
          Would you like to compare it with similar alternatives?"

Use Case 2: Guided Exploration with Fuzzy Preferences

Actor: User with vague preferences
Trigger: User expresses interest in a category with imprecise language


User:    "I need a laptop that's good for creative work, 
          something powerful but not too heavy"
System:  "Got it — a powerful yet portable creative laptop.
          What's your budget range?"
User:    "Somewhere around €1,200 to €1,500"
System:  "Do you have a preference for macOS or Windows?"
User:    "macOS please"
System:  "Here are my top picks for creative work on macOS:
          1. MacBook Pro 14" M3 (€1,499) — 1.55kg, M3 chip
             ✓ Exceptional performance for creative tasks
             ✓ Within your budget
          2. ..."

Note: "powerful but not too heavy" is resolved semantically against product embeddings, not by a fixed weight threshold.
Use Case 3: Critique-Based Refinement with Trade-offs

Actor: User who wants to adjust results
Trigger: User expresses a trade-off or critique


User:    "These are good but I'd like something with a 
          better camera, even if it costs a bit more"
System:  "Understood — I'll prioritise camera quality and 
          relax the budget slightly. Here are updated options:
          [New recommendations prioritising camera score]"

Note: "better camera but costs a bit more" requires understanding a trade-off, not just a simple filter update.
Use Case 4: Fuzzy Expression Grounding

Actor: User using relative or vague language
Trigger: User uses terms like "cheap", "powerful", "quiet"


User:    "I want a cheap washing machine that's quiet"
System:  [Internally: resolves "cheap" as bottom 30% of 
          price range in washing machine database;
          resolves "quiet" as noise_db ≤ category median]
         "For a quiet, budget-friendly washing machine,
          here are my recommendations:
          1. Beko WTL72051W (€299) — 42dB
          ..."

Use Case 5: Zero-Result Constraint Relaxation

Actor: Any user
Trigger: Applied filters return zero matching products


User:    "I need an Android phone under €200 with 
          at least 256GB storage"
System:  "I couldn't find Android phones under €200 with 
          256GB storage in our catalog. 
          The storage requirement is quite restrictive at 
          this price point.
          If I relax the storage to 128GB, I can show you 
          3 options. Would that work?"

Use Case 6: Category Switch Mid-Session

Actor: User who changes product category
Trigger: User mentions a different category


User:    "Actually, forget the laptop. I need a 
          washing machine instead"
System:  "No problem — switching to washing machines.
          Your laptop preferences have been cleared.
          How large is your household? 
          This helps me recommend the right drum size."


3. Revised Architecture Design
3.1 High-Level System Architecture


┌──────────────────────────────────────────────────────────────────┐
│                      PRESENTATION LAYER                          │
│                   Streamlit Web Application                      │
│     Chat Interface │ Quick Buttons │ Product Cards │ Sliders     │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                       SESSION LAYER                              │
│                      Session Manager                             │
│          DialogueState persistence and lifecycle                 │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                   ORCHESTRATION LAYER                            │
│                   LangGraph Workflow                             │
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────────────┐  │
│  │   Intent     │   │    Policy    │   │  Execution Nodes    │  │
│  │  Classifier  │──▶│    Module    │──▶│  (per action type)  │  │
│  │    (LLM)     │   │   (Rules)    │   │                     │  │
│  └──────────────┘   └──────────────┘   └─────────────────────┘  │
│                                                                  │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                    ┌───────────┴────────────┐
                    │                        │
                    ▼                        ▼
┌───────────────────────────┐  ┌─────────────────────────────────┐
│   SEMANTIC LAYER          │  │      DATA ACCESS LAYER          │
│   Vector Store            │  │      Recommendation Engine      │
│   (product embeddings)    │  │   Filtering │ Scoring │ Ranking │
│   Semantic Similarity     │  │   Constraint Relaxation         │
│   Search                  │  │   Explanation Generation        │
└───────────────┬───────────┘  └──────────────┬──────────────────┘
                │                             │
                └──────────────┬──────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
│           Product Database (CSV / SQLite)                        │
│           Product Descriptions (for embedding)                   │
│           Pre-computed Embedding Index (FAISS / ChromaDB)        │
└──────────────────────────────────────────────────────────────────┘

Observability: LangSmith traces all LLM calls and node transitions

3.2 Key Architectural Change: Separated Intent Classifier

In v1.0, the Agent Node performed both intent classification and action selection. This created a single point of failure and made debugging harder.

Revised design splits this into two components:

v1.0 Architecture:
User Message → [Agent Node: intent + action selection] → Router

v2.0 Architecture:
User Message → [Intent Classifier Node: LLM] → [Policy Module: Rules] → Router

Benefits:

    Intent classification failures are isolated and easier to debug
    Policy logic is fully transparent and testable without LLM calls
    Intent classification can be evaluated independently
    The Policy Module encodes dialogue management rules explicitly, making them reviewable by graders

3.3 Key Architectural Addition: Semantic Layer

A lightweight semantic layer sits alongside the structured recommendation engine:

User's Natural Language Needs
           │
           ▼
    Semantic Query Encoder
    (embed user's stated needs)
           │
           ▼
    Vector Similarity Search
    (against product description embeddings)
           │
           ▼
    Semantic Candidate Set
    (top-N semantically similar products)
           │
           ▼
    Hybrid Fusion                ◄── Structured Filtered Candidates
    (combine semantic + filtered)
           │
           ▼
    Final Ranked Recommendations

This is a lightweight form of Retrieval-Augmented Generation applied to product recommendation. The semantic layer does not replace structured filtering — it adds a semantic re-ranking signal on top of it.

3.4 Per-Turn Request Flow

Step 1:  User submits message via Streamlit
Step 2:  Session Manager retrieves DialogueState
Step 3:  Intent Classifier Node (LLM) → IntentResult
Step 4:  Policy Module (deterministic) reads intent + state → Action
Step 5:  Router directs to appropriate execution node
Step 6a: If ASK_CLARIFICATION → Question Selector → Ask Node
Step 6b: If EXTRACT_PREFERENCES → Fuzzy Grounding + Extraction Node
Step 6c: If RECOMMEND_ITEMS → Semantic Query + Hybrid Recommendation Node
Step 6d: If REFINE_RESULTS → Extraction Node + Dynamic Weight Update + Recommendation Node
Step 6e: If SUMMARIZE → Summarize Node
Step 6f: If HANDLE_ERROR → Error Handler Node
Step 7:  State Update Node merges new preferences
Step 8:  Response Generator Node → natural language response
Step 9:  Session Manager saves updated DialogueState
Step 10: Response displayed in Streamlit
Step 11: LangSmith logs full turn trace

3.5 Technology Stack
Layer	Technology	Version	Justification
UI	Streamlit	≥ 1.32	Rapid prototyping; native chat components
Orchestration	LangGraph	≥ 0.1	Stateful graph-based dialogue management
LLM — Primary	GPT-4o-mini	Latest	Intent classification, extraction, response generation
LLM — Selective	GPT-4o	Latest	Complex fuzzy grounding turns; trade-off reasoning
Embeddings	OpenAI text-embedding-3-small	Latest	Product description embedding; user query embedding
Vector Store	ChromaDB (local)	≥ 0.4	Lightweight; no infrastructure; persistent local index
Schema Validation	Pydantic v2	≥ 2.0	Structured output validation throughout pipeline
Data Processing	pandas	≥ 2.0	Product database filtering and scoring
Database	CSV + optional SQLite	—	Transparent; easy to inspect and modify
Observability	LangSmith	Latest	Trace all LLM calls and state transitions
Python	3.11+	≥ 3.11	Full library compatibility

4. Data Design
4.1 Product Database Schema
4.1.1 Universal Fields
Field	Type	Description	Example
product_id	string	Unique identifier	"SP001"
name	string	Full product name	"Samsung Galaxy S24 Ultra"
brand	string	Manufacturer	"Samsung"
price_eur	float	Retail price in EUR	1299.00
release_year	integer	Year of release	2024
image_url	string	Product image URL	"https://..."
availability	boolean	In stock	True
description	string	Natural language product description for embedding	"A premium Android flagship with a 200MP camera..."

Note on the description field: This is new in v2.0. Each product has a human-written natural language description covering its key characteristics, target use case, and standout features. This field is used to generate product embeddings. The description should be 3–5 sentences and written in the same vocabulary users would naturally use (e.g., "great for photography", "ideal for frequent travellers", "very quiet for an open-plan home").

4.1.2 Smartphone-Specific Fields
Field	Type	Description	Example
os	string	Operating system	"Android" / "iOS"
storage_gb	integer	Internal storage	256
ram_gb	integer	RAM	12
battery_mah	integer	Battery capacity	5000
main_camera_mp	integer	Main camera megapixels	200
screen_size_inch	float	Display diagonal	6.8
five_g	boolean	5G support	True
weight_g	integer	Weight in grams	232
4.1.3 Laptop-Specific Fields
Field	Type	Description	Example
os	string	Operating system	"Windows 11" / "macOS"
ram_gb	integer	RAM	16
storage_gb	integer	SSD storage	512
processor_tier	string	Simplified CPU tier	"high" / "mid" / "entry"
gpu_dedicated	boolean	Dedicated GPU	False
screen_size_inch	float	Display diagonal	13.4
weight_kg	float	Weight in kg	1.20
battery_hours	integer	Battery life estimate	12
primary_use	string	Recommended use	"business" / "gaming" / "creative" / "general"
4.1.4 Washing Machine-Specific Fields
Field	Type	Description	Example
capacity_kg	float	Drum capacity	8.0
energy_rating	string	EU energy label	"A"
noise_db	integer	Operating noise	48
loading_type	string	Front or top	"front"
spin_rpm	integer	Max spin speed	1400
steam_function	boolean	Steam wash	True
4.1.5 Pre-computed Score Fields
Field	Description	How Computed
value_score	Price-to-performance ratio (0–1)	Normalised: lower price + higher performance = higher score
performance_score	Category-specific performance (0–1)	Weighted sum of key performance specs, min-max normalised
feature_score	Feature breadth relative to peers (0–1)	Count of premium features, normalised
popularity_score	Simulated editorial rating (0–1)	Manually assigned based on real-world reviews

These scores are computed once during data preparation. They are never computed at runtime.
4.2 Data Volume
Category	Target Products	Rationale
Smartphones	50	Sufficient diversity across price tiers, OS, brands
Laptops	50	Sufficient for use-case and OS combinations
Washing Machines	45	Adequate range of capacity, energy ratings, loading types
Total	145	Credible evaluation; realistic filtering behaviour
4.3 Product Description Examples

These descriptions are the input to the embedding model.

Smartphone example:

    "The Samsung Galaxy S24 Ultra is a premium Android flagship designed for power users and photography enthusiasts. It features a 200MP main camera capable of exceptional detail even in low light, a massive 5,000mAh battery built for all-day use, and a built-in S Pen for productivity. At 232 grams it is on the heavier side but delivers unmatched versatility. Best suited for users who want the absolute best camera on Android without compromise."

Laptop example:

    "The Apple MacBook Air M2 is an ultra-thin, fanless laptop ideal for students and professionals who prioritise portability and battery life. The M2 chip delivers impressive performance for everyday productivity and light creative work. At just 1.24kg it is one of the lightest MacBooks available, and the battery lasts up to 18 hours on a charge. It runs macOS and is particularly well suited to users already in the Apple ecosystem."

Washing Machine example:

    "The Miele WCR870 WPS is a premium front-loading washing machine designed for families and users who demand quiet, energy-efficient laundry. With a noise level of just 46dB during washing, it is exceptionally quiet and well suited to open-plan homes or flats where noise is a concern. The 9kg drum comfortably handles large family loads, and the A energy rating keeps running costs low. It includes a steam function for hygiene-sensitive washing."

4.4 Data Preparation Pipeline

text

Step 1: Collect raw specs from public sources (manufacturer sites, 
        GSMArena, Notebookcheck, retailer sites)
Step 2: Standardise all units and field names
Step 3: Write natural language descriptions for each product (3–5 sentences)
Step 4: Compute normalised score columns
Step 5: Validate schema (automated script checks all required fields)
Step 6: Generate embeddings for all product descriptions
Step 7: Store embeddings in ChromaDB local collection (one per category)
Step 8: Save product records as CSV files in /data directory
Step 9: Verify embedding index is aligned with CSV records (same product IDs)

5. Semantic Layer Design

This section is new in v2.0.
5.1 Purpose and Role

The semantic layer serves three distinct functions:

Function 1 — Fuzzy Expression Grounding
When a user says "cheap", "powerful", "quiet", or "good camera", these expressions cannot be reliably mapped to fixed numeric thresholds. The semantic layer resolves these relative to the current product space.

Function 2 — Semantic Product Retrieval
When a user describes a need in natural language ("something good for travelling photographers"), the semantic layer retrieves products whose descriptions are most similar to this query, independent of structured attribute matching.

Function 3 — Semantic Re-ranking
After structured filtering, the semantic layer re-ranks the remaining candidates by their semantic similarity to the user's aggregated stated needs, providing a richer signal than weighted attribute scores alone.
5.2 Embedding Strategy
5.2.1 What is Embedded
Document Type	Content	When Generated
Product descriptions	Full natural language description per product	Once, at data preparation time
User query	Concatenation of all user-stated preferences and needs so far this session	At recommendation time, per turn
5.2.2 Embedding Model

Model: OpenAI text-embedding-3-small

    Dimension: 1536
    Cost: very low (approximately $0.02 per 1M tokens)
    Sufficient quality for this scope; no fine-tuning required

5.2.3 Vector Store

Technology: ChromaDB (local persistent)

    One collection per product category: smartphones, laptops, washing_machines
    Each document in the collection: product description text + product_id as metadata
    Stored locally; no server infrastructure required
    Index built once during data preparation; loaded at application startup

5.3 Fuzzy Expression Grounding

When the Preference Extraction Node encounters a fuzzy linguistic expression, it follows this process:

text

User says: "I want a cheap phone"

Step 1: Detect that "cheap" is a relative expression 
        (not a numeric value)

Step 2: Query the smartphone database for the current 
        price distribution:
        - Min price: €149
        - 25th percentile: €299
        - Median: €499
        - 75th percentile: €799
        - Max price: €1,299

Step 3: Apply grounding rule:
        "cheap"     → price ≤ 25th percentile (€299)
        "budget"    → price ≤ 33rd percentile
        "mid-range" → price between 33rd and 66th percentile
        "premium"   → price ≥ 75th percentile
        "expensive" → price ≥ 75th percentile

Step 4: Store grounded value as SOFT constraint:
        price_max = 299 (derived from "cheap", can be relaxed)

Step 5: Log the grounding decision for LangSmith trace

Grounding rules by expression type:
Expression	Attribute	Grounding Rule
"cheap", "affordable", "budget"	price_max	≤ 25th percentile of category
"mid-range", "moderate"	price range	33rd–66th percentile of category
"premium", "high-end", "expensive"	price_min	≥ 75th percentile of category
"good battery", "long battery life"	min_battery	≥ 75th percentile of category
"lightweight", "light", "portable"	max_weight	≤ 25th percentile of category
"powerful", "fast", "high performance"	performance_score	≥ 0.7
"quiet", "silent"	max_noise_db	≤ 25th percentile of category
"large capacity", "big drum"	min_capacity_kg	≥ 75th percentile of category
"energy efficient", "eco"	energy_rating_min	A or B rating
"great camera", "good for photos"	Semantic retrieval signal	Camera-related similarity boost

Key design principle: Grounding rules are computed dynamically from the actual product distribution at runtime. They are not hardcoded thresholds. If the product database changes, grounding adapts automatically.
5.4 Semantic Retrieval Process

At recommendation time, the following process runs in parallel with structured filtering:

text

Step 1: QUERY CONSTRUCTION
        Concatenate all user-stated needs from current session:
        "Android phone with good camera and long battery life 
         under €600, lightweight"

Step 2: QUERY EMBEDDING
        Embed the constructed query using 
        text-embedding-3-small → 1536-dim vector

Step 3: SIMILARITY SEARCH
        Query ChromaDB collection for the active category
        Retrieve top-20 most similar products by cosine similarity
        Return: [(product_id, similarity_score), ...]

Step 4: CANDIDATE INTERSECTION
        Structured filtered candidates ∩ Semantic candidates
        = Hybrid candidate set
        
        If intersection is empty:
        → Fall back to structured filtered candidates only
        → Log that semantic signal was not used this turn

Step 5: SCORE FUSION
        For each product in hybrid candidate set:
        
        hybrid_score = α × structured_score + (1-α) × semantic_score
        
        Where:
        α = 0.6 (structured filtering weighted higher by default)
        1-α = 0.4 (semantic similarity contributes significantly)
        
        α is adjusted dynamically:
        - If user has used many fuzzy expressions → α decreases (more semantic weight)
        - If user has given precise numeric constraints → α increases (more structured weight)

Step 6: FINAL RANKING
        Sort by hybrid_score descending
        Return top-3 products

5.5 Dynamic Weight Adjustment

The α parameter (and internal scoring weights) are adjusted based on signals from the conversation:
Signal	Adjustment
User says "cheaper" in REFINE turn	Increase value_score weight
User says "better camera" in REFINE turn	Increase camera_score weight (smartphones)
User says "lighter" in REFINE turn	Increase portability_score weight (laptops)
User says "quieter" in REFINE turn	Increase noise_score weight (washing machines)
User uses many fuzzy expressions	Decrease α (increase semantic weight)
User gives precise numeric constraints	Increase α (increase structured weight)

These adjustments are stored in the preference_weights field of the DialogueState and applied in subsequent recommendation calls.
6. Dialogue System Design
6.1 Dialogue State
6.1.1 Complete State Definition

Session Metadata

    session_id — unique session identifier (UUID)
    turn_count — number of completed turns
    created_at — session creation timestamp

Intent Tracking

    current_intent — intent of the most recent user utterance
    intent_history — list of (turn_number, intent) pairs for all turns
    intent_confidence — confidence score returned by Intent Classifier for current turn

Category

    category — active product category (smartphone / laptop / washing_machine / None)
    category_confirmed — boolean; True after category has been confirmed with user

Preferences

    preferences — category-specific structured preference object (see 6.1.2)
    preference_weights — dynamic scoring weights for current session (dict, float values 0–1)
    fuzzy_expressions — list of fuzzy expressions used this session (for α adjustment)

Constraints

    hard_constraints — list of Constraint objects that must always be satisfied
    soft_constraints — list of Constraint objects that can be relaxed

Semantic State

    semantic_query — current aggregated natural language representation of user needs
    last_semantic_scores — dict of product_id → semantic similarity score from last retrieval

Conversation Management

    asked_attributes — list of attribute names already asked about
    shown_products — list of product IDs shown to user
    rejected_products — list of product IDs explicitly dismissed by user
    last_action — action selected in previous turn
    last_recommendations — product IDs shown in most recent recommendation
    relaxed_constraints — list of constraints that were relaxed in this session

History

    history — list of turn objects: {turn_number, role, message, timestamp}

6.1.2 Category-Specific Preference Objects

SmartphonePreferences
Field	Type	Notes
price_min	float or None	EUR
price_max	float or None	EUR
brand	string or None	
os	enum or None	iOS / Android
min_storage_gb	int or None	
min_ram_gb	int or None	
min_battery_mah	int or None	
min_camera_mp	int or None	
min_screen_size_inch	float or None	
max_screen_size_inch	float or None	
five_g_required	bool or None	
max_weight_g	int or None	

LaptopPreferences
Field	Type	Notes
price_min	float or None	EUR
price_max	float or None	EUR
brand	string or None	
os	enum or None	Windows / macOS / Linux
min_ram_gb	int or None	
min_storage_gb	int or None	
gpu_required	bool or None	
max_weight_kg	float or None	
min_battery_hours	int or None	
primary_use	enum or None	gaming / business / creative / general
min_screen_size_inch	float or None	
max_screen_size_inch	float or None	

WashingMachinePreferences
Field	Type	Notes
price_min	float or None	EUR
price_max	float or None	EUR
brand	string or None	
min_capacity_kg	float or None	
energy_rating_min	enum or None	A / B / C / D
max_noise_db	int or None	
loading_type	enum or None	front / top
min_spin_rpm	int or None	
steam_required	bool or None	
6.1.3 Constraint Object Structure

Each constraint is stored as a structured object:
Field	Type	Description	Example
attribute	string	The preference field this constrains	"price_max"
value	any	The constraint value	500.0
operator	enum	eq / lt / gt / lte / gte / ne	"lte"
constraint_type	enum	HARD / SOFT	"HARD"
confidence	float	Extraction confidence (0–1)	0.95
source_turn	int	Which turn produced this constraint	2
fuzzy_origin	string or None	Original fuzzy expression if grounded	"cheap"
6.2 Intent Classification
6.2.1 Intent Taxonomy
Intent	Code	Description	Example Utterances
Search	SEARCH	User wants a specific named product	"I want the iPhone 15 Pro", "Show me the Galaxy S24 Ultra"
Explore	EXPLORE	User wants help finding a product	"Help me find a good Android phone", "I need a quiet washing machine"
Refine	REFINE	User critiques or adjusts existing results	"Show me something cheaper", "Better battery please", "Not Samsung"
Chitchat	CHITCHAT	Small talk, greetings, off-topic questions	"Thanks!", "What do you think of iPhone?", "Hello"
Unknown	UNKNOWN	Cannot be mapped to supported intent	Gibberish, completely off-topic
6.2.2 Intent Classifier Node

The Intent Classifier is a dedicated LangGraph node that:

    Receives the user message and last 2 turns of history
    Invokes GPT-4o-mini with a structured classification prompt
    Returns a validated IntentResult (intent + confidence)
    Is entirely separate from action selection (which is handled by the Policy Module)

The prompt includes:

    Precise definition of all five intents
    Three labelled examples per intent (15 examples total)
    Instruction to return structured JSON: {intent, confidence, rationale}
    Instruction that rationale is for debugging only and is not used in decisions

Retry and fallback:

    If parsing fails after 3 attempts: return UNKNOWN with confidence 0.0
    If confidence < 0.4: return intent but flag for Policy Module to consider fallback behaviour

6.2.3 Intent Transition Rules
From	To	Condition
Any	SEARCH	Always valid
Any	EXPLORE	Always valid
EXPLORE or SEARCH	REFINE	Only valid if at least one recommendation has been shown
REFINE	REFINE	Always valid (chained refinements)
Any	CHITCHAT	Always valid; system acknowledges and re-anchors
Any	UNKNOWN	Always valid; system asks for clarification
6.3 Policy Module

The Policy Module is a fully deterministic component that maps (current_intent, dialogue_state) → action. It contains no LLM calls.

text

POLICY DECISION LOGIC:

IF category is None:
    → ASK_CLARIFICATION (target: category)

ELSE IF current_intent == UNKNOWN:
    → HANDLE_ERROR (type: unknown_intent)

ELSE IF current_intent == CHITCHAT:
    → HANDLE_ERROR (type: chitchat)

ELSE IF current_intent == SEARCH:
    → EXTRACT_PREFERENCES
    → RECOMMEND_ITEMS (after extraction)

ELSE IF current_intent == EXPLORE:
    known_count = count of non-null fields in preferences
    IF known_count < minimum_required[category]:
        → ASK_CLARIFICATION
    ELSE:
        → RECOMMEND_ITEMS

ELSE IF current_intent == REFINE:
    IF no recommendations have been shown yet:
        → ASK_CLARIFICATION (cannot refine before first recommendation)
    ELSE:
        → EXTRACT_PREFERENCES (update constraints)
        → UPDATE_WEIGHTS (dynamic weight adjustment)
        → RECOMMEND_ITEMS (re-query)

IF user message contains "what do you know" or "summarize":
    → SUMMARIZE (overrides other actions)

Minimum preferences before first recommendation:
Category	Minimum Non-Null Fields
Smartphone	2
Laptop	2
Washing Machine	1
6.4 Action Space
Action	Code	LLM Involved	Description
Ask Clarification	ASK_CLARIFICATION	No (question selection) / Yes (phrasing)	Selects next question and phrases it naturally
Extract Preferences	EXTRACT_PREFERENCES	Yes	Extracts and grounds preferences from user message
Recommend Items	RECOMMEND_ITEMS	No (retrieval) / Yes (response phrasing)	Hybrid retrieval + natural language response
Refine Results	REFINE_RESULTS	Yes (extraction) + No (retrieval)	Updates constraints, adjusts weights, re-recommends
Summarize	SUMMARIZE	Yes (light)	Summarises current preferences in natural language
Handle Error	HANDLE_ERROR	No	Returns safe template-based response
6.5 Question Selection Strategy
6.5.1 Priority Lists

Smartphone:

    price_max
    os
    primary_use (inferred from conversation)
    min_storage_gb
    min_camera_mp
    min_battery_mah
    five_g_required
    max_screen_size_inch
    brand

Laptop:

    price_max
    primary_use
    os
    min_ram_gb
    gpu_required
    max_weight_kg
    min_battery_hours
    min_storage_gb
    brand

Washing Machine:

    price_max
    min_capacity_kg
    energy_rating_min
    max_noise_db
    loading_type
    steam_required
    brand

6.5.2 Question Selection Process

text

Step 1: Load priority list for active category
Step 2: Remove attributes in asked_attributes
Step 3: Remove attributes already populated in preferences
Step 4: Select first remaining attribute
Step 5: If all priority attributes known → ask open-ended question
Step 6: Retrieve question template for selected attribute
Step 7: Pass to Response Generator for natural phrasing
Step 8: Add selected attribute to asked_attributes in state update

6.6 Constraint Management
6.6.1 Hard vs Soft Constraints

Hard constraints:

    Stated with mandatory language: "must", "has to", "only", "I need exactly", "I don't want"
    Applied as strict filters; never silently relaxed
    If hard constraints produce zero results, user is informed and asked to reconsider

Soft constraints:

    Stated with preference language: "I'd prefer", "ideally", "around", "something like", "would be nice"
    Applied after hard constraints
    Relaxed in reverse priority order when results are insufficient

6.6.2 Constraint Relaxation Protocol

text

Step 1: Apply all hard constraints → candidate set H

IF |H| == 0:
    → Inform user: no products match hard constraints
    → Identify most restrictive hard constraint
    → Ask user to reconsider
    → STOP

Step 2: Apply all soft constraints to H → candidate set S

IF |S| >= requested_count:
    → Proceed with S

IF |S| < requested_count:
    Step 3: Remove soft constraints one by one 
            (reverse priority order)
            Re-apply remaining constraints after each removal
            UNTIL |S| >= requested_count OR no soft constraints remain

    Step 4: Inform user which constraints were relaxed:
            "I relaxed your preference for [attribute] 
             because too few products matched all criteria."

IF |S| == 0 after all soft removal:
    → Proceed with hard-filtered set H
    → Inform user that preferences could not be fully satisfied

6.6.3 Contradiction Detection and Resolution

text

Step 1: For each newly extracted constraint:
        Check if same attribute already has a value in preferences

Step 2: If contradiction detected:
        a. Update the preference to the new value (newer = more current)
        b. Notify the user: "I've updated your [attribute] 
           from [old] to [new]."
        c. Log the contradiction in LangSmith trace

Step 3: If logically impossible contradiction detected:
        (e.g., price_min = €800 AND price_max = €400)
        a. Do NOT update state
        b. Ask user to clarify: "You've mentioned a minimum of 
           €800 but a maximum of €400. Could you clarify 
           your budget?"

7. Component Specifications
7.1 Intent Classifier Node

Role: Dedicated LLM-powered node for intent classification only.

Inputs: User message (string), last 4 history messages

Processing:

    Construct classification prompt with message, context, intent definitions, and 15 labelled examples
    Invoke GPT-4o-mini
    Parse response into IntentResult (intent, confidence, rationale)
    Validate intent against taxonomy
    Retry up to 3 times on validation failure
    Default to UNKNOWN on persistent failure

Outputs: IntentResult object

Important: The rationale field in the LLM response is logged to LangSmith as a debug trace. It is never used as input to any downstream decision. This addresses the feedback concern about thought trace reliability.

Prompt structure:

    System role definition
    Intent taxonomy with definitions
    3 labelled examples per intent (15 total)
    Current conversation context (last 2 turns)
    User's latest message
    Output format specification (JSON)
    Explicit instruction: "Choose exactly one intent from the provided list"

7.2 Policy Module

Role: Fully deterministic action selector. Maps (intent, state) → action. No LLM calls.

Inputs: IntentResult, current DialogueState

Processing: Applies the decision logic defined in Section 6.3

Outputs: Action (enum), action arguments (dict)

Testing: This component can be fully unit-tested without any LLM calls. 100% branch coverage is achievable and expected.
7.3 Preference Extraction Node

Role: Extracts structured preferences from user utterances, including fuzzy expression grounding.

Inputs: User message, active category, current preferences (for context), category schema

Processing:

Step 1 — Structured Extraction (LLM):
Prompt instructs the LLM to extract only explicitly stated or clearly implied attributes. Returns structured JSON aligned with the category preference schema. Explicit instruction: "Do NOT invent preferences. Return null for attributes not mentioned."

Step 2 — Fuzzy Expression Detection (LLM):
Prompt asks the LLM to identify any relative or vague expressions in the message and return them as a list. Example: ["cheap", "good battery", "not too heavy"].

Step 3 — Fuzzy Grounding (Deterministic):
For each detected fuzzy expression, apply the grounding rules defined in Section 5.3 using the current product database distribution. Grounded values replace or supplement extracted values.

Step 4 — Constraint Classification (LLM):
For each extracted attribute-value pair, classify as HARD or SOFT based on linguistic markers in the source sentence. Returns constraint type and operator.

Step 5 — Validation (Deterministic):
Validate all extracted values against the schema (correct types, plausible ranges). Flag and log invalid extractions without crashing.

Outputs: Validated preference update dict, list of classified Constraint objects, list of fuzzy expressions used
7.4 Semantic Query Builder

Role: Constructs and embeds the semantic query representing the user's aggregated needs.

Inputs: Full DialogueState (all preferences, constraints, fuzzy expressions, history)

Processing:

    Aggregate all known user needs into a natural language string:
        Start from stated preferences
        Add fuzzy expressions verbatim
        Add use-case signals from conversation history
    Embed the aggregated query using text-embedding-3-small
    Return embedding vector

Example aggregation:

text

Known: {os: Android, price_max: 600, min_battery_mah: 4500}
Fuzzy: ["good camera", "lightweight"]
History signals: "takes a lot of photos", "travels frequently"

→ Semantic query: "Android phone under €600 with excellent 
  camera quality and long battery life, lightweight and 
  suitable for travel photography"

Outputs: Query embedding vector (1536-dim float array)
7.5 Hybrid Recommendation Node

Role: Combines structured filtering with semantic retrieval to produce a final ranked product list with explanations.

Inputs: DialogueState (preferences, constraints, weights, shown/rejected products, semantic query embedding), requested count (default 3)

Processing:

Phase 1 — Structured Filtering:
Apply hard constraints as strict filters. Apply soft constraints. Handle zero-result cases with constraint relaxation (Section 6.6.2). Result: filtered_candidates (product IDs).

Phase 2 — Semantic Retrieval:
Query ChromaDB for the active category using the current semantic query embedding. Retrieve top-20 by cosine similarity. Result: semantic_candidates (product IDs + similarity scores).

Phase 3 — Candidate Fusion:
Compute intersection of filtered_candidates and semantic_candidates.

    If intersection has ≥ requested_count products → use intersection
    If intersection is smaller → use intersection + fill from filtered_candidates by structured score
    If intersection is empty → use filtered_candidates only; log that semantic signal was bypassed

Phase 4 — Score Fusion:
For each product in the fused candidate set:

    Compute structured_score using category-specific weighted formula (Section 7.5.1)
    Retrieve semantic_score from ChromaDB result (cosine similarity, normalised 0–1)
    Compute hybrid_score = α × structured_score + (1-α) × semantic_score
    α value from preference_weights in DialogueState

Phase 5 — Exclusion:
Remove products in shown_products and rejected_products from final ranking.

Phase 6 — Explanation Generation (Deterministic):
For each top-K product, generate template-based explanations:

    Budget match reason (if price satisfies constraint)
    Brand match reason (if brand satisfies preference)
    Key spec highlights (2–3 most relevant specs given known preferences)
    Semantic match reason (if semantic score > 0.7): "Strongly matches your description of [fuzzy_expression]"

Outputs: RecommendationResult containing ranked product list, hybrid scores, explanations, filter statistics, relaxed constraints
7.5.1 Category Scoring Formulas

Default weights (adjusted dynamically via preference_weights):

Smartphones:

text

structured_score = 
    0.25 × camera_score +
    0.25 × battery_score +
    0.30 × value_score +
    0.20 × performance_score

Laptops:

text

structured_score = 
    0.35 × performance_score +
    0.25 × battery_score +
    0.25 × value_score +
    0.15 × feature_score

Washing Machines:

text

structured_score = 
    0.35 × energy_score +
    0.25 × noise_score +
    0.25 × value_score +
    0.15 × capacity_score

7.6 State Update Node

Role: Merges all outputs from the current turn into the DialogueState. Fully deterministic.

Inputs: Current DialogueState, extraction result, action executed, response text

Processing:

    Merge new preference values (newer values override older)
    Add new constraints to hard or soft lists
    Run contradiction detection (Section 6.6.3)
    Update asked_attributes with any attribute queried this turn
    Update shown_products with any products displayed this turn
    Update semantic_query with updated aggregated query string
    Update preference_weights with any dynamic adjustments
    Append turn to history (role, message, timestamp, turn_number)
    Increment turn_count
    Update last_action
    Handle category switch (reset preference object, preserve history)

Outputs: Updated DialogueState
7.7 Question Selector

Role: Deterministic selection of the next attribute to ask about.

Inputs: DialogueState (asked_attributes, preferences, category)

Processing: As defined in Section 6.5.2.

Outputs: (attribute_name, question_template)
7.8 Response Generator Node

Role: Converts structured outputs into natural, contextually appropriate language.

Inputs: Action executed, structured output of that action, last 2 turns of history

Processing:

For ASK_CLARIFICATION: Inject question template into natural phrasing. Light LLM call to vary phrasing across turns (prevents robotic repetition). LLM is grounded — it may only rephrase the given question, not add new questions.

For RECOMMEND_ITEMS / REFINE_RESULTS: Assemble structured recommendation response using templates. Include explanations from Recommendation Node. Add constraint relaxation notice if applicable. No LLM involvement in product facts — all numbers come directly from the database.

For SUMMARIZE: LLM call grounded in the current preference dict — summarises what the system knows in natural language. Instruction: "Summarise only what is in the provided preferences dict. Do not add assumptions."

For HANDLE_ERROR: Template-based response, no LLM.

Outputs: System response string, structured product data for UI rendering

Key principle: The Response Generator never invents product facts. All specifications, prices, and feature statements are populated from database records, not LLM generation. The LLM only handles phrasing and connective language.
7.9 Error Handler Node

Role: Provides safe, graceful responses for all failure modes.
Error Type	Detection	Response Strategy
LLM parsing failure (after retries)	AgentDecision/IntentResult validation fails	Template: "I didn't understand that. Could you rephrase?"
Zero results after all relaxation	RecommendationResult is empty	Inform user; identify most restrictive constraint; ask to reconsider
UNKNOWN intent	IntentResult.intent == UNKNOWN	Re-anchor to supported categories
CHITCHAT intent	IntentResult.intent == CHITCHAT	Polite acknowledgement + re-anchor
Category switch	New category detected in message	Confirm switch; reset preference object
Logically impossible constraints	Contradiction detection in State Update	Ask user to clarify
ChromaDB unavailable	Exception in semantic retrieval	Fall back to structured-only recommendation; log warning
LLM API error	HTTP error from OpenAI	Template response; suggest retry; log error
7.10 Session Manager

Operations:
Operation	Description
create_session()	Creates fresh DialogueState; returns session_id
get_state(session_id)	Returns current DialogueState
save_state(session_id, state)	Persists updated state
reset_session(session_id)	Clears all preferences and history; creates fresh state; preserves session_id
end_session(session_id)	Removes from memory

Storage: In-memory dictionary (course prototype). ChromaDB query cache optional for performance.
8. LangGraph Workflow Specification
8.1 Graph State

The LangGraph graph state carries all information needed for one turn of execution:
Field	Type	Description
dialogue_state	DialogueState	Full domain state
user_message	string	Raw user input
intent_result	IntentResult or None	Output of Intent Classifier
selected_action	Action or None	Output of Policy Module
action_arguments	dict	Arguments for selected action
extraction_result	dict or None	Output of Extraction Node
recommendation_result	RecommendationResult or None	Output of Recommendation Node
semantic_query_embedding	list[float] or None	Output of Semantic Query Builder
system_response	string	Final response for display
structured_products	list or None	Product data for UI cards
error_type	string or None	Error classification if failure
debug_traces	list[dict]	Internal reasoning traces (LangSmith only; never shown to user)
8.2 Node Registry
Node	LLM	Deterministic	Description
intent_classifier_node	✅	—	Classifies user intent
policy_module_node	—	✅	Selects action from intent + state
extract_preferences_node	✅	✅ (grounding)	Extracts + grounds preferences
update_state_node	—	✅	Merges all updates into state
semantic_query_builder_node	✅ (embedding)	—	Builds + embeds semantic query
question_selector_node	—	✅	Selects next question attribute
hybrid_recommendation_node	—	✅	Filters + semantic retrieval + scoring
summarize_node	✅	—	Summarises current preferences
response_generator_node	✅ (light)	✅ (templates)	Generates natural language response
error_handler_node	—	✅	Returns safe fallback response
8.3 Edge Definitions
From Node	Condition	To Node
START	Always	intent_classifier_node
intent_classifier_node	Always	policy_module_node
policy_module_node	action == ASK_CLARIFICATION	question_selector_node
policy_module_node	action == EXTRACT_PREFERENCES	extract_preferences_node
policy_module_node	action == RECOMMEND_ITEMS	semantic_query_builder_node
policy_module_node	action == REFINE_RESULTS	extract_preferences_node
policy_module_node	action == SUMMARIZE	summarize_node
policy_module_node	action == HANDLE_ERROR	error_handler_node
extract_preferences_node	Always	update_state_node
update_state_node	last_action was EXTRACT_PREFERENCES (from EXPLORE/SEARCH)	semantic_query_builder_node
update_state_node	last_action was REFINE_RESULTS	semantic_query_builder_node
update_state_node	last_action was ASK_CLARIFICATION	response_generator_node
semantic_query_builder_node	Always	hybrid_recommendation_node
question_selector_node	Always	response_generator_node
hybrid_recommendation_node	Always	update_state_node (log shown products)
update_state_node	after recommendation logged	response_generator_node
summarize_node	Always	response_generator_node
error_handler_node	Always	response_generator_node
response_generator_node	Always	END
8.4 Complete Workflow Diagram

text

         User Message
               │
               ▼
    ┌──────────────────────┐
    │  intent_classifier   │ ← LLM (GPT-4o-mini)
    │       _node          │
    └──────────┬───────────┘
               │ IntentResult
               ▼
    ┌──────────────────────┐
    │   policy_module      │ ← Deterministic Rules
    │       _node          │
    └──────────┬───────────┘
               │ Action + Arguments
               │
    ┌──────────┼─────────────────────────────────────┐
    │          │                                     │
    ▼          ▼                            ▼        ▼
┌───────┐ ┌─────────┐               ┌──────────┐ ┌──────┐
│ques.  │ │extract_ │               │summarize │ │error │
│select.│ │prefs    │               │_node     │ │_hand.│
│_node  │ │_node    │               └────┬─────┘ └──┬───┘
└───┬───┘ └────┬────┘                    │           │
    │          │ Extraction Result        │           │
    │          ▼                          │           │
    │   ┌─────────────┐                  │           │
    │   │ update_state│                  │           │
    │   │    _node    │                  │           │
    │   └──────┬──────┘                  │           │
    │          │ Updated State            │           │
    │          ▼                          │           │
    │   ┌─────────────────┐              │           │
    │   │ semantic_query  │              │           │
    │   │  _builder_node  │ ← Embedding  │           │
    │   └──────┬──────────┘              │           │
    │          │ Query Vector             │           │
    │          ▼                          │           │
    │   ┌─────────────────┐              │           │
    │   │    hybrid_      │              │           │
    │   │ recommendation  │              │           │
    │   │     _node       │              │           │
    │   └──────┬──────────┘              │           │
    │          │                          │           │
    │          ▼                          │           │
    │   ┌─────────────┐                  │           │
    │   │ update_state│                  │           │
    │   │ (log shown) │                  │           │
    │   └──────┬──────┘                  │           │
    │          │                          │           │
    └──────────┼──────────────────────────┘           │
               │                                      │
               └──────────────────────────────────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │  response_generator   │ ← LLM (light)
                  │        _node          │
                  └───────────┬───────────┘
                              │
                             END
                              │
                              ▼
                       System Response
                    + Product Cards (UI)

9. Evaluation Framework
9.1 Evaluation Strategy

The system is evaluated across four dimensions:

    Component accuracy — intent classification, preference extraction
    Semantic layer quality — retrieval relevance
    End-to-end recommendation quality — hit rate, constraint satisfaction
    Dialogue behaviour — efficiency, error handling, constraint relaxation

9.2 Test Dataset

Total: 30 scripted test dialogues
Category	Normal Cases	Edge Cases	Total
Smartphone	5	5	10
Laptop	4	4	8
Washing Machine	4	4	8
Cross-category (switch)	—	4	4
Total	13	17	30

Edge case types covered:

    Zero-result queries requiring constraint relaxation (4 dialogues)
    Contradictory preferences across turns (4 dialogues)
    Fuzzy-only preferences with no numeric constraints (4 dialogues)
    Category switch mid-session (4 dialogues)
    Chitchat and unknown intent handling (4 dialogues)
    Chained refinements (3+ REFINE turns in sequence) (1 dialogue)

Each test dialogue contains:

    User utterances for all turns
    Ground-truth intent per turn
    Ground-truth preference state after each turn (including constraint type: HARD/SOFT)
    Expected system action per turn
    Expected question attribute (for ASK_CLARIFICATION turns)
    Ground-truth top-3 recommended products (human expert annotation)
    Whether any constraint relaxation should occur

9.3 Metrics
9.3.1 Intent Classification
Metric	Formula	Target
Overall accuracy	Correct / Total	≥ 80%
Per-class precision	TP / (TP + FP)	Reported per class
Per-class recall	TP / (TP + FN)	Reported per class
Per-class F1	2PR / (P + R)	Reported per class

Output: 5×5 confusion matrix across all intent classes.
9.3.2 Preference Extraction
Metric	Formula	Target
Slot accuracy	Correct slots / Expected slots	≥ 75%
Slot precision	Correct slots / Extracted slots	Reported
Missing slot rate	Missing / Expected	Reported
Hallucination rate	Invented / Extracted	≤ 10%
Fuzzy grounding accuracy	Correctly grounded / Total fuzzy	≥ 70%
Constraint type accuracy	Correct HARD/SOFT / Total	≥ 75%
9.3.3 Semantic Layer Quality
Metric	Description	Target
Semantic retrieval relevance	% of semantic candidates judged relevant by a human evaluator (2 team members independently)	≥ 70%
Semantic re-ranking improvement	Does hybrid ranking outperform structured-only on ground-truth hit rate?	Report delta

Semantic relevance is judged by two team members independently on a binary scale (relevant / not relevant). Inter-rater agreement is reported as Cohen's κ.
9.3.4 Recommendation Quality
Metric	Formula	Target
Hit Rate @3	% of dialogues with ≥1 ground-truth product in top 3	≥ 70%
Precision @3	Ground-truth products in top 3 / 3	Reported
Hard constraint satisfaction	% of recommendations satisfying all hard constraints	100%
Constraint relaxation precision	% of relaxation cases handled correctly	≥ 90%
9.3.5 Dialogue Behaviour
Metric	Description	Target
Avg turns to first recommendation	Mean across EXPLORE dialogues	≤ 4 turns
Redundant question rate	Questions about already-known attributes	0%
Safe recovery rate	% of error cases returning valid, non-crashing response	100%
Contradiction resolution rate	% of contradictions correctly detected and resolved	≥ 80%
9.4 Ablation Study

To demonstrate the value of the semantic layer, the following ablation is conducted:
System Variant	Description
Full System	Hybrid: structured filtering + semantic re-ranking
Ablation A	Structured-only: no semantic layer
Ablation B	Semantic-only: no structured filtering

All three variants are evaluated on the same 30 test dialogues. Hit Rate @3 and Precision @3 are compared across variants. This provides quantitative evidence that the semantic layer adds measurable value.
9.5 Evaluation Procedure

text

For each test dialogue:
    Step 1: Feed user utterances to system one at a time
    Step 2: After each turn, record:
            - Detected intent (vs ground truth)
            - Extracted preferences (vs ground truth)
            - Selected action (vs ground truth)
            - Semantic candidates retrieved
    Step 3: After final turn, record:
            - Recommended product IDs (vs ground truth)
            - Constraint satisfaction status
            - Relaxation events (if any)
    
After all dialogues:
    Step 4: Compute all metrics in Section 9.3
    Step 5: Run ablation study (Section 9.4)
    Step 6: Produce confusion matrix for intent classification
    Step 7: Compile results into evaluation report

10. Implementation Plan
10.1 Team Roles
Member	Primary Responsibility	Secondary Responsibility
Member 1	Data Layer + Semantic Layer + Recommendation Engine	Ablation Study
Member 2	Pydantic Models + Dialogue State + Session Manager + Preference Extraction	Contradiction Detection
Member 3	LangGraph Workflow + Intent Classifier + Policy Module + Error Handler	LangSmith Integration
Member 4	Streamlit UI + Response Generator + Question Selector	Test Dialogue Creation + Evaluation Script
10.2 Project Phases
Phase 1 — Foundation (Days 1–5)

Goal: All data structures defined and agreed upon. Product database ready. Skeleton compiles.
Task	Owner	Deliverable
Define all Pydantic models	Member 2	models.py — all models instantiate
Collect product data (50 per category)	Member 1	Raw CSV files
Write product descriptions (50 per category)	Member 1	Descriptions in CSV
Compute pre-normalised score columns	Member 1	Final CSVs with scores
Implement Session Manager	Member 2	session_manager.py
LangGraph skeleton (all nodes as stubs)	Member 3	workflow.py — compiles
Streamlit skeleton (chat interface)	Member 4	app.py — accepts input
LangSmith project setup	Member 3	Environment configured
requirements.txt finalised	Member 3	All versions pinned

Phase 1 Exit Criteria:

    All Pydantic models instantiate without errors
    All CSVs load correctly with all required fields
    LangGraph skeleton compiles and routes dummy input
    Streamlit app runs

Phase 2 — Core Logic (Days 6–11)

Goal: End-to-end EXPLORE dialogue works for all categories.
Task	Owner	Deliverable
Intent Classifier Node (LLM + validation + retry)	Member 3	intent_classifier.py
Policy Module (fully deterministic)	Member 3	policy_module.py
Preference Extraction Node (structured extraction)	Member 2	extraction_node.py
Fuzzy Expression Grounding (deterministic)	Member 2	Part of extraction_node.py
State Update Node	Member 2	state_update_node.py
Structured Recommendation Engine (filtering + scoring)	Member 1	recommendation_engine.py
Question Selector	Member 4	question_selector.py
Response Generator (template-based)	Member 4	response_generator.py
Connect all nodes in LangGraph	Member 3	workflow.py — full connections
Connect Streamlit to workflow	Member 4	app.py — end-to-end

Phase 2 Exit Criteria:

    Full EXPLORE dialogue works for smartphones, laptops, washing machines
    LangSmith shows correct node traces
    Recommendations displayed with explanations

Phase 3 — Semantic Layer + Advanced Features (Days 12–16)

Goal: Semantic layer active. All use cases handled. Edge cases covered.
Task	Owner	Deliverable
Generate and store product embeddings	Member 1	ChromaDB collections built
Semantic Query Builder Node	Member 1	semantic_query_builder.py
Hybrid Recommendation Node (semantic + structured fusion)	Member 1	Updated recommendation_engine.py
Dynamic weight adjustment	Member 1	Weight update logic in State Update
REFINE intent handling (update + re-recommend)	Member 3	REFINE flow working
Constraint relaxation logic	Member 1	Zero-result cases handled
Contradiction detection	Member 2	Contradictions resolved correctly
Category switch handling	Member 2	Category switch resets correctly
Error Handler Node (all error types)	Member 3	All errors produce safe responses
Quick-action buttons in UI	Member 4	Buttons functional
Product card rendering in UI	Member 4	Cards display with specs
Write 30 test dialogues with ground truth	All	test_dialogues.json

Phase 3 Exit Criteria:

    All 6 use cases work end-to-end
    Semantic retrieval returns relevant results
    Zero-result queries handled without crashing
    Category switches handled correctly
    All 30 test dialogues written

Phase 4 — Evaluation + Polish (Days 17–20)

Goal: System evaluated, documented, demo-ready.
Task	Owner	Deliverable
Evaluation script	Member 4	evaluate.py
Run evaluation on 30 test dialogues	Member 4	Results tables
Ablation study execution	Member 1	Ablation results
Confusion matrix production	Member 4	Confusion matrix figure
Semantic relevance human evaluation	Members 1+2	Relevance judgements + κ
LangSmith trace review + agent prompt tuning	Member 3	Improved prompts
UI final polish + demo preparation	Member 4	Demo-ready app
Final report writing	All	Report document
Demo script preparation	All	Demo script

Phase 4 Exit Criteria:

    Intent accuracy ≥ 80%
    Hit Rate @3 ≥ 70%
    Ablation shows semantic layer improves Hit Rate @3
    All error cases produce safe responses
    Demo runs without interruption for 10+ minutes

10.3 Timeline

text

Day:  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19  20
      ├────────────────┤──────────────────────────┤──────────────────┤────────┤
             Phase 1         Phase 2                    Phase 3        Phase 4
           Foundation       Core Logic           Semantic + Advanced   Polish

10.4 Repository Structure:

/crs-project
│
├── /data
│   ├── smartphones.csv            # 50 products with descriptions + scores
│   ├── laptops.csv                # 50 products with descriptions + scores
│   ├── washing_machines.csv       # 45 products with descriptions + scores
│   └── /embeddings                # ChromaDB persistent storage
│       ├── /smartphones
│       ├── /laptops
│       └── /washing_machines
│
├── /src
│   ├── models.py                  # All Pydantic models
│   ├── session_manager.py         # Session lifecycle
│   ├── workflow.py                # LangGraph graph definition
│   ├── intent_classifier.py       # Intent Classifier Node
│   ├── policy_module.py           # Policy Module (deterministic)
│   ├── extraction_node.py         # Preference Extraction + Fuzzy Grounding
│   ├── state_update_node.py       # State Update Node
│   ├── semantic_query_builder.py  # Semantic Query Builder Node
│   ├── recommendation_engine.py   # Hybrid Recommendation Node
│   ├── question_selector.py       # Question Selector
│   ├── response_generator.py      # Response Generator Node
│   ├── error_handler.py           # Error Handler Node
│   └── config.py                  # Model names, thresholds, weights
│
├── /scripts
│   └── build_embeddings.py        # One-time embedding generation script
│
├── /evaluation
│   ├── test_dialogues.json        # 30 annotated test dialogues
│   ├── evaluate.py                # Full evaluation script
│   └── results/                   # Evaluation output files
│
├── /tests
│   ├── test_models.py
│   ├── test_policy_module.py      # Full unit test coverage (no LLM)
│   ├── test_recommendation.py
│   ├── test_extraction.py
│   └── test_fuzzy_grounding.py
│
├── app.py                         # Streamlit entry point
├── requirements.txt               # Pinned versions
├── .env.example                   # API key template
└── README.md

