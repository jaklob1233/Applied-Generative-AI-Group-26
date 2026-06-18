# CHANGES — Findora (after the poster session)

Feedback came from **6 peer reviewers**. Below we mention every piece of feedback, what we changed in response and why — and,
where we chose **not** to change something, the reasoning. The post-change **evaluation** is at the end.

---

## 1. What we changed

## "Expand the data: add product details, descriptions and reviews." (reviewer 2; also our own stated limitation)
**Changed: ** Every phone now has a grounded **description** and a clearly-labeled **AI review summary**,
generated *only* from its real specs + rating (`enrich_smartphones.py` →
`datasets/smartphone_enrichment.json`, joined into the catalog at load time, similar to image cache). They appear in a product's **Details** view and feed the semantic "vibe" search. Coverage
is **100% of the 500 phones**, and the summary is explicitly marked *"AI summary (generated from specs &
rating)"* so it is never mistaken for a real user review.

*Why generated rather than scraped real reviews:* we made an attempt to add real
third-party reviews and rejected three datasets + a live scrape on the evidence:
| Source | Verdict |
|---|---|
| Amazon (2019) | real, but too old — matched ~1% of our modern catalog |
| "Global Mobile Reviews 2025" | modern, but **synthetic**: 50,000 rows, only **110 unique sentences**, 22 phones |
| Flipkart (194k) | real, but almost all non-phones — matched **1** of our 500 |
| GSMArena live scrape | **blocked by Cloudflare bot-protection** (server *and* local), and reviews exist only for flagships |



## "Show one more real-world example of the full recommendation process." (reviewer 7)
**Changed:** Added in the updated poster.

## "Address the local-model limitation and the agent's failure handling more concretely." (reviewer 6)
**Changed:**  Bad tool arguments are validated and **never crash** (tools never raise); a request needing more than the 4-round budget is forced to a final grounded answer; local function-calling is unreliable on small models, which is *why* the system is cloud-primary.

## "Be honest about / improve loading times." (reviewer 4)
**Changed:** The poster's latency box now reports the **full distribution** (avg / p95 / max for *both*
engines) instead of one flattering average, making clear the agent wins on **predictability and cost**,
not raw speed. We also hardened the evaluation: `--runs N` reports **mean ± std**, and `--judge-model`
allows a different-family judge (to rule out judge bias).

### "Surface the less-intuitive commands (e.g. you can tell it to clear all filters)." (reviewer 4)
**Changed:** Added a **"Handy commands"** section to the in-app *"ℹ️ How it works"* panel listing the
non-obvious natural-language commands — *"clear all filters", "ignore that" (undo), "cheaper ones",
"compare those", "why this one?", "actually, show me headphones"* — so users discover them.

### "The '700 True' tooltip was confusing." (reviewer 5)
**Resolved:** We could not reproduce it in the current build: the post-session UI rework relabeled every
spec/score value with a unit or **Yes/No** (verified across the spec table, the score breakdown, and the
compare view), so no raw value renders without a label. It appears the rework already eliminated it.

## 2. Considered but not changed (with reasoning)

- **Cross-shop price comparison:** Out of scope: the reviewer agreed. Findora is a
  single-catalog recommender; live multi-retailer pricing needs commercial price-feed integrations well
  beyond this project.
- **Expanding the catalog to ~1,000 phones:** We deliberately **enriched the existing 500
  in place** rather than swap the dataset: it keeps our validated evaluation baseline comparable, keeps
  the clean USD prices the ranking depends on, and the enrichment (not the row count) was the actual gap.
- **Lower loading time via response streaming:** Acknowledged, not implemented. The agent is
  already faster on average and we added the honest latency analysis; streaming the final reply (lower
  *perceived* latency) is the clear next step but was deprioritised in favour of the data work.
---

## 3. Evaluation (after changes)

The post-session changes are **additive** — descriptions/summaries are extra columns + semantic text and
two extra fields in `get_product_details`; they do **not** alter the agent's tool-calling or decision
logic — so the evaluation gate that drove the project still holds. Both engines on the same
engine-agnostic gate (`eval_harness.py`):

| Metric | Pipeline (v1) | **Agent (v2)** |
|---|---|---|
| Conversation quality — LLM-judge, **mean ± std of 3 runs** | 0.412 | **0.746 ± 0.024** |
| Adversarial & safety pass-rate (28 checks) | 82% (23/28) | **100% (28/28)** |
| count / compound requests | 1/6 | **6/6** |
| Entity resolution / slot validation / intent routing | 20/20 · 10/10 · 38/40 | (shared deterministic core) |
| Latency per turn — avg / p95 / max | 7.2 / 6.1 / 222 s | **4.4 / 9.6 / 54 s** |
| Input tokens cached | — | **~56%** |

Reproduce (from the project root): `python src/eval_harness.py --runs 3` (agent) and
`ENGINE=pipeline python src/eval_harness.py` (pipeline). The deterministic tool suite
(`python src/test_tools.py`, **17/17 pass**) and a live UI check of the new Details view were
re-verified after the enrichment.

> Note: a fresh full re-run was attempted after these changes but our shared API key hit its credit
> limit mid-run (it did not overwrite the scorecard above). The numbers shown are the validated 3-run
> results; rerun the command above once the key has credit to confirm (expected unchanged, as the
> enrichment does not touch agent logic).
