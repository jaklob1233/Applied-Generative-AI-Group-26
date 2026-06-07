"""
make_poster.py
Generates poster.pdf — three vertically-stacked A3-landscape pages
(page 1 = top, 2 = middle, 3 = bottom), as required for submission.

  python make_poster.py

Pulls quantitative metrics from eval_results.json (run evaluate.py first).
Edit the CONFIG block below to set group members / title.
"""

import json
import os

from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.colors import HexColor, white, Color
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

# ── CONFIG — EDIT THESE ──────────────────────────────────────────────────────
TITLE = "Conversational Recommender System"
SUBTITLE = ("A multi-turn LLM shop assistant for smartphones & headphones — validated NLU, "
            "hybrid semantic retrieval, learned ranking & full observability")
GROUP = "Group 26"
COURSE = "Applied Generative AI · SS 2026"
MEMBERS = ["Arthur Galois", "Jakob Meltzer", "Sofie Parkesaas", "Muhammad Azeem Shahzad"]
# ─────────────────────────────────────────────────────────────────────────────

PAGE = landscape(A3)            # (1190.55, 841.89) pts
PW, PH = PAGE

INDIGO = HexColor(0x6366F1)
VIOLET = HexColor(0x8B5CF6)
PINK   = HexColor(0xEC4899)
INK    = HexColor(0x1E293B)
MUTED  = HexColor(0x64748B)
LINE   = HexColor(0xE2E6F0)
SOFT   = HexColor(0xF5F6FC)
LAV    = HexColor(0xEEF2FF)
GREEN  = HexColor(0x16A34A)


def load_metrics():
    if os.path.exists("eval_results.json"):
        with open("eval_results.json", encoding="utf-8") as f:
            return json.load(f)
    return None


# ── Low-level drawing helpers (top-left coordinate system) ───────────────────

def Y(top):
    """Convert a top-down y to reportlab's bottom-up y."""
    return PH - top


def gradient_band(c, x, top, w, h, c1, c2, steps=120):
    strip = w / steps
    for i in range(steps):
        t = i / (steps - 1)
        col = Color(c1.red + (c2.red - c1.red) * t,
                    c1.green + (c2.green - c1.green) * t,
                    c1.blue + (c2.blue - c1.blue) * t)
        c.setFillColor(col)
        c.rect(x + i * strip, Y(top + h), strip + 1, h, stroke=0, fill=1)


def panel(c, x, top, w, h, title, accent=VIOLET, fill=white):
    c.setFillColor(fill)
    c.setStrokeColor(LINE)
    c.setLineWidth(1)
    c.roundRect(x, Y(top + h), w, h, 10, stroke=1, fill=1)
    # accent bar
    c.setFillColor(accent)
    c.roundRect(x, Y(top + h), 6, h, 3, stroke=0, fill=1)
    c.rect(x + 3, Y(top + h), 4, h, stroke=0, fill=1)
    if title:
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 15)
        c.drawString(x + 18, Y(top + 22), title)
    return top + 34  # y where body content can start


def wrapped(c, text, x, top, w, font="Helvetica", size=10.5, leading=14,
            color=INK, bold_lead=None):
    c.setFillColor(color)
    c.setFont(font, size)
    lines = simpleSplit(text, font, size, w)
    for ln in lines:
        c.drawString(x, Y(top), ln)
        top += leading
    return top


def bullets(c, items, x, top, w, size=10.5, leading=14, gap=4, color=INK):
    for it in items:
        c.setFillColor(VIOLET)
        c.setFont("Helvetica-Bold", size)
        c.drawString(x, Y(top), "•")
        sub = simpleSplit(it, "Helvetica", size, w - 14)
        c.setFillColor(color)
        c.setFont("Helvetica", size)
        for j, ln in enumerate(sub):
            c.drawString(x + 14, Y(top), ln)
            top += leading
        top += gap
    return top


def chip(c, x, top, label, fill=LAV, fg=HexColor(0x4338CA)):
    w = c.stringWidth(label, "Helvetica-Bold", 9) + 16
    c.setFillColor(fill)
    c.roundRect(x, Y(top + 16), w, 16, 8, stroke=0, fill=1)
    c.setFillColor(fg)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x + 8, Y(top + 11.5), label)
    return x + w + 6


# ── PAGE 1 — Title · Group · Abstract · Problem ──────────────────────────────

def page1(c, m):
    # Hero banner
    gradient_band(c, 0, 0, PW, 150, INDIGO, PINK)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 40)
    c.drawString(40, Y(64), TITLE)
    c.setFont("Helvetica", 15)
    for i, ln in enumerate(simpleSplit(SUBTITLE, "Helvetica", 15, PW - 360)):
        c.drawString(40, Y(92 + i * 20), ln)
    # group block (right)
    c.setFont("Helvetica-Bold", 20)
    c.drawRightString(PW - 40, Y(56), GROUP)
    c.setFont("Helvetica", 12)
    c.drawRightString(PW - 40, Y(78), COURSE)
    c.drawRightString(PW - 40, Y(98), " · ".join(MEMBERS))

    top = 175
    colw = (PW - 80 - 30) / 2

    # Abstract (left)
    by = panel(c, 40, top, colw, 250, "Abstract", INDIGO)
    abstract = (
        "We build a conversational recommender that acts as a shop assistant for tech products. "
        "Through multi-turn natural-language dialogue it elicits user preferences, reasons over an "
        "explicit dialogue state, and recommends concrete products. NLU is split into a routing + "
        "slot-filling pipeline guarded by schema validation and an entity-resolution layer; vague "
        "language (“cheap”, “good camera”) is grounded in dataset percentiles; retrieval blends "
        "structured pandas filters with dense-embedding semantic search for use-case (“vibe”) queries; "
        "products are ranked by TOPSIS, upgradable to a learned ranker via a click-feedback flywheel. "
        "A LangGraph state machine orchestrates four nodes; every turn is traced (LangSmith) and logged "
        "for analytics, and recommendations show real product photos."
    )
    wrapped(c, abstract, 58, by + 6, colw - 36, leading=15.5, size=11)

    # Problem (right)
    px = 40 + colw + 30
    py = panel(c, px, top, colw, 250, "Initial Situation & Problem", PINK)
    py = wrapped(c, "Traditional recommenders work silently (ranked lists, “people also bought”). "
                    "Users often start vague and refine as they talk. A conversational recommender (CRS) "
                    "must instead elicit preferences in real time and adapt.",
                 px + 18, py + 4, colw - 36, leading=15, size=11)
    py += 6
    c.setFillColor(INK); c.setFont("Helvetica-Bold", 11)
    c.drawString(px + 18, Y(py), "Core challenges / questions:"); py += 18
    bullets(c, [
        "Robust intent detection & preference elicitation from messy natural language.",
        "Reasoning over a dialogue state to pick the next best action — not blindly answering.",
        "Safely translating conversational critiques (“cheaper but better camera”) into DB queries.",
        "What information does the system actually need to reliably recommend a set of items?",
    ], px + 18, py, colw - 36, size=10.5, leading=13.5)

    # Sample use cases strip (full width)
    uy = top + 268
    by = panel(c, 40, uy, PW - 80, 300, "Sample Use Cases", VIOLET)
    cards = [
        ("1 · Specific lookup", "“A Samsung Android phone with 8GB RAM”",
         "Pre-fills brand+OS+spec, recommends immediately — ranked, no needless questions."),
        ("2 · Guided exploration", "“I want a smartphone”",
         "Asks a focused sequence: OS → price → battery → camera → RAM, with ‘any/skip’."),
        ("3 · Refinement / critique", "“cheaper ones” · “bigger battery”",
         "Anchors relative terms to the last results; refines are additive; ‘forget X’ removes."),
        ("4 · Vague & vibe language", "“cheap phone for gaming”",
         "Maps ‘cheap’→p25 price; ‘gaming’ via semantic embeddings over spec descriptions."),
    ]
    cw = (PW - 80 - 36 - 30) / 4
    for i, (h, ex, d) in enumerate(cards):
        cx = 58 + i * (cw + 10)
        c.setFillColor(SOFT); c.setStrokeColor(LINE)
        c.roundRect(cx, Y(by + 232), cw, 232, 8, stroke=1, fill=1)
        c.setFillColor(HexColor(0x4338CA)); c.setFont("Helvetica-Bold", 11)
        c.drawString(cx + 12, Y(by + 22), h)
        c.setFillColor(VIOLET); c.setFont("Helvetica-Oblique", 10)
        ey = by + 40
        for ln in simpleSplit(ex, "Helvetica-Oblique", 10, cw - 24):
            c.drawString(cx + 12, Y(ey), ln); ey += 13
        c.setFillColor(INK); c.setFont("Helvetica", 10)
        ey += 4
        for ln in simpleSplit(d, "Helvetica", 10, cw - 24):
            c.drawString(cx + 12, Y(ey), ln); ey += 13

    _footer(c, 1)
    c.showPage()


# ── PAGE 2 — Approach / Architecture (visual centrepiece) ────────────────────

def page2(c, m):
    c.setFillColor(INK); c.setFont("Helvetica-Bold", 26)
    c.drawString(40, Y(48), "Approach — System Architecture")
    c.setFillColor(MUTED); c.setFont("Helvetica", 13)
    c.drawString(40, Y(70), "LangGraph 4-node pipeline · de-risked NLU · hybrid retrieval · learned ranking · observability")

    # ── Pipeline (the centrepiece) ──
    pipe = ["User\nmessage", "ROUTER\nintent · conf\nsort · category",
            "SLOT\nEXTRACTOR\nfilters", "VALIDATE\n+ RESOLVE\ndrop bad slots",
            "STATE\nUPDATER\nmerge · switch\nundo", "RETRIEVE\n& ACT\nfilter → rank",
            "RESPONSE\nLLM reply"]
    n = len(pipe)
    bw, bh, gap = 138, 96, 22
    total = n * bw + (n - 1) * gap
    x0 = (PW - total) / 2
    ytop = 100
    fills = [SOFT, LAV, LAV, HexColor(0xFCE7F3), LAV, HexColor(0xDCFCE7), LAV]
    for i, label in enumerate(pipe):
        x = x0 + i * (bw + gap)
        c.setFillColor(fills[i]); c.setStrokeColor(VIOLET if i in (1, 2) else LINE)
        c.setLineWidth(1.5 if i in (1, 2) else 1)
        c.roundRect(x, Y(ytop + bh), bw, bh, 9, stroke=1, fill=1)
        lines = label.split("\n")
        c.setFillColor(INK); c.setFont("Helvetica-Bold", 11.5)
        c.drawCentredString(x + bw / 2, Y(ytop + 22), lines[0])
        c.setFillColor(MUTED); c.setFont("Helvetica", 9)
        for j, sub in enumerate(lines[1:]):
            c.drawCentredString(x + bw / 2, Y(ytop + 40 + j * 12), sub)
        if i < n - 1:
            ay = ytop + bh / 2
            c.setStrokeColor(VIOLET); c.setLineWidth(2)
            c.line(x + bw, Y(ay), x + bw + gap, Y(ay))
            c.setFillColor(VIOLET)
            p = c.beginPath(); p.moveTo(x + bw + gap, Y(ay))
            p.lineTo(x + bw + gap - 6, Y(ay - 4)); p.lineTo(x + bw + gap - 6, Y(ay + 4))
            p.close(); c.drawPath(p, fill=1, stroke=0)

    # ── Cross-cutting platform rails ──
    rails = [
        ("MEMORY", "per-user profiles · ‘use my usual’ · personalised greeting", INDIGO),
        ("RANKING", "TOPSIS (MCDM) ↔ learned LTR · semantic blend · feedback flywheel", PINK),
        ("OBSERVABILITY", "LangSmith tracing · per-turn JSONL · intent & funnel analytics", VIOLET),
        ("IMAGES", "offline enrichment → GSMArena/Wikipedia → image_url → cards", INDIGO),
    ]
    ry = 230
    rw = (PW - 80 - 3 * 14) / 4
    for i, (h, d, col) in enumerate(rails):
        rx = 40 + i * (rw + 14)
        c.setFillColor(white); c.setStrokeColor(col); c.setLineWidth(1.4)
        c.roundRect(rx, Y(ry + 66), rw, 66, 8, stroke=1, fill=1)
        c.setFillColor(col); c.setFont("Helvetica-Bold", 12)
        c.drawString(rx + 12, Y(ry + 22), h)
        c.setFillColor(INK); c.setFont("Helvetica", 9.5)
        ey = ry + 38
        for ln in simpleSplit(d, "Helvetica", 9.5, rw - 24):
            c.drawString(rx + 12, Y(ey), ln); ey += 12

    # ── Key design decisions (left) + Models & Tools (right) ──
    dtop = 320
    colw = (PW - 80 - 30) / 2
    dy = panel(c, 40, dtop, colw, 300, "Key Design Decisions", INDIGO)
    bullets(c, [
        "Split NLU into a small router + a slot extractor instead of one fragile mega-prompt.",
        "Validate every LLM-proposed slot against a schema; drop unknown / out-of-range / wrong-type.",
        "Entity resolution as a real layer (alias dictionary + fuzzy match), not ever-growing prompt rules.",
        "Explicit DialogueState (TypedDict) + LangGraph; information-gathering gate decides ask vs. recommend.",
        "Confidence-gated clarification; out-of-scope intent answers honestly instead of misfiring.",
        "Hybrid retrieval: exact pandas filters + dense embeddings (model2vec) for ‘vibe’ queries.",
        "Transparent TOPSIS ranking; pluggable learned ranker trained from click feedback.",
    ], 58, dy + 4, colw - 36, size=10.5, leading=13.5, gap=3)

    tx = 40 + colw + 30
    ty = panel(c, tx, dtop, colw, 300, "Models & Tools", PINK)
    ty += 2
    rows = [
        ("Orchestration", "LangGraph state machine (4 nodes) + LangChain"),
        ("LLM", "OpenRouter · GPT-4o-mini (router, extractor, responder)"),
        ("Embeddings", "model2vec · potion-base-8M (static, no GPU)"),
        ("Data / retrieval", "pandas + numpy · 2×500 product catalog"),
        ("Ranking", "TOPSIS (Hwang & Yoon) · numpy logistic LTR"),
        ("UI", "Streamlit chat · comparison tables · product cards"),
        ("Observability", "LangSmith · structured JSONL logs"),
        ("Images", "GSMArena / Wikipedia / Openverse enrichment"),
    ]
    for k, v in rows:
        c.setFillColor(HexColor(0xBE185D)); c.setFont("Helvetica-Bold", 10.5)
        c.drawString(tx + 18, Y(ty), k)
        c.setFillColor(INK); c.setFont("Helvetica", 10.5)
        c.drawString(tx + 150, Y(ty), v)
        ty += 20

    # ── Conversational capabilities (fills the lower band) ──
    cap_top = 638
    panel(c, 40, cap_top, PW - 80, 165, "Conversational Capabilities — intents → system actions", VIOLET)
    caps = [
        ("explore", "category only → guided questions (OS→price→battery→camera→RAM)"),
        ("specific", "concrete criteria → recommend or one narrowing question"),
        ("refine", "react to results: ‘cheaper’ anchored to last set; additive merges"),
        ("summarize", "recap understood preferences in plain language"),
        ("done", "wrap-up / pick one / restart → clean close + new-search"),
        ("chitchat", "greetings/thanks → warm reply, product state preserved"),
        ("ambiguous", "too vague → ask to clarify (confidence-gated)"),
        ("out_of_scope", "photos/buy/stock/warranty → honest ‘can’t, but I can…’"),
    ]
    gw = (PW - 80 - 36 - 3 * 12) / 4
    for i, (k, v) in enumerate(caps):
        col, row = i % 4, i // 4
        gx = 58 + col * (gw + 12)
        gy = cap_top + 42 + row * 58
        c.setFillColor(SOFT); c.setStrokeColor(LINE)
        c.roundRect(gx, Y(gy + 48), gw, 48, 7, stroke=1, fill=1)
        c.setFillColor(HexColor(0x4338CA)); c.setFont("Helvetica-Bold", 10)
        c.drawString(gx + 10, Y(gy + 16), k)
        c.setFillColor(INK); c.setFont("Helvetica", 8.6)
        ey = gy + 28
        for ln in simpleSplit(v, "Helvetica", 8.6, gw - 20)[:2]:
            c.drawString(gx + 10, Y(ey), ln); ey += 10.5
    # extra-capability chips
    cy = cap_top + 160
    c.setFillColor(MUTED); c.setFont("Helvetica-Oblique", 9)
    c.drawString(58, Y(cap_top + 158),
                 "Cross-cutting: skip (‘any’) · undo (‘ignore that’) · multi-intent · category-switch · "
                 "vague-term grounding (percentiles) · persistent memory · quick-reply buttons.")

    _footer(c, 2)
    c.showPage()


# ── PAGE 3 — Results / Demo · Learnings & Limitations ────────────────────────

def page3(c, m):
    c.setFillColor(INK); c.setFont("Helvetica-Bold", 26)
    c.drawString(40, Y(48), "Results, Demo & Learnings")

    # Metrics row
    suites = (m or {}).get("suites", {})
    def pct(name, default="—"):
        s = suites.get(name)
        return f"{s['pct']}%" if s else default
    lat = (m or {}).get("avg_latency_s_per_turn")
    img = (m or {}).get("image_coverage_pct", {})

    metrics = [
        (pct("Intent routing"), "Intent routing\naccuracy"),
        (pct("Entity resolution"), "Entity resolution\naccuracy"),
        (pct("Slot validation"), "Slot validation\n(bad slots caught)"),
        (pct("End-to-end task success"), "End-to-end\ntask success"),
        (f"{img.get('smartphone','—')}%", "Phone image\ncoverage"),
        (f"{lat}s" if lat else "—", "Avg latency\nper turn"),
    ]
    mw = (PW - 80 - 5 * 12) / 6
    mtop = 72
    for i, (val, lab) in enumerate(metrics):
        mx = 40 + i * (mw + 12)
        gradient_band(c, mx, mtop, mw, 90, INDIGO, VIOLET)
        c.setFillColor(white); c.setFont("Helvetica-Bold", 30)
        c.drawCentredString(mx + mw / 2, Y(mtop + 44), str(val))
        c.setFont("Helvetica", 9.5)
        for j, ln in enumerate(lab.split("\n")):
            c.drawCentredString(mx + mw / 2, Y(mtop + 62 + j * 12), ln)
    c.setFillColor(MUTED); c.setFont("Helvetica-Oblique", 9)
    c.drawString(40, Y(mtop + 104), "Metrics from evaluate.py: 20 resolution + 10 validation (deterministic), "
                                    "22 intent + 10 multi-turn scenarios (LLM-graded).")

    # Example conversation (left)
    ctop = 200
    colw = (PW - 80 - 30) / 2
    cy = panel(c, 40, ctop, colw, 300, "Demo — Example Conversation", VIOLET)
    convo = [
        ("u", "a cheap android phone with a good camera"),
        ("a", "Found 60 matches — top pick: Realme 9 Pro 5G. (price≤$147, camera≥64MP applied)"),
        ("u", "show me cheaper ones"),
        ("a", "Narrowed to phones under $147, cheapest first. Want to compare a few?"),
        ("u", "can you show pictures of this?"),
        ("a", "I can't show photos here, but I can share full specs or compare options."),
    ]
    yy = cy + 6
    for who, txt in convo:
        is_u = who == "u"
        bw = colw - 60
        lines = simpleSplit(txt, "Helvetica", 10, bw - 20)
        bh = 12 + len(lines) * 13
        bx = (40 + colw - 18 - bw) if is_u else (58)
        c.setFillColor(LAV if is_u else SOFT)
        c.setStrokeColor(LINE)
        c.roundRect(bx, Y(yy + bh), bw, bh, 8, stroke=1, fill=1)
        c.setFillColor(HexColor(0x4338CA) if is_u else INK)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(bx + 10, Y(yy + 11), "You" if is_u else "Assistant")
        c.setFillColor(INK); c.setFont("Helvetica", 10)
        for j, ln in enumerate(lines):
            c.drawString(bx + 10, Y(yy + 24 + j * 13), ln)
        yy += bh + 8

    # Learnings & limitations (right)
    lx = 40 + colw + 30
    ly = panel(c, lx, ctop, colw, 300, "Learnings & Limitations", PINK)
    c.setFillColor(GREEN); c.setFont("Helvetica-Bold", 11)
    c.drawString(lx + 18, Y(ly), "What worked"); ly += 16
    ly = bullets(c, [
        "Schema validation eliminated a whole class of bugs (hallucinated / misplaced filters).",
        "Entity resolution was the single biggest robustness win (‘iphone’→apple, typos).",
        "Dense embeddings beat TF-IDF for ‘vibe’ queries (‘long flights’ ≈ noise-cancelling).",
        "Splitting NLU + an info-gathering gate made multi-turn behaviour reliable.",
    ], lx + 18, ly, colw - 36, size=10, leading=12.5, gap=2)
    ly += 4
    c.setFillColor(HexColor(0xB91C1C)); c.setFont("Helvetica-Bold", 11)
    c.drawString(lx + 18, Y(ly), "What was hard / we'd change"); ly += 16
    bullets(c, [
        "LLMs occasionally emit a ‘null’ string or leak a percentile price cap — fixed with code guards.",
        "Three LLM calls/turn → ~10s latency; route with a smaller/faster model next.",
        "Free image sources miss budget brands — solved via GSMArena (100% phones; ToS caveat).",
        "Learned ranker needs real click data to beat TOPSIS; flywheel is built, awaiting traffic.",
    ], lx + 18, ly, colw - 36, size=10, leading=12.5, gap=2)

    # ── Lower band: methodology + reflection ──
    btop = 520
    my = panel(c, 40, btop, colw, 268, "Evaluation Methodology", INDIGO)
    bullets(c, [
        "Deterministic suites (free, instant): 20 entity-resolution cases (aliases, typos, synonyms) "
        "and 10 slot-validation cases (control fields, impossible values, unknown keys).",
        "LLM suites: 22 single-utterance intent cases (incl. prior-context turns) and 10 multi-turn "
        "scenarios graded by a state predicate (correct action + filters + non-empty results).",
        "Reproducible via `python evaluate.py` → eval_results.json; this poster renders those numbers.",
        "Iterative: the harness caught two real bugs (undo over-revert, fresh-search contamination) "
        "which were fixed — task success rose 70% → 90%.",
    ], 58, my + 4, colw - 36, size=10, leading=12.5, gap=3)

    rx2 = 40 + colw + 30
    ry2 = panel(c, rx2, btop, colw, 268, "Reflection — What does the system need?", PINK)
    bullets(c, [
        "The product category — gates the schema, question order, and ranking weights.",
        "Structured preferences — extracted directly (specific) or elicited via questions (explore), "
        "then validated against the schema.",
        "An anchor for relative critiques — the previous recommendation’s distribution, to ground "
        "‘cheaper’ / ‘bigger battery’.",
        "A multi-criteria scoring function — TOPSIS over normalised specs, upgradable to a learned ranker.",
        "Dataset statistics — percentiles that turn vague language (‘cheap’, ‘premium’) into concrete bounds.",
    ], rx2 + 18, ry2 + 4, colw - 36, size=10, leading=12.5, gap=3)

    _footer(c, 3)
    c.showPage()


def _footer(c, n):
    c.setStrokeColor(LINE); c.setLineWidth(1)
    c.line(40, Y(PH - 28), PW - 40, Y(PH - 28))
    c.setFillColor(MUTED); c.setFont("Helvetica", 9)
    c.drawString(40, Y(PH - 16), f"{TITLE} — {GROUP}")
    c.drawCentredString(PW / 2, Y(PH - 16), "Applied Generative AI · SS 2026")
    c.drawRightString(PW - 40, Y(PH - 16), f"Page {n} of 3")


def main():
    m = load_metrics()
    c = canvas.Canvas("poster.pdf", pagesize=PAGE)
    page1(c, m)
    page2(c, m)
    page3(c, m)
    c.save()
    print("Wrote poster.pdf (3 × A3 landscape)" + ("" if m else "  [no eval_results.json — run evaluate.py for metrics]"))


if __name__ == "__main__":
    main()
