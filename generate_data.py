"""
generate_data.py
Run once to create the product CSV files under products/
Uses the LLM to generate realistic synthetic data.
"""

import os
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── LLM setup ────────────────────────────────────────────────────────────────

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "openai")
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── Prompts ───────────────────────────────────────────────────────────────────

SMARTPHONE_PROMPT = """Generate exactly 80 realistic smartphone products as a JSON array.
Each object must have EXACTLY these fields (no extras):
- id: string like "sp_001"
- brand: one of [Apple, Samsung, Google, OnePlus, Xiaomi, Sony, Motorola]
- model: realistic model name
- price_eur: integer between 150 and 1500
- display_inches: float between 5.5 and 7.0
- battery_mah: integer between 3000 and 6000
- camera_mp: integer, main camera megapixels between 12 and 200
- os: one of [iOS, Android]
- ram_gb: one of [4, 6, 8, 12, 16]
- storage_gb: one of [64, 128, 256, 512, 1024]
- release_year: integer between 2021 and 2024
- weight_g: integer between 150 and 230
- has_5g: boolean
- color: one of [Black, White, Blue, Silver, Gold, Green, Purple]

Return ONLY the JSON array, no explanation, no markdown."""

WASHING_MACHINE_PROMPT = """Generate exactly 60 realistic washing machine products as a JSON array.
Each object must have EXACTLY these fields:
- id: string like "wm_001"
- brand: one of [Bosch, Siemens, Miele, LG, Samsung, AEG, Whirlpool, Beko]
- model: realistic model name
- price_eur: integer between 300 and 2000
- capacity_kg: one of [6, 7, 8, 9, 10, 11, 12]
- energy_class: one of [A, B, C, D] (A is best)
- noise_db: integer between 44 and 78
- spin_rpm: one of [1000, 1200, 1400, 1600]
- load_type: one of [front, top]
- has_steam: boolean
- has_wifi: boolean
- warranty_years: one of [2, 3, 5, 10]
- color: one of [White, Silver, Black, Anthracite]

Return ONLY the JSON array, no explanation, no markdown."""

LAPTOP_PROMPT = """Generate exactly 70 realistic laptop products as a JSON array.
Each object must have EXACTLY these fields:
- id: string like "lp_001"
- brand: one of [Apple, Dell, HP, Lenovo, Asus, Acer, Microsoft, Razer]
- model: realistic model name
- price_eur: integer between 400 and 3500
- display_inches: float between 13.0 and 17.3
- ram_gb: one of [8, 16, 32, 64]
- storage_gb: one of [256, 512, 1024, 2048]
- cpu_brand: one of [Intel, AMD, Apple]
- gpu_type: one of [integrated, dedicated]
- battery_wh: integer between 40 and 100
- weight_kg: float between 0.9 and 3.5
- os: one of [Windows, macOS, Linux]
- release_year: integer between 2021 and 2024
- has_touchscreen: boolean
- category: one of [ultrabook, gaming, workstation, budget, 2-in-1]

Return ONLY the JSON array, no explanation, no markdown."""


# ── Generator ─────────────────────────────────────────────────────────────────

def generate_products(prompt: str, filename: str, llm):
    print(f"Generating {filename}...")
    response = llm.invoke(prompt)
    text = response.content.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    data = json.loads(text.strip())
    df = pd.DataFrame(data)
    path = Path("products") / filename
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  ✓ Saved {len(df)} products to {path}")
    return df


def main():
    llm = get_llm()
    generate_products(SMARTPHONE_PROMPT,    "smartphones.csv",      llm)
    generate_products(WASHING_MACHINE_PROMPT, "washing_machines.csv", llm)
    generate_products(LAPTOP_PROMPT,        "laptops.csv",          llm)
    print("\n✅ All product data generated!")


if __name__ == "__main__":
    main()
