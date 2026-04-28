# scripts/validate_data.py

import pandas as pd
import sys

REQUIRED_FIELDS = {
    "smartphones": [
        "product_id", "name", "brand", "price_eur", "release_year",
        "availability", "description", "os", "storage_gb", "ram_gb",
        "battery_mah", "main_camera_mp", "screen_size_inch", "five_g",
        "weight_g", "value_score", "performance_score",
        "feature_score", "popularity_score"
    ],
    "laptops": [
        "product_id", "name", "brand", "price_eur", "release_year",
        "availability", "description", "os", "ram_gb", "storage_gb",
        "processor_tier", "gpu_dedicated", "screen_size_inch",
        "weight_kg", "battery_hours", "primary_use",
        "value_score", "performance_score",
        "feature_score", "popularity_score"
    ]
}

def validate_csv(path: str, category: str):
    df = pd.read_csv(path)
    required = REQUIRED_FIELDS[category]
    missing_cols = [c for c in required if c not in df.columns]
    null_counts = df[required].isnull().sum()
    
    print(f"\n{'='*40}")
    print(f"Category: {category} | Rows: {len(df)}")
    if missing_cols:
        print(f"MISSING COLUMNS: {missing_cols}")
        sys.exit(1)
    if null_counts.any():
        print(f"NULL VALUES FOUND:\n{null_counts[null_counts > 0]}")
    duplicate_ids = df[df['product_id'].duplicated()]['product_id'].tolist()
    if duplicate_ids:
        print(f"DUPLICATE IDs: {duplicate_ids}")
        sys.exit(1)
    # Score range check
    for score_col in ["value_score", "performance_score",
                       "feature_score", "popularity_score"]:
        if not df[score_col].between(0, 1).all():
            print(f"SCORE OUT OF RANGE: {score_col}")
            sys.exit(1)
    print(f"Validation PASSED")

validate_csv("data/smartphones.csv", "smartphones")
validate_csv("data/headphones.csv", "headphones")
