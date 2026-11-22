"""Check the outer train/test split for label leakage.

Run from the project root:
    python check_split.py
"""

import pandas as pd
from pandas.api.types import is_numeric_dtype

FEATURES_PATH = "data/processed/feature_store.parquet"
PREDICTIONS_PATH = "artifacts/risk_predictions.csv"
HORIZON = 13  # label looks this many weeks ahead

# ----------------------------
# Load feature store
# ----------------------------
feat = pd.read_parquet(FEATURES_PATH)

required = {"split", "week_idx"}
missing = required - set(feat.columns)
if missing:
    raise ValueError(f"Missing columns in feature store: {sorted(missing)}")

print("Split counts:")
print(feat["split"].value_counts(), "\n")

# Warn about unexpected split labels
extra = set(feat["split"].dropna().unique()) - {"train", "test"}
if extra:
    print(
        f"NOTE: Other split labels present: {sorted(map(str, extra))}. "
        "The gap below ignores them; check they are intended.\n"
    )

train = feat.loc[feat["split"] == "train", "week_idx"]
test = feat.loc[feat["split"] == "test", "week_idx"]

if train.empty:
    raise ValueError("No rows labelled exactly 'train'.")

if test.empty:
    raise ValueError("No rows labelled exactly 'test'.")

# ----------------------------
# Compute gap
# ----------------------------
last_train = train.max()
first_test = test.min()

if is_numeric_dtype(feat["week_idx"]):
    gap = int(first_test - last_train - 1)
else:
    last_train = pd.to_datetime(last_train)
    first_test = pd.to_datetime(first_test)
    gap = (first_test - last_train).days // 7 - 1

verdict = (
    "OK (gap >= prediction horizon)"
    if gap >= HORIZON
    else f"LEAKAGE RISK (gap {gap}, need at least {HORIZON})"
)

print(f"Last train week : {last_train}")
print(f"First test week : {first_test}")
print(f"Gap             : {gap}")
print(f"VERDICT         : {verdict}")

# ----------------------------
# Predictions sanity check
# ----------------------------
pred = pd.read_csv(PREDICTIONS_PATH)

if "dealer_id" not in pred.columns:
    raise ValueError("dealer_id column missing from predictions.")

print("\nSample dealer IDs:")
print(pred["dealer_id"].head(10).tolist())

print(f"\nRows           : {len(pred):,}")
print(f"Unique dealers : {pred['dealer_id'].nunique():,}")

if len(pred) != pred["dealer_id"].nunique():
    print("WARNING: Duplicate dealer IDs found in predictions.")
else:
    print("One prediction row per dealer: YES")