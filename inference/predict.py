#!/usr/bin/env python3

from pathlib import Path
import argparse
import json

import pandas as pd
import xgboost as xgb


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--features", default="data/processed/feature_store.parquet")
    parser.add_argument("--feature-list", default="data/processed/feature_list.json")
    parser.add_argument("--model", default="artifacts/xgb.json")
    parser.add_argument("--out", default="artifacts/risk_predictions.csv")

    args = parser.parse_args()

    df = pd.read_parquet(args.features)

    with open(args.feature_list) as f:
        meta = json.load(f)

    FEATURES = meta["features"]

    model = xgb.XGBClassifier()
    model.load_model(args.model)

    df["risk_probability"] = model.predict_proba(df[FEATURES])[:, 1]

    latest_week = df["week_start"].max()

    latest = (
        df[df["week_start"] == latest_week]
        .copy()
        .sort_values("risk_probability", ascending=False)
    )

    latest["risk_rank"] = range(1, len(latest) + 1)

    latest[
        ["dealer_id", "week_start", "risk_probability", "risk_rank"]
    ].to_csv(args.out, index=False)

    print(f"Saved {len(latest)} predictions to {args.out}")


if __name__ == "__main__":
    main()