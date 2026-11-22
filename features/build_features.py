#!/usr/bin/env python3
"""
Feature engineering, label generation and time-based splitting.

Prediction task (made at the end of every week t, for every dealer):
    label_60dpd_13w = 1  if the dealer has an open invoice that is 60+ days past due
                          at ANY week-end in t+1 ... t+13   (~90 days ahead)

Leakage safeguards
------------------
1. Every feature at week t uses only data from weeks <= t (verified by tests/test_no_leakage.py).
2. Dealer-weeks that are ALREADY 60+ DPD at week t are excluded: we predict NEW delinquency,
   not something that has already happened.
3. The last 13 weeks have no complete forward window, so they have no label and are dropped
   (right-censoring) rather than being treated as "no default".
4. Train/test split is by time, with a 13-week purge gap, because label windows look 13 weeks ahead:
   a training label at week t would otherwise overlap the first test weeks.
5. Hidden-truth columns never enter the feature store. `feature_list.json` is the single source of
   truth for which columns a model may use.

Usage
-----
python features/build_features.py --data data/synthetic --out data/processed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from registry import REGISTRY

HORIZON_WEEKS = 13          # ~90 days
DPD_THRESHOLD = 60
WARMUP_WEEKS = 12           # longest rolling window; earlier rows have incomplete history
WINDOWS = (4, 8, 12)
TARGET = "label_60dpd_13w"
ID_COLS = ["dealer_id", "week_start", "week_idx", "split"]
TIER_ORDER = {"small": 0, "medium": 1, "large": 2}


def _slope(y: np.ndarray) -> float:
    """Least-squares slope of y against 0..n-1, ignoring NaNs."""
    ok = ~np.isnan(y)
    if ok.sum() < 3:
        return np.nan
    x = np.arange(len(y), dtype=float)[ok]
    x -= x.mean()
    yy = y[ok]
    return float((x * (yy - yy.mean())).sum() / (x ** 2).sum())


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.where(b > 0)).replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- features
def add_features(snap: pd.DataFrame, dealers: pd.DataFrame) -> pd.DataFrame:
    """Return one row per dealer-week with engineered features. Uses only past data per row."""
    df = snap.sort_values(["dealer_id", "week_idx"]).reset_index(drop=True).copy()
    by = lambda col: df.groupby("dealer_id", sort=False)[col]          # noqa: E731
    roll = lambda col, w, fn, mp=1: by(col).transform(                  # noqa: E731
        lambda s: getattr(s.rolling(w, min_periods=mp), fn)())

    # ---- current-state ratios (scale-free, so a small and a large dealer are comparable)
    overdue = df[["overdue_1_30", "overdue_31_60", "overdue_61_90", "overdue_90_plus"]].sum(axis=1)
    df["share_overdue"] = _safe_div(overdue, df["outstanding_total"]).fillna(0)
    df["share_overdue_31plus"] = _safe_div(
        df[["overdue_31_60", "overdue_61_90", "overdue_90_plus"]].sum(axis=1), df["outstanding_total"]).fillna(0)
    df["share_overdue_61plus"] = _safe_div(
        df[["overdue_61_90", "overdue_90_plus"]].sum(axis=1), df["outstanding_total"]).fillna(0)

    # ---- payment behaviour
    df["delay_last_known"] = by("avg_payment_delay_days").ffill()   # NaN in weeks with no payments -> carry forward
    for w in WINDOWS:
        df[f"delay_mean_{w}w"] = roll("delay_last_known", w, "mean")
        df[f"orders_sum_{w}w"] = roll("orders_value", w, "sum")
        df[f"collected_sum_{w}w"] = roll("collections_value", w, "sum")
        df[f"collection_ratio_{w}w"] = _safe_div(df[f"collected_sum_{w}w"], df[f"orders_sum_{w}w"])
    df["delay_std_8w"] = roll("delay_last_known", 8, "std", mp=3)
    df["delay_slope_8w"] = by("delay_last_known").transform(
        lambda s: s.rolling(8, min_periods=4).apply(_slope, raw=True))
    df["delay_change_4w"] = df["delay_mean_4w"] - by("delay_mean_4w").shift(4)

    # ---- ageing / delinquency trajectory
    df["dpd_max_8w"] = roll("max_dpd_open", 8, "max")
    df["dpd_change_4w"] = df["max_dpd_open"] - by("max_dpd_open").shift(4)
    df["dpd_change_8w"] = df["max_dpd_open"] - by("max_dpd_open").shift(8)
    df["outstanding_to_orders_4w"] = _safe_div(df["outstanding_total"], df["orders_sum_4w"])

    # ---- business activity
    df["orders_trend"] = _safe_div(df["orders_sum_4w"], df["orders_sum_12w"] / 3) - 1   # last 4w vs avg 4w block of last 12w
    df["invoices_4w"] = roll("invoices_raised", 4, "sum")
    df["gst_change_12w"] = _safe_div(df["gst_turnover_monthly"], by("gst_turnover_monthly").shift(12)) - 1
    df["outstanding_to_gst"] = _safe_div(df["outstanding_total"], df["gst_turnover_monthly"])

    # ---- external signals
    df["score_change_8w"] = df["credit_score"] - by("credit_score").shift(8)
    df["score_change_12w"] = df["credit_score"] - by("credit_score").shift(12)

    # ---- peer context (same-week cross-section only -> no look-ahead)
    wk = df.groupby("week_idx")
    df["delay_8w_vs_peers"] = df["delay_mean_8w"] - wk["delay_mean_8w"].transform("median")
    df["score_vs_peers"] = df["credit_score"] - wk["credit_score"].transform("median")

    # ---- static attributes and calendar
    d = dealers.set_index("dealer_id")
    df["size_tier"] = df["dealer_id"].map(d["size_tier"]).map(TIER_ORDER)
    df["credit_terms_days"] = df["dealer_id"].map(d["credit_terms_days"])
    onboard = pd.to_datetime(df["dealer_id"].map(d["onboarded_on"]))
    df["tenure_years"] = (df["week_start"] - onboard).dt.days / 365.25
    for z in sorted(dealers["zone"].unique()):
        df[f"zone_{z.lower()}"] = (df["dealer_id"].map(d["zone"]) == z).astype(int)
    month = df["week_start"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df


RAW_COLS = {  # columns that exist in the snapshot file or are intermediate helpers
    "orders_value", "invoices_raised", "collections_value", "avg_payment_delay_days", "outstanding_total",
    "overdue_1_30", "overdue_31_60", "overdue_61_90", "overdue_90_plus", "gst_turnover_monthly",
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model-usable columns: everything engineered or current-state, excluding ids, label and eval-only cols."""
    skip = set(ID_COLS) | {TARGET} | {c for c in df.columns if c.startswith("_eval_")}
    skip |= {"delay_last_known"}                                  # helper
    skip |= {c for c in df.columns if c.startswith(("orders_sum_", "collected_sum_"))}  # absolute sizes; ratios kept
    skip |= RAW_COLS                                              # raw absolute-size columns; scale-free versions kept
    return [c for c in df.columns if c not in skip]


# --------------------------------------------------------------------------- label
def add_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("dealer_id", sort=False)["max_dpd_open"]
    # max over weeks t .. t+H-1 (reverse rolling), then shift(-1) -> max over t+1 .. t+H
    fwd_incl = g.transform(lambda s: s[::-1].rolling(HORIZON_WEEKS, min_periods=HORIZON_WEEKS).max()[::-1])
    fwd = fwd_incl.groupby(df["dealer_id"]).shift(-1)
    df[TARGET] = (fwd > DPD_THRESHOLD).astype(float).where(fwd.notna())

    # evaluation-only: weeks until the dealer first crosses 60 DPD (never a feature)
    first = df[df["max_dpd_open"] > DPD_THRESHOLD].groupby("dealer_id")["week_idx"].min()
    df["_eval_weeks_to_event"] = df["dealer_id"].map(first) - df["week_idx"]
    return df


# --------------------------------------------------------------------------- split
def assign_split(df: pd.DataFrame, train_end: int | None, n_weeks: int) -> tuple[pd.DataFrame, int, int]:
    last_labelled = n_weeks - 1 - HORIZON_WEEKS
    if train_end is None:
        train_end = WARMUP_WEEKS + int(0.55 * (last_labelled - WARMUP_WEEKS))
    test_start = train_end + HORIZON_WEEKS + 1          # purge gap = label horizon
    df = df.copy()
    df["split"] = np.select([df["week_idx"] <= train_end, df["week_idx"] >= test_start], ["train", "test"], "purge")
    return df, train_end, test_start


# --------------------------------------------------------------------------- main
def build(snap: pd.DataFrame, dealers: pd.DataFrame, train_end: int | None = None):
    n_weeks = int(snap["week_idx"].max()) + 1
    df = add_label(add_features(snap, dealers))
    df, train_end, test_start = assign_split(df, train_end, n_weeks)

    n0 = len(df)
    df = df[df["week_idx"] >= WARMUP_WEEKS]                       # warm-up
    df = df[df[TARGET].notna()]                                   # no complete forward window
    df = df[df["max_dpd_open"] <= DPD_THRESHOLD]                  # not already delinquent
    df = df[df["split"] != "purge"]                               # purge gap
    df[TARGET] = df[TARGET].astype(int)
    feats = feature_columns(df)
    missing = [f for f in feats if f not in REGISTRY]
    if missing:
        raise KeyError(f"Features missing from registry.py: {missing}")
    report = dict(rows_before=n0, rows_after=len(df), train_end_week=train_end, test_start_week=test_start)
    return df.reset_index(drop=True), feats, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--train-end", type=int, default=None, help="last training week index")
    a = ap.parse_args()

    snap = pd.read_csv(Path(a.data) / "weekly_snapshots.csv", parse_dates=["week_start"])
    dealers = pd.read_csv(Path(a.data) / "dealer_master.csv")
    df, feats, rep = build(snap, dealers, a.train_end)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    keep = ID_COLS + feats + [TARGET] + [c for c in df.columns if c.startswith("_eval_")]
    df[keep].to_parquet(out / "feature_store.parquet", index=False)
    json.dump(dict(features=feats, registry={f: REGISTRY[f] for f in feats}, target=TARGET, id_cols=ID_COLS, horizon_weeks=HORIZON_WEEKS,
                   dpd_threshold=DPD_THRESHOLD, **rep),
              open(out / "feature_list.json", "w"), indent=2)

    print(f"feature store: {len(df):,} rows x {len(feats)} features  ->  {out}/feature_store.parquet")
    print(f"split: train weeks {WARMUP_WEEKS}-{rep['train_end_week']} | purge | test weeks {rep['test_start_week']}+")
    for s, g in df.groupby("split"):
        print(f"  {s:5s} rows={len(g):6,}  dealers={g.dealer_id.nunique():3d}  positive rate={g[TARGET].mean():.1%}")


if __name__ == "__main__":
    main()
