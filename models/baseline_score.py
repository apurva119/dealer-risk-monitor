"""
Transparent weighted-rule baseline (0-100). This is what a credit analyst would build without ML.

The weights below were set from domain reasoning BEFORE looking at any test results and are NOT fitted
to data. Its job is to be the benchmark that the ML models have to beat, and its component points double
as a built-in explanation (which rules fired).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHTS = {  # maximum points per rule; sums to 100
    "oldest_invoice_dpd": 25,
    "avg_payment_delay": 20,
    "aged_balance_31plus": 15,
    "delay_trend": 10,
    "credit_score_drop": 10,
    "low_credit_score": 5,
    "order_decline": 5,
    "court_case": 5,
    "gst_inactive": 5,
}
assert sum(WEIGHTS.values()) == 100


def _scaled(x: pd.Series, lo: float, hi: float) -> pd.Series:
    return ((x.fillna(lo) - lo) / (hi - lo)).clip(0, 1)


def baseline_components(df: pd.DataFrame) -> pd.DataFrame:
    w = WEIGHTS
    return pd.DataFrame({
        "oldest_invoice_dpd": w["oldest_invoice_dpd"] * _scaled(df["max_dpd_open"], 0, 60),
        "avg_payment_delay": w["avg_payment_delay"] * _scaled(df["delay_mean_8w"], 0, 45),
        "aged_balance_31plus": w["aged_balance_31plus"] * _scaled(df["share_overdue_31plus"], 0, 1),
        "delay_trend": w["delay_trend"] * _scaled(df["delay_slope_8w"], 0, 1.5),
        "credit_score_drop": w["credit_score_drop"] * _scaled(-df["score_change_12w"], 0, 100),
        "low_credit_score": w["low_credit_score"] * _scaled(700 - df["credit_score"], 0, 250),
        "order_decline": w["order_decline"] * _scaled(-df["orders_trend"], 0, 0.5),
        "court_case": w["court_case"] * df["court_case_active"],
        "gst_inactive": w["gst_inactive"] * (1 - df["gst_active"]),
    }, index=df.index)


def baseline_score(df: pd.DataFrame) -> pd.Series:
    return baseline_components(df).sum(axis=1)
