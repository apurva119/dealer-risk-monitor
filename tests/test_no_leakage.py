"""
Leakage and correctness tests for the feature pipeline.
Run:  pytest -q
A small synthetic portfolio is generated once per test session.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "features"))
import build_features as bf  # noqa: E402


@pytest.fixture(scope="session")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    subprocess.run([sys.executable, str(ROOT / "simulator" / "simulate.py"),
                    "--dealers", "120", "--weeks", "110", "--seed", "7", "--out", str(out)],
                   check=True, capture_output=True)
    snap = pd.read_csv(out / "weekly_snapshots.csv", parse_dates=["week_start"])
    dealers = pd.read_csv(out / "dealer_master.csv")
    return snap, dealers


def test_features_use_only_past_data(data):
    """Features at week t must be identical whether or not future weeks exist."""
    snap, dealers = data
    T = 60
    full = bf.add_features(snap, dealers)
    trunc = bf.add_features(snap[snap.week_idx <= T], dealers)
    cols = bf.feature_columns(full)
    a = full[full.week_idx <= T].sort_values(["dealer_id", "week_idx"])[cols].reset_index(drop=True)
    b = trunc.sort_values(["dealer_id", "week_idx"])[cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-9)


def test_label_matches_brute_force(data):
    snap, dealers = data
    df = bf.add_label(bf.add_features(snap, dealers))
    rng = np.random.default_rng(0)
    for dealer in rng.choice(df.dealer_id.unique(), 15, replace=False):
        g = df[df.dealer_id == dealer].sort_values("week_idx").reset_index(drop=True)
        for t in range(0, len(g) - bf.HORIZON_WEEKS):
            window = g.loc[t + 1: t + bf.HORIZON_WEEKS, "max_dpd_open"]
            assert g.loc[t, bf.TARGET] == float((window > bf.DPD_THRESHOLD).any())
        assert g[bf.TARGET].iloc[-bf.HORIZON_WEEKS:].isna().all()      # censored, not "0"


def test_split_is_purged_and_ordered(data):
    snap, dealers = data
    df, feats, rep = bf.build(snap, dealers)
    tr, te = df[df.split == "train"], df[df.split == "test"]
    assert tr.week_idx.max() + bf.HORIZON_WEEKS < te.week_idx.min()    # no label-window overlap
    assert len(tr) > 0 and len(te) > 0


def test_only_new_delinquency_is_predicted(data):
    snap, dealers = data
    df, _, _ = bf.build(snap, dealers)
    assert (df.max_dpd_open <= bf.DPD_THRESHOLD).all()
    assert df[bf.TARGET].isin([0, 1]).all()


def test_no_forbidden_columns_in_features(data):
    snap, dealers = data
    df, feats, _ = bf.build(snap, dealers)
    bad = [c for c in feats if c.startswith(("_", "label", "health", "archetype"))
           or c in {"dealer_id", "week_start", "week_idx", "split"}]
    assert not bad, bad


def test_every_feature_is_documented(data):
    """No feature can reach a model without a description, category, unit and risk direction."""
    from registry import REGISTRY
    snap, dealers = data
    _, feats, _ = bf.build(snap, dealers)
    assert set(feats) <= set(REGISTRY), set(feats) - set(REGISTRY)
    assert set(REGISTRY) <= set(feats), f"registry entries with no feature: {set(REGISTRY) - set(feats)}"
    for name, meta in REGISTRY.items():
        assert {"description", "category", "unit", "risk_direction"} <= set(meta), name
        assert meta["risk_direction"] in (-1, 0, 1), name
