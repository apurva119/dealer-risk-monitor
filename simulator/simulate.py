#!/usr/bin/env python3
"""
Synthetic dealer-network simulator for the Dealer Risk Monitor project.

Core idea
---------
Every dealer has a hidden "health" score (0-100) that evolves week by week.
All observable signals (payment delays, order volume, credit score, GST turnover,
court cases, GST status) are *generated from* health, so they are naturally
correlated instead of independently random. Days-past-due (DPD) is never generated
directly: it is derived from invoice due dates vs. payment dates, the way a real
receivables system computes it.

Outputs (in --out directory)
----------------------------
dealer_master.csv        one row per dealer (static attributes)
invoices.csv             invoice-level history (paid_date empty = still open)
weekly_snapshots.csv     one row per dealer per week (what a monitoring tool would see)
events.csv               discrete events (court case filed, GST went inactive)
_truth_health.csv        hidden health per dealer-week  -> NEVER use as a model feature
_truth_dealers.csv       archetype per dealer           -> NEVER use as a model feature

Usage
-----
python simulator/simulate.py --dealers 500 --weeks 156 --seed 42 --out data/synthetic
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ARCHETYPE_SHARES = {
    "stable": 0.55,
    "seasonal": 0.20,
    "growing": 0.10,
    "deteriorating": 0.10,
    "shock": 0.05,
}

# invoices per week, typical invoice value (INR) by dealer size
TIERS = {
    "small": dict(share=0.50, inv_per_week=3, amount=80_000),
    "medium": dict(share=0.35, inv_per_week=7, amount=250_000),
    "large": dict(share=0.15, inv_per_week=14, amount=600_000),
}

CITIES = [
    ("Lucknow", "UP", "North"), ("Kanpur", "UP", "North"), ("Jaipur", "RJ", "North"),
    ("Chandigarh", "PB", "North"), ("Delhi", "DL", "North"), ("Mumbai", "MH", "West"),
    ("Pune", "MH", "West"), ("Ahmedabad", "GJ", "West"), ("Surat", "GJ", "West"),
    ("Bengaluru", "KA", "South"), ("Chennai", "TN", "South"), ("Hyderabad", "TS", "South"),
    ("Kochi", "KL", "South"), ("Kolkata", "WB", "East"), ("Patna", "BR", "East"),
    ("Bhubaneswar", "OD", "East"),
]
PREFIX = ["Apex", "Healthbridge", "Northcare", "Vidhan", "Metro", "Sunrise", "City", "Ayush",
          "Medi", "Prime", "Lifeline", "Shree", "Global", "United", "Royal", "Sai", "Om", "Bharat"]
SUFFIX = ["Medico", "Traders", "Distributors", "Surgicals", "Pharma", "Drug House",
          "Agencies", "Enterprises", "Healthcare"]


# --------------------------------------------------------------------------- dealers
def generate_dealers(n: int, start: pd.Timestamp, rng: np.random.Generator) -> pd.DataFrame:
    counts = {k: int(round(v * n)) for k, v in ARCHETYPE_SHARES.items()}
    counts["stable"] += n - sum(counts.values())
    archetypes = np.repeat(list(counts), list(counts.values()))
    rng.shuffle(archetypes)

    tier_names = list(TIERS)
    tiers = rng.choice(tier_names, n, p=[TIERS[t]["share"] for t in tier_names])

    rows, used = [], set()
    for i in range(n):
        city, state, zone = CITIES[rng.integers(len(CITIES))]
        name = f"{rng.choice(PREFIX)} {rng.choice(SUFFIX)}"
        if name in used:
            name = f"{name} {city}"
        while name in used:
            name = f"{name} II"
        used.add(name)
        rows.append(dict(
            dealer_id=f"D{i + 1:04d}",
            dealer_name=name,
            city=city, state=state, zone=zone,
            size_tier=tiers[i],
            credit_terms_days=int(rng.choice([30, 45, 60], p=[0.4, 0.4, 0.2])),
            onboarded_on=(start - pd.Timedelta(days=int(rng.integers(365, 365 * 8)))).date(),
            _habit_days=float(rng.normal(0, 3)),   # dealer's own payment style (helper)
            _resilience=float(rng.normal(0, 7)),   # some dealers keep paying despite low 'health' (and vice-versa)
            _score_offset=float(rng.normal(0, 45)), # static bureau-score offset: absolute level is a weak signal
            _archetype=archetypes[i],               # hidden (moved to truth file)
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- health
def simulate_health(arch: str, n_weeks: int, monsoon: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """health(t+1) = health(t) + mean_reversion + archetype_drift + random_dip + noise."""
    base = {"stable": 76, "seasonal": 72, "growing": 68, "deteriorating": 76, "shock": 77}[arch]
    base = float(np.clip(base + rng.normal(0, 5), 55, 92))
    # onset spread across the timeline; negative = already sliding at week 0 (portfolio starts at steady state, not cold)
    onset = int(rng.integers(-35, max(9, int(0.65 * n_weeks))))
    slope = rng.uniform(0.9, 1.7)
    rescued = rng.random() < 0.30                                   # ~30% of sliding dealers get turned around
    rescue_wk = onset + int(rng.integers(15, 40))
    shock_wk = int(rng.integers(25, max(26, n_weeks - 15)))
    shock_size = rng.uniform(40, 60)

    h = np.empty(n_weeks)
    h[0] = base
    if arch == "deteriorating" and onset < 0:
        h[0] = max(30.0, base - slope * (-onset))
    for t in range(1, n_weeks):
        target = base
        if arch == "seasonal":
            target = base - 16 * monsoon[t]            # monsoon cash crunch
        elif arch == "growing":
            target = min(92.0, base + 0.2 * t)
        drift = 0.12 * (target - h[t - 1])             # mean reversion

        if arch == "deteriorating" and t >= onset:
            drift = -slope                              # steady slide ...
            if rescued and t >= rescue_wk:
                drift = 0.08 * (base - h[t - 1])         # ... unless the dealer recovers
        if arch == "shock":
            if shock_wk <= t < shock_wk + 3:
                drift = -shock_size / 3                 # sudden collapse
            elif t >= shock_wk + 3:
                drift = 0.03 * ((base - 0.6 * shock_size) - h[t - 1])  # slow partial recovery

        dip = -rng.uniform(6, 12) if rng.random() < 0.005 else 0.0   # random bad quarter
        h[t] = np.clip(h[t - 1] + drift + dip + rng.normal(0, 1.6), 3, 98)
    return h


# --------------------------------------------------------------------------- invoices
def simulate_invoices(d: pd.Series, health: np.ndarray, weeks64: np.ndarray, festive: np.ndarray,
                      cutoff: np.datetime64, rng: np.random.Generator) -> pd.DataFrame:
    tier = TIERS[d.size_tier]
    terms = np.timedelta64(int(d.credit_terms_days), "D")
    out = []
    for t, wk in enumerate(weeks64):
        h = float(np.clip(health[t] + d._resilience, 3, 98))
        growth = 1 + 0.004 * t if d._archetype == "growing" else 1.0
        lam = tier["inv_per_week"] * growth * festive[t] * (0.5 + 0.5 * h / 100)  # stressed dealers order less
        n = rng.poisson(lam)
        if n == 0:
            continue
        amount = np.round(rng.lognormal(np.log(tier["amount"]), 0.5, n), -2)
        inv_date = wk + rng.integers(0, 7, n).astype("timedelta64[D]")
        due = inv_date + terms

        mu = max(-4.0, (72 - h) * 0.85) + d._habit_days      # healthier -> pays on time / early
        sd = 3 + 0.12 * max(0.0, 72 - h)
        delay = np.rint(rng.normal(mu, sd, n)).astype("timedelta64[D]")
        paid = np.maximum(due + delay, inv_date + np.timedelta64(1, "D"))

        never = rng.random(n) < np.clip((28 - h) / 100, 0, 0.25)  # very sick dealers stop paying some invoices
        paid = np.where(never | (paid > cutoff), np.datetime64("NaT", "D"), paid)

        out.append(pd.DataFrame(dict(dealer_id=d.dealer_id, invoice_date=inv_date, due_date=due,
                                     paid_date=paid, amount=amount)))
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- signals & events
def simulate_signals(health: np.ndarray, weeks: pd.DatetimeIndex, dealer_id: str, score_offset: float,
                     rng: np.random.Generator):
    n = len(health)
    court, gst = np.zeros(n, int), np.ones(n, int)
    events, c, g = [], False, True
    for t in range(n):
        h = health[t]
        if not c and rng.random() < (0.04 if h < 35 else 0.0004):   # small false-positive rate when healthy
            c = True
            events.append(dict(dealer_id=dealer_id, week_start=weeks[t], event_type="court_case_filed"))
        if g and rng.random() < (0.03 if h < 22 else 0.0002):
            g = False
            events.append(dict(dealer_id=dealer_id, week_start=weeks[t], event_type="gst_registration_inactive"))
        court[t], gst[t] = int(c), int(g)

    # bureau score refreshes every 4 weeks and lags health (uses trailing average)
    score, last, off = np.empty(n), None, int(rng.integers(0, 4))
    for t in range(n):
        if last is None or t % 4 == off:
            last = 520 + score_offset + 3.2 * health[max(0, t - 3):t + 1].mean() + rng.normal(0, 25) - 30 * court[t]
        score[t] = np.clip(round(last), 300, 900)
    return court, gst, score, events


# --------------------------------------------------------------------------- snapshots
def build_snapshots(inv: pd.DataFrame, dealers: pd.DataFrame, weeks: pd.DatetimeIndex,
                    court: np.ndarray, gst: np.ndarray, score: np.ndarray,
                    health: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    nd, nw = len(dealers), len(weeks)
    code = dealers.reset_index(drop=True).reset_index().set_index("dealer_id")["index"]
    dcode = inv["dealer_id"].map(code).values
    idate, due, paid = inv["invoice_date"].values, inv["due_date"].values, inv["paid_date"].values
    amt = inv["amount"].values
    delay = (paid - due) / np.timedelta64(1, "D")           # NaN where paid is NaT

    def by_dealer(mask, w=None):
        return np.bincount(dcode[mask], weights=None if w is None else w[mask], minlength=nd)

    frames = []
    for w, start in enumerate(weeks):
        S = start.to_datetime64().astype("datetime64[D]")
        E = S + np.timedelta64(6, "D")

        raised = (idate >= S) & (idate <= E)
        coll = (paid >= S) & (paid <= E)
        is_open = (idate <= E) & (np.isnat(paid) | (paid > E))
        dpd = (E - due) / np.timedelta64(1, "D")

        n_coll = by_dealer(coll)
        avg_delay = np.divide(by_dealer(coll, np.nan_to_num(delay)), n_coll,
                              out=np.full(nd, np.nan), where=n_coll > 0)
        maxdpd = np.zeros(nd)
        np.maximum.at(maxdpd, dcode[is_open], np.clip(dpd[is_open], 0, None))

        frames.append(pd.DataFrame(dict(
            dealer_id=dealers["dealer_id"].values,
            week_start=start,
            week_idx=w,
            orders_value=by_dealer(raised, amt),
            invoices_raised=by_dealer(raised),
            collections_value=by_dealer(coll, amt),
            avg_payment_delay_days=avg_delay,
            outstanding_total=by_dealer(is_open, amt),
            open_invoice_count=by_dealer(is_open),
            overdue_1_30=by_dealer(is_open & (dpd > 0) & (dpd <= 30), amt),
            overdue_31_60=by_dealer(is_open & (dpd > 30) & (dpd <= 60), amt),
            overdue_61_90=by_dealer(is_open & (dpd > 60) & (dpd <= 90), amt),
            overdue_90_plus=by_dealer(is_open & (dpd > 90), amt),
            max_dpd_open=maxdpd,
            credit_score=score[:, w],
            gst_active=gst[:, w],
            court_case_active=court[:, w],
            _health=health[:, w],
        )))
    snap = pd.concat(frames, ignore_index=True).sort_values(["dealer_id", "week_idx"]).reset_index(drop=True)

    # GST turnover: monthly refresh, loosely tied to purchases and (hidden) health
    orders4 = snap.groupby("dealer_id")["orders_value"].transform(lambda s: s.rolling(4, min_periods=1).sum())
    raw = orders4 * 1.3 * (0.7 + 0.3 * snap["_health"] / 100) * rng.lognormal(0, 0.08, len(snap))
    off = snap["dealer_id"].map({d: int(rng.integers(0, 4)) for d in dealers["dealer_id"]})
    held = raw.where((snap["week_idx"] + off) % 4 == 0)
    snap["gst_turnover_monthly"] = held.groupby(snap["dealer_id"]).transform(lambda s: s.ffill().bfill()).round(-2)
    return snap


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dealers", type=int, default=500)
    ap.add_argument("--weeks", type=int, default=156)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", default="2024-01-01", help="must be a Monday")
    ap.add_argument("--out", default="data/synthetic")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    weeks = pd.date_range(a.start, periods=a.weeks, freq="W-MON")
    weeks64 = weeks.values.astype("datetime64[D]")
    cutoff = weeks64[-1] + np.timedelta64(6, "D")
    monsoon = np.isin(weeks.month, [6, 7, 8, 9]).astype(float)
    festive = np.where(np.isin(weeks.month, [10, 11]), 1.25, 1.0)

    dealers = generate_dealers(a.dealers, weeks[0], rng)
    H = np.zeros((a.dealers, a.weeks)); C = np.zeros_like(H, dtype=int)
    G = np.zeros_like(C); S = np.zeros_like(H)
    invoices, events = [], []

    for i, d in dealers.iterrows():
        H[i] = simulate_health(d._archetype, a.weeks, monsoon, rng)
        C[i], G[i], S[i], ev = simulate_signals(H[i], weeks, d.dealer_id, d._score_offset, rng)
        events += ev
        invoices.append(simulate_invoices(d, H[i], weeks64, festive, cutoff, rng))

    inv = pd.concat(invoices, ignore_index=True).sort_values(["invoice_date", "dealer_id"]).reset_index(drop=True)
    inv.insert(0, "invoice_id", [f"INV{i + 1:07d}" for i in range(len(inv))])
    snap = build_snapshots(inv, dealers, weeks, C, G, S, H, rng)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dealers.drop(columns=[c for c in dealers.columns if c.startswith("_")]).to_csv(out / "dealer_master.csv", index=False)
    inv.to_csv(out / "invoices.csv", index=False)
    snap.drop(columns=["_health"]).to_csv(out / "weekly_snapshots.csv", index=False)
    pd.DataFrame(events).sort_values("week_start").to_csv(out / "events.csv", index=False)
    dealers[["dealer_id", "_archetype", "_resilience"]].rename(columns={"_archetype": "archetype", "_resilience": "resilience"}).to_csv(
        out / "_truth_dealers.csv", index=False)
    snap[["dealer_id", "week_start", "_health"]].rename(columns={"_health": "health"}).to_csv(
        out / "_truth_health.csv", index=False)

    json.dump(dict(simulator_version="1.2", created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   seed=a.seed, dealers=a.dealers, weeks=a.weeks, start=a.start,
                   n_invoices=len(inv), n_snapshots=len(snap), n_events=len(events)),
              open(out / "run_meta.json", "w"), indent=2)

    # ---- sanity report
    arch = dealers.set_index("dealer_id")["_archetype"]
    peak = snap.groupby("dealer_id")["max_dpd_open"].max()
    print(f"dealers={a.dealers} weeks={a.weeks} invoices={len(inv):,} snapshots={len(snap):,} events={len(events)}")
    print("\nShare of dealers that ever reach 60+ DPD, by archetype:")
    print((peak.gt(60).groupby(arch).mean().round(2)).to_string())
    print(f"\nOverall: {peak.gt(60).mean():.0%} of dealers hit 60+ DPD at some point")
    print("\nWithin-dealer correlation of hidden health with signals (dealer size removed):")
    cols = ["avg_payment_delay_days", "max_dpd_open", "credit_score", "gst_turnover_monthly", "orders_value"]
    demeaned = snap[cols + ["_health"]].groupby(snap["dealer_id"]).transform(lambda x: x - x.mean())
    print(demeaned.corr()["_health"].drop("_health").round(2).to_string())

if __name__ == "__main__":
    main()
