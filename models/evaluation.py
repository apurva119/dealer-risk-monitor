"""
Evaluation focused on how the collections team actually uses the output:
"If we can only investigate N dealers each week, how much future delinquency do we catch?"

weekly_topn      Each test week: rank dealers, take the top N. Recall@N = share of that week's
                 at-risk dealers (will hit 60+ DPD within 13 weeks) found in the top N; Precision@N
                 = share of the top N that were truly at risk. Averaged over weeks.
event_capture    Dealer-level and lead-time aware. For each dealer that actually crossed 60 DPD, was it
                 in the top N at least once, at least `lead` weeks before it crossed?
bootstrap_ap     PR-AUC confidence interval, resampling dealers (not rows) because a dealer's weekly
                 rows are strongly correlated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

LABEL = "label_60dpd_13w"


def _ranked(df: pd.DataFrame, score: str, seed: int = 0) -> pd.DataFrame:
    """Add per-week rank (1 = riskiest); ties broken randomly but reproducibly."""
    rng = np.random.default_rng(seed)
    d = df.assign(_tie=rng.random(len(df)))
    d = d.sort_values(["week_idx", score, "_tie"], ascending=[True, False, False])
    d["rank"] = d.groupby("week_idx").cumcount() + 1
    return d


def weekly_topn(df: pd.DataFrame, score: str, ns) -> pd.DataFrame:
    d = _ranked(df, score)
    rows = []
    for n in ns:
        top = d[d["rank"] <= n]
        hits = top.groupby("week_idx")[LABEL].sum()
        pos = d.groupby("week_idx")[LABEL].sum()
        size = d.groupby("week_idx").size().clip(upper=n)
        valid = pos > 0
        rows.append(dict(n=n,
                         recall=float((hits[valid] / pos[valid]).mean()),
                         precision=float((hits / size).mean()),
                         random_recall=float((size[valid] / d.groupby("week_idx").size()[valid]).mean())))
    return pd.DataFrame(rows)


def event_capture(df: pd.DataFrame, score: str, ns, lead: int = 4) -> pd.DataFrame:
    d = _ranked(df, score)
    pos = d[d[LABEL] == 1]
    ev = (pos.assign(event_week=pos["week_idx"] + pos["_eval_weeks_to_event"])
             .groupby("dealer_id")["event_week"].first())
    d = d[d["dealer_id"].isin(ev.index)].copy()
    d["event_week"] = d["dealer_id"].map(ev)
    d = d[d["week_idx"] <= d["event_week"] - lead]          # only alerts raised >= `lead` weeks early
    best = d.groupby("dealer_id")["rank"].min().reindex(ev.index)   # NaN = no early week observable
    first_alert = {}
    out = []
    for n in ns:
        observable = best.notna()
        caught = (best <= n) & observable
        wk = d[d["rank"] <= n].groupby("dealer_id")["week_idx"].min()
        lead_wks = (ev.reindex(wk.index) - wk)
        out.append(dict(n=n, events=int(observable.sum()), capture=float(caught.sum() / max(observable.sum(), 1)),
                        median_lead_weeks=float(lead_wks.median()) if len(lead_wks) else np.nan))
    return pd.DataFrame(out)


def bootstrap_ap(df: pd.DataFrame, score: str, n_boot: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx_by = df.reset_index(drop=True).groupby("dealer_id").indices
    dealers = np.array(list(idx_by))
    y, s = df[LABEL].values, df[score].values
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(dealers, len(dealers), replace=True)
        idx = np.concatenate([idx_by[d] for d in pick])
        if y[idx].sum() > 0:
            vals.append(average_precision_score(y[idx], s[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarize(df: pd.DataFrame, score: str) -> dict:
    lo, hi = bootstrap_ap(df, score)
    return dict(pr_auc=float(average_precision_score(df[LABEL], df[score])), pr_auc_ci95=[lo, hi],
                roc_auc=float(roc_auc_score(df[LABEL], df[score])))
