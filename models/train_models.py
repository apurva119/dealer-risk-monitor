#!/usr/bin/env python3
"""
Train and compare: weighted-rule baseline vs logistic regression vs XGBoost.

Hyper-parameters are chosen on a temporal validation slice carved from the END of the training period
(with the same 13-week purge gap), so the test period is touched exactly once, for the final report.

Usage
-----
python models/train_models.py --data data/processed --out artifacts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import evaluation as ev  # noqa: E402
from baseline_score import baseline_score  # noqa: E402

SEED = 0
NS = (10, 25, 50, 100)
LEAD_WEEKS = 4
MODELS = {"Weighted baseline": "score_baseline", "Logistic regression": "score_logreg", "XGBoost": "score_xgb"}


def inner_split(train: pd.DataFrame, train_end: int, horizon: int, val_weeks: int = 15):
    val_start = train_end - val_weeks + 1
    fit_end = val_start - horizon - 1                      # purge gap between inner-train and validation
    return train[train.week_idx <= fit_end], train[train.week_idx >= val_start]


def make_logreg(C: float):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def make_xgb(depth: int, mcw: int, n_est: int, spw: float, early: int | None):
    return xgb.XGBClassifier(n_estimators=n_est, learning_rate=0.05, max_depth=depth, min_child_weight=mcw,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, scale_pos_weight=spw,
                             eval_metric="aucpr", early_stopping_rounds=early, random_state=SEED, n_jobs=4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(Path(a.data) / "feature_store.parquet")
    meta = json.load(open(Path(a.data) / "feature_list.json"))
    F, T = meta["features"], meta["target"]
    train, test = df[df.split == "train"], df[df.split == "test"].copy()
    fit, val = inner_split(train, meta["train_end_week"], meta["horizon_weeks"])
    print(f"inner-train {len(fit):,} rows ({fit[T].sum()} pos) | validation {len(val):,} rows ({val[T].sum()} pos) | "
          f"train {len(train):,} ({train[T].sum()} pos) | test {len(test):,} ({test[T].sum()} pos)")

    # ---- logistic regression: choose regularisation on validation
    best_lr = max(((average_precision_score(val[T], make_logreg(C).fit(fit[F], fit[T]).predict_proba(val[F])[:, 1]), C)
                   for C in (0.01, 0.1, 1.0)))
    C = best_lr[1]
    logreg = make_logreg(C).fit(train[F], train[T])

    # ---- XGBoost: small grid, early stopping on validation, refit on full train
    spw = float(np.sqrt((fit[T] == 0).sum() / max(fit[T].sum(), 1)))
    best_xgb = None
    for depth, mcw in [(3, 5), (4, 5), (3, 20), (5, 20)]:
        m = make_xgb(depth, mcw, 800, spw, 40).fit(fit[F], fit[T], eval_set=[(val[F], val[T])], verbose=False)
        ap_val = average_precision_score(val[T], m.predict_proba(val[F])[:, 1])
        if best_xgb is None or ap_val > best_xgb[0]:
            best_xgb = (ap_val, depth, mcw, m.best_iteration + 1)
    _, depth, mcw, n_iter = best_xgb
    spw_full = float(np.sqrt((train[T] == 0).sum() / train[T].sum()))
    xgbm = make_xgb(depth, mcw, max(50, int(n_iter * 1.15)), spw_full, None).fit(train[F], train[T])
    print(f"chosen: logreg C={C} | xgboost depth={depth} min_child_weight={mcw} trees={int(n_iter * 1.15)}")

    # ---- score the test period (touched once)
    test["score_baseline"] = baseline_score(test).values
    test["score_logreg"] = logreg.predict_proba(test[F])[:, 1]
    test["score_xgb"] = xgbm.predict_proba(test[F])[:, 1]

    metrics = {}
    late = test.max_dpd_open <= 30                                  # dealers not yet visibly late
    for name, col in MODELS.items():
        m = ev.summarize(test, col)
        m["pr_auc_not_yet_late"] = float(average_precision_score(test.loc[late, T], test.loc[late, col]))
        m["weekly_topn"] = ev.weekly_topn(test, col, NS).to_dict("records")
        m[f"event_capture_{LEAD_WEEKS}w_lead"] = ev.event_capture(test, col, NS, LEAD_WEEKS).to_dict("records")
        metrics[name] = m

    # ---- report
    wk = test.groupby("week_idx")
    print(f"\ntest: {wk.ngroups} weeks | ~{wk.size().mean():.0f} dealers/week | ~{wk[T].sum().mean():.1f} at-risk dealers/week "
          f"| base rate {test[T].mean():.2%}\n")
    print(f"{'':22s}{'PR-AUC [95% CI]':26s}{'ROC':>6s}{'late-only PR':>14s}")
    for name, m in metrics.items():
        lo, hi = m["pr_auc_ci95"]
        print(f"{name:22s}{m['pr_auc']:.3f} [{lo:.3f}-{hi:.3f}]".ljust(48) + f"{m['roc_auc']:.3f}{m['pr_auc_not_yet_late']:>12.3f}")
    print("\nWeekly Recall@N (share of that week's at-risk dealers found in the top N)")
    print(f"{'':22s}" + "".join(f"{'N=' + str(n):>9s}" for n in NS))
    for name, m in metrics.items():
        print(f"{name:22s}" + "".join(f"{r['recall']:>9.0%}" for r in m["weekly_topn"]))
    rnd = metrics["XGBoost"]["weekly_topn"]
    print(f"{'random ranking':22s}" + "".join(f"{r['random_recall']:>9.0%}" for r in rnd))
    print(f"\nEvent capture: share of dealers that went 60+ DPD that were in the top N at least {LEAD_WEEKS} weeks earlier")
    print(f"{'':22s}" + "".join(f"{'N=' + str(n):>9s}" for n in NS) + f"   (events={metrics['XGBoost'][f'event_capture_{LEAD_WEEKS}w_lead'][0]['events']})")
    for name, m in metrics.items():
        print(f"{name:22s}" + "".join(f"{r['capture']:>9.0%}" for r in m[f"event_capture_{LEAD_WEEKS}w_lead"]))

    # ---- plots
    curve_ns = list(range(5, 151, 5))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for name, col in MODELS.items():
        c1 = ev.weekly_topn(test, col, curve_ns); c2 = ev.event_capture(test, col, curve_ns, LEAD_WEEKS)
        ax[0].plot(c1.n, c1.recall, label=name, lw=2); ax[1].plot(c2.n, c2.capture, label=name, lw=2)
    ax[0].plot(c1.n, c1.random_recall, "k--", lw=1, label="Random")
    ax[0].set(title="Weekly recall vs collections capacity", xlabel="Dealers investigated per week (N)",
              ylabel="Share of at-risk dealers found")
    ax[1].set(title=f"Dealers caught ≥{LEAD_WEEKS} weeks before 60+ DPD", xlabel="Dealers investigated per week (N)",
              ylabel="Share of delinquent dealers caught")
    for x in ax:
        x.set_ylim(0, 1.02); x.grid(alpha=.3); x.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out / "ranking_eval.png", dpi=150)

    # ---- artifacts
    joblib.dump(logreg, out / "logreg.joblib"); xgbm.save_model(out / "xgb.json")
    keep = ["dealer_id", "week_start", "week_idx", T, "_eval_weeks_to_event"] + list(MODELS.values())
    test[keep].to_parquet(out / "test_predictions.parquet", index=False)
    json.dump(dict(chosen=dict(logreg_C=C, xgb_depth=depth, xgb_min_child_weight=mcw, xgb_trees=int(n_iter * 1.15)),
                   metrics=metrics), open(out / "metrics.json", "w"), indent=2)
    print(f"\nsaved models, predictions, metrics.json and ranking_eval.png to {out}/")


if __name__ == "__main__":
    main()
