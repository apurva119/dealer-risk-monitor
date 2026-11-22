"""Hand-checkable tests for the ranking metrics and the baseline score."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
import evaluation as ev  # noqa: E402
from baseline_score import WEIGHTS, baseline_score  # noqa: E402


@pytest.fixture
def frame():
    # week 0 ranking: A(.9) B(.8) C(.2) D(.1) ; week 1 ranking: C(.9) D(.8) B(.2) A(.1)
    rows = [  # week, dealer, score, label, weeks_to_event
        (0, "A", .9, 0, None), (0, "B", .8, 1, 6), (0, "C", .2, 1, 5), (0, "D", .1, 0, None),
        (1, "A", .1, 0, None), (1, "B", .2, 0, None), (1, "C", .9, 1, 4), (1, "D", .8, 0, None),
    ]
    return pd.DataFrame(rows, columns=["week_idx", "dealer_id", "score", "label_60dpd_13w", "_eval_weeks_to_event"])


def test_weekly_topn_by_hand(frame):
    r = ev.weekly_topn(frame, "score", [1, 2]).set_index("n")
    assert r.loc[1, "recall"] == pytest.approx((0 / 2 + 1 / 1) / 2)     # week0 top1=A (miss), week1 top1=C (hit)
    assert r.loc[2, "recall"] == pytest.approx((1 / 2 + 1 / 1) / 2)     # week0 {A,B} finds B, week1 {C,D} finds C
    assert r.loc[2, "precision"] == pytest.approx(0.5)


def test_event_capture_by_hand(frame):
    # events: B crosses at week 6, C at week 5. With lead >= 4 only alerts in weeks <= 2 / <= 1 count.
    r = ev.event_capture(frame, "score", [1, 2], lead=4).set_index("n")
    assert r.loc[1, "events"] == 2
    assert r.loc[1, "capture"] == pytest.approx(0.5)     # only C is ever rank 1 (week 1)
    assert r.loc[2, "capture"] == pytest.approx(1.0)     # B is rank 2 in week 0, C rank 1 in week 1


def test_event_capture_ignores_late_alerts(frame):
    # demanding 6 weeks of lead: B's event is week 6 -> only week 0 alerts count; C (event week 5) is unobservable
    r = ev.event_capture(frame, "score", [2], lead=6).iloc[0]
    assert r.events == 1 and r.capture == 1.0


def test_baseline_bounds():
    healthy = dict(max_dpd_open=0, delay_mean_8w=0, share_overdue_31plus=0, delay_slope_8w=0, score_change_12w=0,
                   credit_score=780, orders_trend=0.1, court_case_active=0, gst_active=1)
    risky = dict(max_dpd_open=60, delay_mean_8w=45, share_overdue_31plus=1, delay_slope_8w=1.5, score_change_12w=-100,
                 credit_score=450, orders_trend=-0.5, court_case_active=1, gst_active=0)
    s = baseline_score(pd.DataFrame([healthy, risky]))
    assert sum(WEIGHTS.values()) == 100
    assert s.iloc[0] == pytest.approx(0) and s.iloc[1] == pytest.approx(100)
