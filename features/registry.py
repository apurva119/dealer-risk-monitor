"""
Feature registry: one place that describes every model feature.

Consumed by: build_features.py (validation + feature_list.json), the SHAP narrative generator
(groups drivers by category, phrases them with unit, knows which direction is risky) and the dashboard.

risk_direction:  +1 = higher value means higher risk | -1 = lower value means higher risk | 0 = context / non-monotonic
"""
from __future__ import annotations

PAY, AGE, BIZ, EXT, PEER, PROF, CAL = (
    "Payment Behaviour", "Ageing & Delinquency", "Business Activity",
    "External Signals", "Peer Context", "Dealer Profile", "Calendar")


def _f(description: str, category: str, unit: str, risk_direction: int) -> dict:
    return dict(description=description, category=category, unit=unit, risk_direction=risk_direction)


REGISTRY: dict[str, dict] = {
    # ---- payment behaviour
    "delay_mean_4w": _f("Mean payment delay (days past due date) over the last 4 weeks", PAY, "days", 1),
    "delay_mean_8w": _f("Mean payment delay over the last 8 weeks", PAY, "days", 1),
    "delay_mean_12w": _f("Mean payment delay over the last 12 weeks", PAY, "days", 1),
    "delay_std_8w": _f("Volatility (std. dev.) of payment delay over 8 weeks", PAY, "days", 1),
    "delay_slope_8w": _f("Trend in payment delay over 8 weeks (days of extra delay per week)", PAY, "days/week", 1),
    "delay_change_4w": _f("Change in 4-week mean delay versus 4 weeks ago", PAY, "days", 1),
    "collection_ratio_4w": _f("Payments received as a share of new orders, last 4 weeks", PAY, "ratio", -1),
    "collection_ratio_8w": _f("Payments received as a share of new orders, last 8 weeks", PAY, "ratio", -1),
    "collection_ratio_12w": _f("Payments received as a share of new orders, last 12 weeks", PAY, "ratio", -1),
    # ---- ageing & delinquency
    "max_dpd_open": _f("Days past due of the oldest unpaid invoice today", AGE, "days", 1),
    "dpd_max_8w": _f("Highest days-past-due seen in the last 8 weeks", AGE, "days", 1),
    "dpd_change_4w": _f("Change in oldest-invoice days-past-due versus 4 weeks ago", AGE, "days", 1),
    "dpd_change_8w": _f("Change in oldest-invoice days-past-due versus 8 weeks ago", AGE, "days", 1),
    "open_invoice_count": _f("Number of unpaid invoices", AGE, "count", 1),
    "share_overdue": _f("Share of outstanding balance that is overdue", AGE, "ratio", 1),
    "share_overdue_31plus": _f("Share of outstanding balance more than 30 days overdue", AGE, "ratio", 1),
    "share_overdue_61plus": _f("Share of outstanding balance more than 60 days overdue", AGE, "ratio", 1),
    "outstanding_to_orders_4w": _f("Outstanding balance relative to the last 4 weeks of orders", AGE, "ratio", 1),
    # ---- business activity
    "orders_trend": _f("Orders in the last 4 weeks versus the average 4-week block of the last 12 weeks", BIZ, "ratio", -1),
    "invoices_4w": _f("Number of invoices raised in the last 4 weeks", BIZ, "count", -1),
    "gst_change_12w": _f("Change in reported GST turnover versus 12 weeks ago", BIZ, "ratio", -1),
    "outstanding_to_gst": _f("Outstanding balance relative to reported monthly GST turnover", BIZ, "ratio", 1),
    # ---- external signals
    "credit_score": _f("Latest credit-bureau score", EXT, "points", -1),
    "score_change_8w": _f("Change in credit score over 8 weeks", EXT, "points", -1),
    "score_change_12w": _f("Change in credit score over 12 weeks", EXT, "points", -1),
    "court_case_active": _f("A court case has been filed against the dealer (1 = yes)", EXT, "flag", 1),
    "gst_active": _f("GST registration is active (1 = yes)", EXT, "flag", -1),
    # ---- peer context
    "delay_8w_vs_peers": _f("8-week mean delay minus the portfolio median for the same week", PEER, "days", 1),
    "score_vs_peers": _f("Credit score minus the portfolio median for the same week", PEER, "points", -1),
    # ---- profile
    "size_tier": _f("Dealer size (0 = small, 1 = medium, 2 = large)", PROF, "ordinal", 0),
    "credit_terms_days": _f("Credit period granted to the dealer", PROF, "days", 0),
    "tenure_years": _f("Years since the dealer was onboarded", PROF, "years", 0),
    "zone_east": _f("Dealer is in the East zone", PROF, "flag", 0),
    "zone_north": _f("Dealer is in the North zone", PROF, "flag", 0),
    "zone_south": _f("Dealer is in the South zone", PROF, "flag", 0),
    "zone_west": _f("Dealer is in the West zone", PROF, "flag", 0),
    # ---- calendar
    "month_sin": _f("Seasonality: sine of calendar month", CAL, "cyclic", 0),
    "month_cos": _f("Seasonality: cosine of calendar month", CAL, "cyclic", 0),
}
