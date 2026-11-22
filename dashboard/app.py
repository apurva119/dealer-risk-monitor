import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]  # app.py lives in dashboard/, artifacts/ is one level up
DATA_PATH = ROOT / "artifacts" / "risk_predictions.csv"

MODEL_INFO = "XGBoost + Logistic Regression"
HORIZON_WEEKS = 13
TIER_COLORS = {"High": "#E5484D", "Medium": "#F5A524", "Low": "#30A46C"}

st.set_page_config(page_title="Dealer Risk Monitor", page_icon="📈", layout="wide")


def natural_key(value):
    """Sort IDs like D2 before D10. Safe for IDs with no digits or numeric IDs."""
    m = re.match(r"(\D*)(\d*)", str(value))
    return (m.group(1), int(m.group(2) or 0), str(value))


# ==========================
# Data loading (cached + validated)
# ==========================
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Prediction file contains no rows.")
    missing = {"dealer_id", "risk_probability"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    return df


try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Predictions file not found at `{DATA_PATH}`. Run the scoring pipeline first.")
    st.stop()
except ValueError as e:
    st.error(str(e))
    st.stop()


# ==========================
# Sidebar: business rules
# ==========================
st.sidebar.header("Risk policy")
high_t = st.sidebar.slider("High-risk threshold", 0.0, 1.0, 0.70, 0.05)
med_t = min(st.sidebar.slider("Medium-risk threshold", 0.0, 1.0, 0.40, 0.05), high_t)
st.sidebar.caption(
    "Dealers at or above the high threshold are flagged for immediate review. "
    "Adjust thresholds to match the team's review capacity."
)

df = df.copy()
p = df["risk_probability"]
df["risk_tier"] = np.select([p >= high_t, p >= med_t], ["High", "Medium"], default="Low")

# Rank 1 = riskiest. method="min" gives tied dealers the same rank.
df = df.sort_values("risk_probability", ascending=False).reset_index(drop=True)
df.insert(0, "rank", df["risk_probability"].rank(ascending=False, method="min").astype(int))

# ==========================
# Header + KPIs
# ==========================
st.title("📈 Dealer Risk Monitor")
st.caption("Model-scored risk for every dealer, with a review queue prioritised by risk.")

counts = df["risk_tier"].value_counts()
n_high, n_med, n_low = (int(counts.get(t, 0)) for t in ("High", "Medium", "Low"))

k1, k2, k3, k4 = st.columns(4)
k1.metric("Dealers scored", f"{len(df):,}")
k2.metric("High-risk dealers", f"{n_high:,}", f"{n_high / len(df):.1%} of portfolio", delta_color="off")
k3.metric("Average risk", f"{df['risk_probability'].mean():.2%}")
k4.metric("Highest risk", f"{df['risk_probability'].max():.2%}")

st.markdown(
    f"**Portfolio:** 🔴 {n_high:,} High &nbsp;·&nbsp; 🟠 {n_med:,} Medium &nbsp;·&nbsp; 🟢 {n_low:,} Low"
)

tab_overview, tab_queue, tab_dealer = st.tabs(["Overview", "Review queue", "Dealer lookup"])

# ==========================
# Tab 1: Overview
# ==========================
with tab_overview:
    left, right = st.columns([2, 1])

    with left:
        fig = px.histogram(
            df, x="risk_probability", nbins=30, color="risk_tier",
            color_discrete_map=TIER_COLORS,
            category_orders={"risk_tier": ["Low", "Medium", "High"]},
            title="Distribution of dealer risk",
            labels={"risk_probability": "Risk probability"},
        )
        fig.add_vline(x=med_t, line_dash="dot", line_color=TIER_COLORS["Medium"], annotation_text="Medium")
        fig.add_vline(x=high_t, line_dash="dash", line_color=TIER_COLORS["High"], annotation_text="High")
        fig.update_layout(bargap=0.05, legend_title_text="Tier")
        st.plotly_chart(fig, width="stretch")

    with right:
        tier_counts = (
            df["risk_tier"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
        )
        tier_counts.columns = ["risk_tier", "dealers"]
        fig2 = px.pie(
            tier_counts, names="risk_tier", values="dealers", hole=0.55,
            color="risk_tier", color_discrete_map=TIER_COLORS, title="Portfolio by tier",
        )
        st.plotly_chart(fig2, width="stretch")

# ==========================
# Tab 2: Review queue
# ==========================
with tab_queue:
    c1, c2 = st.columns([1, 2])
    top_n = c1.slider("Dealers to show", 5, 100, 20)
    tiers = c2.multiselect("Tiers", ["High", "Medium", "Low"], default=["High", "Medium"])

    queue = df[df["risk_tier"].isin(tiers)].head(top_n)  # df is already sorted by risk

    if queue.empty:
        st.info("No dealers match the selected tiers.")
    else:
        top = queue.head(10).assign(dealer_id=lambda d: d["dealer_id"].astype(str))
        fig_top = px.bar(
            top, x="risk_probability", y="dealer_id", orientation="h",
            color="risk_tier", color_discrete_map=TIER_COLORS,
            title="Top 10 in the current queue",
            labels={"risk_probability": "Risk probability", "dealer_id": "Dealer"},
        )
        fig_top.update_layout(yaxis={"categoryorder": "total ascending"}, legend_title_text="Tier")
        st.plotly_chart(fig_top, width="stretch")

        st.dataframe(
            queue,
            hide_index=True,
            width="stretch",
            column_config={
                "risk_probability": st.column_config.ProgressColumn(
                    "Risk", min_value=0.0, max_value=1.0, format="%.2f"
                ),
            },
        )
        st.download_button(
            "Download review queue (CSV)",
            queue.to_csv(index=False).encode("utf-8"),
            file_name="dealer_review_queue.csv",
            mime="text/csv",
        )

# ==========================
# Tab 3: Dealer lookup
# ==========================
with tab_dealer:
    dealer_ids = sorted(df["dealer_id"].unique(), key=natural_key)
    dealer_id = st.selectbox("Dealer ID", dealer_ids)
    row = df[df["dealer_id"] == dealer_id].iloc[0]

    g1, g2 = st.columns([1, 2])

    with g1:
        if row["risk_tier"] == "High":
            st.error("🔴 HIGH RISK")
        elif row["risk_tier"] == "Medium":
            st.warning("🟠 MEDIUM RISK")
        else:
            st.success("🟢 LOW RISK")

        st.metric("Portfolio rank", f"#{row['rank']}", f"of {len(df):,} dealers", delta_color="off")

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=row["risk_probability"] * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": TIER_COLORS[row["risk_tier"]]},
                    "steps": [
                        {"range": [0, med_t * 100], "color": "rgba(48,164,108,0.15)"},
                        {"range": [med_t * 100, high_t * 100], "color": "rgba(245,165,36,0.15)"},
                        {"range": [high_t * 100, 100], "color": "rgba(229,72,77,0.15)"},
                    ],
                },
            )
        )
        gauge.update_layout(height=250, margin=dict(t=20, b=0, l=20, r=20))
        st.plotly_chart(gauge, width="stretch")

    with g2:
        numeric_cols = [
            c
            for c in df.select_dtypes(include="number").columns
            if c.lower() not in {"risk_probability", "rank"}
            and not c.lower().endswith("_id")
        ]
        if numeric_cols:
            st.markdown("**This dealer vs portfolio median**")
            compare = pd.DataFrame(
                {
                    "Dealer": row[numeric_cols].astype(float),
                    "Portfolio median": df[numeric_cols].median(),
                }
            )
            st.dataframe(compare.round(2), width="stretch")
        else:
            st.dataframe(row.to_frame("value"), width="stretch")

# ==========================
# Footer
# ==========================
st.divider()
st.caption(
    f"Dealer Risk Monitoring System · {MODEL_INFO} · Prediction horizon: {HORIZON_WEEKS} weeks"
)