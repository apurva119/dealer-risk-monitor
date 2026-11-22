# Dealer Risk Monitor

> A machine learning application for prioritising high-risk dealers using temporal modelling and an interactive Streamlit dashboard.

Dealer Risk Monitor predicts which dealers are most likely to become at risk within the next **13 weeks** using machine learning (Logistic Regression and XGBoost). The model ranks dealers by predicted risk, helping reviewers focus their limited weekly review capacity on the highest-priority cases.

> **Live Demo:** Coming soon. The application currently runs locally using Streamlit.

---

## Dashboard

| Overview | Review Queue | Dealer Lookup |
|----------|--------------|---------------|
| ![](images/overview.png) | ![](images/queue.png) | ![](images/lookup.png) |

---

## Highlights

- Time-aware evaluation using a temporal train/test split
- Verified 13-week purge gap matching the prediction horizon
- Benchmarked a rule-based baseline, Logistic Regression, and XGBoost
- Business-focused evaluation using PR-AUC and Weekly Recall@25
- Interactive Streamlit dashboard for operational risk review
- Downloadable ranked review queue for operational teams

---
## Key Results

- PR-AUC improved from **0.824 (Rule-based)** to **0.849 (XGBoost)**
- Predicts dealer risk **13 weeks in advance**
- Evaluated using a leakage-free temporal split with a verified 13-week purge gap
- Prioritises the Top-25 dealers each week for manual review

---

## Business Problem

Operational teams have limited capacity to manually review dealers each week. The objective of this project is to identify dealers that are likely to become at risk within the next **13 weeks**, allowing reviewers to intervene before payment issues become more severe.

Each week, every dealer receives a predicted risk score. Dealers are ranked by predicted risk, and the highest-risk dealers are prioritised for manual review.

Because the objective is prioritisation rather than binary classification, the project is evaluated using ranking-based metrics that reflect the real operational workflow.

---

## Dataset

This project uses anonymised dealer-level data. The raw dataset is proprietary and is **not included** in this repository.

The repository contains only the source code, trained model artifacts, and demo prediction files required to reproduce the dashboard.

| Item | Value |
|------|------:|
| Test period | 46 weeks |
| Dealers per week | ~464 |
| At-risk dealers per week | ~3.6 (0.77%) |
| Prediction horizon | 13 weeks |

The dataset is highly imbalanced (0.77% positives), making **PR-AUC** and **Weekly Recall@25** more informative than overall accuracy.

A random Top-25 review queue would identify only about **5%** of at-risk dealers.

---

## Feature Engineering

Features are engineered from historical dealer behaviour using rolling windows and trend-based statistics. Every feature at week **t** is computed using only information available at week **t**, preventing temporal data leakage.

Feature groups include:

- **Payment behaviour** – average payment delay, delay trends, delay volatility and collection ratios over rolling 4-, 8- and 12-week windows.
- **Ageing & delinquency** – days past due, overdue balance ratios, unpaid invoice counts and outstanding balances.
- **Business activity** – order trends, invoice volumes, GST turnover changes and outstanding balance relative to business activity.
- **External signals** – credit bureau scores, recent score changes, GST registration status and active court cases.
- **Peer context** – dealer performance relative to the portfolio median, including payment delays and credit scores.
- **Dealer profile** – dealer size, credit terms, tenure and geographic region.
- **Calendar features** – cyclical month encodings to capture seasonality.

---

## Models

The following approaches were evaluated:

- Rule-based baseline
- Logistic Regression
- XGBoost

---

## Evaluation

Models are evaluated using a **strictly time-based train/test split**.

### Temporal Split

Training uses earlier weeks while testing uses later weeks. Random train/test splits are never used.

### Purge Gap

A verified **13-week purge gap** is maintained between the final training week (**83**) and the first testing week (**97**), matching the prediction horizon and preventing temporal leakage.

The split can be verified by running:

```bash
python check_split.py
```

The script validates the temporal split boundary. Feature engineering is designed so that every feature at week **t** uses only information available at that time.

### Metrics

Primary evaluation metrics:

- **PR-AUC** – appropriate for highly imbalanced datasets.
- **Weekly Recall@25** – reflects the operational review capacity of reviewing only the top 25 dealers each week.

---

## Results

| Model | PR-AUC |
|------|-------:|
| Rule-based baseline | 0.824 |
| Logistic Regression | 0.844 |
| **XGBoost** | **0.849** |

XGBoost achieved the highest PR-AUC (0.849), followed closely by Logistic Regression (0.844) and the rule-based baseline (0.824). Although the improvement is modest, the machine learning models provide better prioritisation of the limited weekly review queue, enabling reviewers to focus on dealers most likely to become at risk.

---

## Dashboard Features

- Adjustable High and Medium risk thresholds
- Automatic risk tiers
- Portfolio ranking
- Ranked weekly review queue
- CSV export
- Dealer lookup
- Dealer vs portfolio median comparison
- Interactive risk gauge

The dashboard scores one prediction per dealer and presents a prioritised review queue for operational users.

Demo prediction files are included, allowing the dashboard to run without access to the original dataset.

---

## Pipeline

```text
Raw Dealer Data
        │
        ▼
Feature Engineering
        │
        ▼
Temporal Train/Test Split
        │
        ▼
Model Training
(Logistic Regression / XGBoost)
        │
        ▼
Risk Prediction
        │
        ▼
Interactive Streamlit Dashboard
```

---

## How to Run

### Dashboard

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt

streamlit run dashboard/app.py
```

### Training Pipeline

Place the raw source dataset inside the `data/` directory before running the training pipeline.

```bash
pip install -r requirements-train.txt

python models/train_models.py

python inference/predict.py
```

---

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- Streamlit
- Plotly
- Git

---

## Project Structure

```text
dealer-risk-monitor/
├── dashboard/          # Streamlit application
├── features/           # Feature engineering
├── inference/          # Prediction pipeline
├── models/             # Model training
├── notebooks/          # Exploratory analysis
├── artifacts/          # Saved models and predictions
├── images/             # README screenshots
├── data/               # Raw data (not included)
├── check_split.py
├── requirements.txt
├── requirements-train.txt
└── README.md
```

---

## Limitations

- The dashboard identifies **who** is high risk but not **why**.
- Predictions are generated offline rather than through a live inference service.
- Labels look 13 weeks ahead, so consecutive weekly observations for the same dealer are not independent.
- Performance estimates are based on a relatively small number of positive events because dealer risk is rare.
- The rule-based baseline already performs strongly, so the machine learning models provide incremental rather than dramatic performance improvements.

---

## Future Work

- SHAP explanations for individual dealer predictions
- Automated model retraining
- Real-time inference service
- Model monitoring and drift detection
- CI/CD pipeline
- Streamlit Cloud deployment

---

## License

This project is released under the MIT License.