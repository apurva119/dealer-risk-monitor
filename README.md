# Dealer Risk Monitor

Dealer Risk Monitor predicts which dealers are most likely to become at risk within the next 13 weeks and ranks them by predicted risk, helping reviewers focus their limited weekly review capacity on the highest-priority cases.

![Dashboard Overview](images/overview.png)

> **Live demo:** TODO – add the Streamlit Cloud URL after deployment.

---

## Highlights

- Time-aware evaluation using a temporal train/test split
- Verified 13-week purge gap matching the prediction horizon
- XGBoost and Logistic Regression benchmarked against a rule-based baseline
- Business-focused evaluation using PR-AUC and Weekly Recall@25
- Interactive Streamlit dashboard for operational risk review
- Downloadable ranked review queue for operational teams

---

## Business Problem

**TODO**

Describe:

- What "at-risk" means (how the target label is defined)
- The 13-week prediction horizon
- Who uses the predictions
- Why only the Top-25 highest-risk dealers are reviewed each week

---

## Dataset

**TODO**

State whether the dataset is:

- Real
- Anonymised
- Synthetic

Explain any publishing restrictions if applicable.

| Item | Value |
|------|------:|
| Test period | 46 weeks |
| Dealers per week | ~464 |
| At-risk dealers per week | ~3.6 (0.77%) |
| Prediction horizon | 13 weeks |

The dataset is highly imbalanced (0.77% positives), making PR-AUC and Recall@25 more informative than accuracy.

A random Top-25 review queue would identify only about 5% of at-risk dealers.

---

## Feature Engineering

**TODO**

Describe the feature groups, for example:

- Payment behaviour
- Dealer activity
- Historical trends
- Rolling statistics
- Portfolio characteristics

State that every feature at week **t** is computed only from information available at week **t**.

---

## Models

- Rule-based baseline (**TODO:** describe the rule)
- Logistic Regression
- XGBoost

---

## Evaluation

The models are evaluated using a strictly time-based split.

### Temporal split

Training uses earlier weeks while testing uses later weeks.

Random train/test splits are never used.

### Purge gap

The temporal train/test split includes a **verified 13-week gap** between the final training week (**83**) and the first test week (**97**), matching the 13-week prediction horizon.

The verification can be reproduced by running:

```bash
python check_split.py
```

The script verifies the temporal split boundary only. Feature engineering should also be reviewed separately to ensure every feature at week **t** uses only information available at week **t**.

### Metrics

The primary evaluation metrics are:

- **PR-AUC**, appropriate for a 0.77% positive rate
- **Weekly Recall@25**, reflecting the operational review capacity of 25 dealers per week

### Statistical comparison

**TODO**

Report the paired bootstrap confidence interval comparing PR-AUC between models.

---

## Results

| Model | PR-AUC | ROC-AUC | Weekly Recall@25 |
|------|-------:|--------:|-----------------:|
| Rule-based baseline | 0.824 | TODO | TODO |
| Logistic Regression | 0.844 | TODO | TODO |
| XGBoost | **0.849** | TODO | TODO |

**TODO**

Summarise the practical findings.

Suggested wording (adapt after your statistical comparison):

> All three approaches performed strongly on the held-out temporal test set. XGBoost achieved the highest PR-AUC, although the improvement over Logistic Regression and the rule-based baseline was modest. The primary benefit of the machine learning models is improved prioritisation of the weekly review queue rather than a dramatic increase in predictive performance.

Also report the practical improvement, such as the number of additional at-risk dealer-weeks identified during the 46-week test period.

---

## Dashboard

| Overview | Review Queue | Dealer Lookup |
|----------|--------------|---------------|
| ![](images/overview.png) | ![](images/queue.png) | ![](images/lookup.png) |

### Features

- Adjustable High and Medium risk thresholds
- Automatic risk tiers
- Portfolio ranking
- Ranked review queue
- CSV export
- Dealer lookup
- Dealer vs portfolio median comparison
- Interactive risk gauge

The dashboard scores one prediction per dealer and presents a prioritised review queue for operational users.

---

## How to Run

### Dashboard only (uses committed demo predictions)

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt

streamlit run dashboard/app.py
```

### Full training pipeline

**TODO**

Describe where the raw data should be placed inside `data/`.

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

---

## Project Structure

```text
dealer-risk-monitor/
│
├── dashboard/
│   └── app.py
│
├── models/
│
├── inference/
│
├── notebooks/
│
├── artifacts/
│   ├── risk_predictions.csv
│   └── ranking_eval.png
│
├── images/
│   ├── overview.png
│   ├── queue.png
│   └── lookup.png
│
├── data/
│
├── requirements.txt
├── requirements-train.txt
├── check_split.py
└── README.md
```

---

## Limitations

- The dashboard explains **who** is high risk but not **why**.
- Predictions are generated offline rather than through a live inference service.
- Labels look 13 weeks ahead, so consecutive weekly observations for the same dealer are not independent.
- **TODO:** Report the number of positive events in the test period.
- **TODO:** Discuss the simplicity of the rule-based baseline and the relatively small performance improvement achieved by the machine learning models.

---

## Future Work

- SHAP explanations for individual dealer predictions
- Automated model retraining
- Real-time inference pipeline
- Model monitoring and drift detection
- CI/CD deployment pipeline