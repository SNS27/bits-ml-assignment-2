# 🎓 Student Outcome Prediction – ML Assignment 2 (M.Tech DSE)

*Author: Shaleen Sharma (BITS ID 2025NS04049), M.Tech (DSE), BITS Pilani WILP, Machine Learning, Assignment 2.*

## a. Problem statement
Universities lose students quietly: some drop out, some stay enrolled past the normal duration, some graduate.
The goal is to **predict which of these three outcomes a student will reach** from enrolment details plus
first- and second-semester performance, so that tutoring teams can step in early. Five classifiers are tuned and
compared on the same hold-out set and served through an interactive Streamlit app.

## b. Dataset description
- **Source:** full_dataset.csv, *Predict Students' Dropout and Academic Success* (Realinho et al., 2021, CC BY 4.0).
- **Size:** 4424 students × 36 raw features (demographics, socio-economic status, application mode, course, curricular units, grades, macro-economic indicators).
- **Task:** multi-class classification. **Class distribution:** Graduate: 2209, Dropout: 1421, Enrolled: 794 (imbalanced).
- **Hold-out split:** 22% of the data (974 rows), stratified, `random_state=8131`, saved as `test_data.csv`.
- **Engineered features (inside every model pipeline):** pass-rate for semester 1 and 2, grade shift between semesters, and total units passed.
- **Imbalance handling:** `class_weight="balanced"` for Logistic Regression, Decision Tree and SVM.
- **Tuning:** 4-fold grid search optimising macro-F1. StandardScaler used for Logistic Regression, kNN and SVM.
- **Metric averaging:** Precision / Recall / F1 are weighted; AUC is one-vs-rest weighted; MCC is the multi-class generalisation.
- Tested with Python 3.13.18 and scikit-learn 1.6.1 (pinned in `requirements.txt`).

## c. GitHub repository
https://github.com/SNS27/bits-ml-assignment-2

Live app: https://student-outcome-predictor-shaleen-sharma.streamlit.app/

## d. Models used

| ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.7444 | 0.8946 |      0.767  |   0.7444 | 0.7534 | 0.5943 |
| Decision Tree          |     0.7454 | 0.8608 |      0.7576 |   0.7454 | 0.7498 | 0.5894 |
| kNN                    |     0.7331 | 0.8683 |      0.7202 |   0.7331 | 0.7137 | 0.5559 |
| Naive Bayes            |     0.7207 | 0.8365 |      0.706  |   0.7207 | 0.708  | 0.5348 |
| Support Vector Machine |     0.7474 | 0.8963 |      0.7711 |   0.7474 | 0.7557 | 0.5987 |

*Quick read:* Highest F1: **Support Vector Machine**; highest MCC: **Support Vector Machine**; lowest MCC: **Naive Bayes** (spread 0.064).

Selected hyper-parameters:

- **Logistic Regression**: `{'C': 1}`
- **Decision Tree**: `{'criterion': 'gini', 'max_depth': 5, 'min_samples_leaf': 15}`
- **kNN**: `{'n_neighbors': 7, 'p': 1, 'weights': 'uniform'}`
- **Naive Bayes**: `{'var_smoothing': 1e-09}`
- **Support Vector Machine**: `{'C': 2, 'gamma': 'scale'}`

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong second-best model (Accuracy = 0.7444, AUC = 0.8946, F1 = 0.7534, MCC = 0.5943) with `C = 1` and balanced class weighting. Feature standardization combined with engineered pass-rate and grade-shift indicators allows clean separation of the major classes, though it slightly under-captures non-linear transitions for the minority Enrolled students. |
| Decision Tree | Solid performance (Accuracy = 0.7454, F1 = 0.7498) using `gini`, `max_depth = 5`, and `min_samples_leaf = 15`. Pruning successfully prevents overfitting on high-cardinality features like Course codes and captures clear boundary rules on 2nd-semester units approved, but piecewise-constant probabilities result in a lower AUC (0.8608). |
| kNN | Moderate performance (Accuracy = 0.7331, AUC = 0.8683, F1 = 0.7137, MCC = 0.5559) with `k = 7`, `p = 1` (Manhattan distance), and uniform weighting. Manhattan distance handles the mixed feature space well, but the lack of native class weighting causes boundary noise in favour of the majority Graduate class. |
| Naive Bayes | Lowest scorer across metrics (Accuracy = 0.7207, AUC = 0.8365, F1 = 0.7080, MCC = 0.5348) with `var_smoothing = 1e-09`. GaussianNB's feature-independence assumption is heavily violated because 1st- and 2nd-semester curricular metrics and pass rates are inherently collinear, leading to overconfident posteriors and lower precision (0.7060). |
| Support Vector Machine | **Top-performing classifier across all six metrics** (Accuracy = 0.7474, AUC = 0.8963, Precision = 0.7711, Recall = 0.7474, F1 = 0.7557, MCC = 0.5987) with an RBF kernel (`C = 2`, `gamma = 'scale'`). The non-linear projection effectively maps complex academic shifts and socio-economic signals, achieving the cleanest boundary separation. |
| **Overall winner for this dataset** | **Support Vector Machine (SVM)** is the overall winner across every metric (F1 = 0.7557, AUC = 0.8963, MCC = 0.5987), with **Logistic Regression** serving as a reliable runner-up (F1 = 0.7534) for linear interpretability. |

## Repository layout
```
app.py                  Streamlit front-end (dashboard + report, ROC, single-student tabs)
studentfeatures.py      engineered features used by every pipeline
scoring.py              six-metric scorecard
requirements.txt  README.md  test_data.csv
.streamlit/config.toml  custom theme
model/                  train_and_tune.py, *.joblib (5 models), run_meta.json, metrics_summary.csv
data/                   full_dataset.csv
```
## How to run
- **Setup (once):** `bash setup.sh` (Git Bash/Linux/macOS) or `setup.bat` (Windows)
- **Train (optional, models are already saved):** `.venv/Scripts/python model/train_and_tune.py` (Windows) or `.venv/bin/python model/train_and_tune.py`
- **Launch the app:** `run_app.bat` (Windows) or `bash run_app.sh`. Loads the saved models and bundled `test_data.csv`, no downloads.
- **Test with the sample:** open the app (it shows `test_data.csv` by default), or download the sample from the sidebar and upload it again; the metrics must stay identical.
- **Verify:** `python tools/verify_models.py` reloads all five pipelines and checks all 30 metric values against the experiment.
