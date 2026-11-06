# 🎓 Student Outcome Prediction – ML Assignment 2 (M.Tech DSE)

*Author: Shaleen Sharma (BITS ID 2025NS04049), M.Tech (DSE), BITS Pilani WILP, Machine Learning, Assignment 2.*

## a. Problem statement
Universities lose students quietly: some drop out, some stay enrolled past the normal duration, some graduate.
The goal is to **predict which of these three outcomes a student will reach** from enrolment details plus
first- and second-semester performance, so that tutoring teams can step in early. Five classifiers are tuned and
compared on the same hold-out set and served through an interactive Streamlit app.

## b. Dataset description
- **Source:** {{SOURCE}}, *Predict Students' Dropout and Academic Success* (Realinho et al., 2021, CC BY 4.0).
- **Size:** {{N_ROWS}} students × {{N_FEATURES}} raw features (demographics, socio-economic status, application mode, course, curricular units, grades, macro-economic indicators).
- **Task:** multi-class classification. **Class distribution:** {{CLASS_COUNTS}} (imbalanced).
- **Hold-out split:** {{HOLDOUT}} of the data ({{N_TEST}} rows), stratified, `random_state={{SEED}}`, saved as `test_data.csv`.
- **Engineered features (inside every model pipeline):** pass-rate for semester 1 and 2, grade shift between semesters, and total units passed.
- **Imbalance handling:** `class_weight="balanced"` for Logistic Regression, Decision Tree and SVM.
- **Tuning:** {{CV}}-fold grid search optimising macro-F1. StandardScaler used for Logistic Regression, kNN and SVM.
- **Metric averaging:** Precision / Recall / F1 are weighted; AUC is one-vs-rest weighted; MCC is the multi-class generalisation.
- Tested with Python {{PY}} and scikit-learn {{SKLEARN}} (pinned in `requirements.txt`).

## c. GitHub repository
https://github.com/SNS27/bits-ml-assignment-2

Live app: <PASTE STREAMLIT URL>

## d. Models used

{{RESULTS_TABLE}}

*Quick read:* {{QUICK_READ}}

Selected hyper-parameters:

{{BEST_PARAMS}}

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | TODO |
| Decision Tree | TODO |
| kNN | TODO |
| Naive Bayes | TODO |
| Support Vector Machine | TODO |
| **Overall winner for this dataset** | TODO |

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
