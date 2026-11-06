"""
Train, tune and save the five classifiers.  Run from the project root:

    python model/train_and_tune.py                       # downloads UCI dataset 697
    python model/train_and_tune.py --csv data/x.csv --target Target
"""
import argparse, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from scoring import score_predictions
from studentfeatures import add_progress_features

# >>> CHOOSE YOUR OWN VALUES HERE (they change every number in your report) <<<
RANDOM_STATE = 8131
HOLDOUT_FRACTION = 0.22
CV_SPLITS = 4
# -----------------------------------------------------------------------------

ap = argparse.ArgumentParser()
ap.add_argument("--csv")
ap.add_argument("--target")
ap.add_argument("--uci-id", type=int, default=697)
cli = ap.parse_args()


def tidy(name):
    return re.sub(r"\s+", " ", str(name)).strip()


def fetch_table():
    if cli.csv:
        tbl = pd.read_csv(cli.csv)
        return tbl, tidy(cli.target), Path(cli.csv).name
    from ucimlrepo import fetch_ucirepo
    bundle = fetch_ucirepo(id=cli.uci_id)
    tbl = pd.concat([bundle.data.features, bundle.data.targets], axis=1)
    return tbl, tidy(bundle.data.targets.columns[0]), f"UCI Machine Learning Repository (id {cli.uci_id})"


table, label_col, origin = fetch_table()
DATASET_NAME = "Predict Students' Dropout and Academic Success" if not cli.csv else Path(cli.csv).stem
table.columns = [tidy(c) for c in table.columns]
table = table.dropna().reset_index(drop=True)
table.to_csv(ROOT / "data" / "full_dataset.csv", index=False)

encoder = LabelEncoder().fit(table[label_col])
class_names = [str(c) for c in encoder.classes_]
feats = table.drop(columns=label_col)
labels = pd.Series(encoder.transform(table[label_col]), name=label_col)
assert feats.shape[1] >= 12 and len(table) >= 500, "need >=12 features and >=500 rows"

f_train, f_hold, l_train, l_hold = train_test_split(
    feats, labels, test_size=HOLDOUT_FRACTION, stratify=labels, random_state=RANDOM_STATE)
hold_out = f_hold.copy()
hold_out[label_col] = encoder.inverse_transform(l_hold)
hold_out.to_csv(ROOT / "test_data.csv", index=False)


def build(estimator, scale):
    steps = [("derive", FunctionTransformer(add_progress_features))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", estimator))
    return Pipeline(steps)


CANDIDATES = [
    ("Logistic Regression", "logreg", build(LogisticRegression(max_iter=3000, class_weight="balanced"), True),
     {"clf__C": [0.05, 0.3, 1, 3]}),
    ("Decision Tree", "dtree", build(DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE), False),
     {"clf__max_depth": [5, 7, 10], "clf__min_samples_leaf": [3, 8, 15], "clf__criterion": ["gini", "entropy"]}),
    ("kNN", "knn", build(KNeighborsClassifier(), True),
     {"clf__n_neighbors": [7, 11, 17, 25], "clf__weights": ["uniform", "distance"], "clf__p": [1, 2]}),
    ("Naive Bayes", "gnb", build(GaussianNB(), False),
     {"clf__var_smoothing": [1e-9, 1e-8, 1e-6, 1e-4]}),
    ("Support Vector Machine", "svm", build(SVC(class_weight="balanced", random_state=RANDOM_STATE), True),
     {"clf__C": [0.5, 2, 8], "clf__gamma": ["scale", 0.02]}),
]

scoreboard, chosen = [], {}
for display, stem, pipe, grid in CANDIDATES:
    search = GridSearchCV(pipe, grid, cv=CV_SPLITS, scoring="f1_macro", n_jobs=-1).fit(f_train, l_train)
    fitted = search.best_estimator_
    if stem == "svm":                               # probabilities are needed for AUC
        fitted.set_params(clf__probability=True).fit(f_train, l_train)
    joblib.dump(fitted, ROOT / "model" / f"{stem}.joblib")
    chosen[display] = {k.split("__")[1]: v for k, v in search.best_params_.items()}
    scoreboard.append({"ML Model Name": display,
                       **score_predictions(l_hold, fitted.predict(f_hold), fitted.predict_proba(f_hold))})
    print(f"finished {display}: {chosen[display]}")

full_precision = {row["ML Model Name"]: {k: float(v) for k, v in row.items() if k != "ML Model Name"} for row in scoreboard}
json.dump(full_precision, open(ROOT / "model" / "metrics_full.json", "w"), indent=2)
board = pd.DataFrame(scoreboard).round(4)
board.to_csv(ROOT / "model" / "metrics_summary.csv", index=False)
json.dump({"target": label_col, "classes": class_names, "raw_features": list(feats.columns),
           "best_params": chosen, "random_state": RANDOM_STATE, "sklearn": sklearn.__version__,
           "dataset_name": DATASET_NAME, "dataset_source": origin, "python": sys.version.split()[0]},
          open(ROOT / "model" / "run_meta.json", "w"), indent=2, default=str)

# ---- README generation -----------------------------------------------------
top_f1 = board.loc[board["F1"].idxmax(), "ML Model Name"]
top_mcc = board.loc[board["MCC"].idxmax(), "ML Model Name"]
low_mcc = board.loc[board["MCC"].idxmin(), "ML Model Name"]
quick = (f"Highest F1: **{top_f1}**; highest MCC: **{top_mcc}**; lowest MCC: **{low_mcc}** "
         f"(spread {board['MCC'].max() - board['MCC'].min():.3f}).")
dist = ", ".join(f"{k}: {v}" for k, v in table[label_col].value_counts().items())
text = (ROOT / "README_template.md").read_text(encoding="utf-8")
subs = {"{{RESULTS_TABLE}}": board.to_markdown(index=False), "{{QUICK_READ}}": quick,
        "{{N_ROWS}}": str(len(table)), "{{N_FEATURES}}": str(feats.shape[1]), "{{CLASS_COUNTS}}": dist,
        "{{N_TEST}}": str(len(l_hold)), "{{SEED}}": str(RANDOM_STATE), "{{SOURCE}}": origin,
        "{{HOLDOUT}}": f"{HOLDOUT_FRACTION:.0%}", "{{CV}}": str(CV_SPLITS),
        "{{BEST_PARAMS}}": "\n".join(f"- **{m}**: `{p}`" for m, p in chosen.items()),
        "{{SKLEARN}}": sklearn.__version__, "{{PY}}": sys.version.split()[0]}
for key, val in subs.items():
    text = text.replace(key, val)
(ROOT / "README.md").write_text(text, encoding="utf-8")
print("\n" + board.to_markdown(index=False))
print(f"\nTip: pin scikit-learn=={sklearn.__version__} in requirements.txt before deploying.")
