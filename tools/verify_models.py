"""Reload the saved pipelines, evaluate test_data.csv, and compare all 30 metric values with the experiment."""
import json, sys
from pathlib import Path
import joblib, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import studentfeatures  # noqa: F401
from scoring import score_predictions

TOL = 1e-9
meta = json.loads((ROOT / "model/run_meta.json").read_text())
ref = json.loads((ROOT / "model/metrics_full.json").read_text())
stems = {"Logistic Regression": "logreg", "Decision Tree": "dtree", "kNN": "knn",
         "Naive Bayes": "gnb", "Support Vector Machine": "svm"}
df = pd.read_csv(ROOT / "test_data.csv")
X, y = df[meta["raw_features"]], df[meta["target"]].astype(str).map({c: i for i, c in enumerate(meta["classes"])})
bad = checked = 0
for name, stem in stems.items():
    m = joblib.load(ROOT / "model" / f"{stem}.joblib")
    got = score_predictions(y, m.predict(X), m.predict_proba(X))
    for k, v in got.items():
        checked += 1
        ok = abs(v - ref[name][k]) <= TOL
        bad += (not ok)
        if not ok:
            print(f"MISMATCH {name} {k}: reloaded={v:.12f} experiment={ref[name][k]:.12f}")
print(f"checked {checked} metric values (tolerance {TOL}); mismatches: {bad}")
print("PASS" if bad == 0 and checked == 30 else "FAIL")
sys.exit(0 if bad == 0 and checked == 30 else 1)
