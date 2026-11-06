import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, auc, classification_report,
                             confusion_matrix, roc_curve)
from sklearn.preprocessing import label_binarize

import studentfeatures  # noqa: F401  (lets joblib unpickle the saved pipelines)
from scoring import score_predictions

HERE = Path(__file__).resolve().parent
META = json.loads((HERE / "model" / "run_meta.json").read_text())
LABEL, CLASSES, RAW = META["target"], META["classes"], META["raw_features"]
DATASET = META.get("dataset_name", "Student outcomes")
LINEUP = {"Logistic Regression": "logreg", "Decision Tree": "dtree", "kNN": "knn",
          "Naive Bayes": "gnb", "Support Vector Machine": "svm"}
PALETTE = ["#D1495B", "#EDAE49", "#00798C", "#30638E", "#6C5B7B", "#8D99AE"]
SAMPLE = HERE / "test_data.csv"

st.set_page_config(page_title="Campus Outcome Lab", page_icon="🎓", layout="wide")
st.markdown(
    """
    <style>
      .hero {background: linear-gradient(120deg,#D1495B 0%,#EDAE49 100%); padding: 1.2rem 1.6rem;
             border-radius: 14px; color: white; margin-bottom: 1rem;}
      .hero h1 {margin: 0; font-size: 1.9rem; color: white;}
      .hero p {margin: .3rem 0 0 0; opacity: .95;}
      div[data-testid="stMetric"] {background:#fff; border-left:6px solid #D1495B; padding:.6rem .9rem;
             border-radius:8px; box-shadow:0 1px 4px rgba(0,0,0,.08);}
    </style>
    <div class="hero"><h1>🎓 Campus Outcome Lab</h1>
    <p>Will a student graduate, stay enrolled or drop out? Five classic classifiers, evaluated on a labelled test file.</p></div>
    """,
    unsafe_allow_html=True,
)


class BadUpload(Exception):
    pass


@st.cache_resource
def fetch_model(label):
    return joblib.load(HERE / "model" / f"{LINEUP[label]}.joblib")


def read_labelled_csv(raw):
    """Validate a labelled test file; return (frame, numeric features, integer truth, notes)."""
    if not raw or not raw.strip():
        raise BadUpload("The file is empty.")
    try:
        frame = pd.read_csv(io.BytesIO(raw))
    except pd.errors.EmptyDataError:
        raise BadUpload("The file has no columns or rows.")
    except Exception as exc:
        raise BadUpload(f"The file could not be read as CSV ({exc}).")
    if frame.empty:
        raise BadUpload("The file has headers but no data rows.")
    frame.columns = [" ".join(str(c).split()) for c in frame.columns]
    missing = [c for c in RAW + [LABEL] if c not in frame.columns]
    if missing:
        raise BadUpload(f"{len(missing)} required column(s) are missing, e.g. {missing[:4]}. "
                        f"Expected the {len(RAW)} feature columns plus '{LABEL}'.")
    feats = frame[RAW].apply(pd.to_numeric, errors="coerce")
    bad = feats.columns[feats.isna().any()].tolist()
    if bad:
        raise BadUpload(f"Non-numeric or empty values found in: {bad[:5]}.")
    text = frame[LABEL].astype(str).str.strip()
    unknown = sorted(set(text) - set(CLASSES))
    if unknown:
        raise BadUpload(f"Unknown labels in '{LABEL}': {unknown[:5]}. Allowed labels: {CLASSES}.")
    truth = text.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    extra = [c for c in frame.columns if c not in RAW + [LABEL]]
    notes = [f"Ignored {len(extra)} extra column(s)."] if extra else []
    return frame, feats, truth, notes


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Control panel")
    uploaded = st.file_uploader("Upload labelled test CSV (optional)", type="csv")
    chosen = st.selectbox("Model", list(LINEUP))
    with open(SAMPLE, "rb") as fh:
        st.download_button("⬇ Download bundled sample CSV", fh, file_name="test_data.csv", mime="text/csv")
    st.caption(f"Expected headers: the {len(RAW)} original predictor columns plus `{LABEL}` "
               f"(values: {', '.join(CLASSES)}). Column order does not matter.")

# ------------------------------------------------------------------ load data
if uploaded is None:
    raw_bytes, origin = SAMPLE.read_bytes(), "bundled sample (test_data.csv)"
else:
    raw_bytes, origin = uploaded.getvalue(), f"uploaded file ({getattr(uploaded, 'name', 'upload.csv')})"
try:
    frame, X, truth, notes = read_labelled_csv(raw_bytes)
except BadUpload as err:
    st.error(f"❌ Upload rejected: {err}")
    st.stop()

st.markdown(f"**Dataset:** {DATASET} &nbsp;|&nbsp; **Active data:** {origin} &nbsp;|&nbsp; "
            f"**Rows evaluated:** {len(frame)}")
for n in notes:
    st.caption(n)
if len(set(truth)) < 2:
    st.warning("This file contains a single outcome class, so AUC is unavailable and some scores are not meaningful.")

# ------------------------------------------------------------------ evaluate all five
results = {}
for name in LINEUP:
    mdl = fetch_model(name)
    prob = mdl.predict_proba(X)
    pred = mdl.predict(X)
    results[name] = (pred, prob, score_predictions(truth, pred, prob))
guess, probs, card = results[chosen]

st.subheader(f"{chosen}")
for slot, (name, val) in zip(st.columns(6), card.items()):
    slot.metric(name, "n/a" if val != val else f"{val:.3f}")

left, right = st.columns([1, 1])
with left:
    st.markdown("**Confusion matrix**")
    normalise = st.toggle("Row-normalised view", value=False)
    cm_counts = confusion_matrix(truth, guess, labels=range(len(CLASSES)))
    cm = confusion_matrix(truth, guess, labels=range(len(CLASSES)), normalize="true" if normalise else None)
    fig, ax = plt.subplots(figsize=(5, 4.3))
    ConfusionMatrixDisplay(np.nan_to_num(cm), display_labels=CLASSES).plot(
        ax=ax, cmap="OrRd", colorbar=False, values_format=".2f" if normalise else "d")
    ax.set_xlabel("Predicted outcome")
    ax.set_ylabel("Actual outcome")
    plt.xticks(rotation=20)
    st.pyplot(fig)
    st.caption(f"Confusion-matrix counts total {int(cm_counts.sum())} = rows evaluated.")
with right:
    st.markdown("**All five models on the active data**")
    board = pd.DataFrame({n: r[2] for n, r in results.items()}).T
    shown = board.round(4)
    try:
        st.dataframe(shown.style.highlight_max(axis=0, color="#F6C9CF").format("{:.4f}", na_rep="n/a"))
    except Exception:
        st.dataframe(shown)
    metric_pick = st.selectbox("Chart this metric", list(board.columns), index=4)
    st.bar_chart(board[metric_pick])

t_report, t_roc, t_one = st.tabs(["Classification report", "ROC curves", "Single student"])
with t_report:
    rep = pd.DataFrame(classification_report(truth, guess, labels=range(len(CLASSES)),
                                             target_names=CLASSES, output_dict=True,
                                             zero_division=0)).T.round(3)
    st.dataframe(rep)
    st.bar_chart(rep.loc[CLASSES, ["precision", "recall", "f1-score"]], stack=False)
    out = frame.copy()
    out["predicted_" + LABEL] = [CLASSES[i] for i in guess]
    for j, cname in enumerate(CLASSES):
        out[f"prob_{cname}"] = probs[:, j].round(4)
    st.download_button("⬇ Download predictions", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{LINEUP[chosen]}.csv", mime="text/csv")
with t_roc:
    onehot = label_binarize(truth, classes=list(range(len(CLASSES))))
    fig2, ax2 = plt.subplots(figsize=(6, 4.6))
    drawn = 0
    for j, cname in enumerate(CLASSES):
        if onehot[:, j].sum() == 0 or onehot[:, j].sum() == len(onehot):
            continue
        fpr, tpr, _ = roc_curve(onehot[:, j], probs[:, j])
        ax2.plot(fpr, tpr, color=PALETTE[j % len(PALETTE)], lw=2, label=f"{cname} (AUC {auc(fpr, tpr):.3f})")
        drawn += 1
    if drawn:
        ax2.plot([0, 1], [0, 1], "--", color="grey")
        ax2.set_xlabel("False positive rate")
        ax2.set_ylabel("True positive rate")
        ax2.legend(loc="lower right")
        st.pyplot(fig2)
    else:
        st.info("ROC curves need at least two outcome classes in the data.")
with t_one:
    row = int(st.number_input("Row number in the active data", 0, len(frame) - 1, 0, step=1))
    c1, c2 = st.columns(2)
    c1.metric("Model says", CLASSES[int(guess[row])])
    c2.metric("Actual outcome", CLASSES[int(truth[row])])
    st.bar_chart(pd.Series(probs[row], index=CLASSES, name="probability"))
