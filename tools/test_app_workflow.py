"""Headless evaluator-workflow test for app.py (Streamlit AppTest). Run from the project root."""
import io, json, re, sys
from pathlib import Path
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
META = json.loads((ROOT / "model/run_meta.json").read_text())
REF = json.loads((ROOT / "model/metrics_full.json").read_text())
SAMPLE = pd.read_csv(ROOT / "test_data.csv")
MODELS = list(REF)
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

HARNESS = '''
import io, sys, streamlit as st
sys.path.insert(0, r"{root}")
class _Up(io.BytesIO):
    name = "{name}"
_blob = open(r"{blob}", "rb").read() if {use} else None
st.file_uploader = lambda *a, **k: (_Up(_blob) if _blob is not None else None)
__file__ = r"{root}/app.py"
exec(open(__file__).read())
'''
results = []


def record(name, ok, detail=""):
    results.append((name, ok, detail)); print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def launch(df=None, raw=None, name="upload.csv"):
    blob = ROOT / "tmp_upload.csv"
    use = df is not None or raw is not None
    if use:
        blob.write_bytes(raw if raw is not None else df.to_csv(index=False).encode())
    code = HARNESS.format(root=str(ROOT).replace("\\", "/"), name=name, blob=str(blob), use=use)
    at = AppTest.from_string(code, default_timeout=180).run()
    return at


def shown_metrics(at):
    return [m.value for m in at.metric[:6]]


def matches_ref(at, model):
    vals = shown_metrics(at)
    return all(v == "n/a" or abs(float(v) - REF[model][k]) <= 6e-4 for v, k in zip(vals, METRICS))


def info_text(at):
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])


def rows_shown(at):
    m = re.search(r"Rows evaluated:\*\*\s*(\d+)", info_text(at)); return int(m.group(1)) if m else None


def cm_total(at):
    m = re.search(r"counts total (\d+)", info_text(at)); return int(m.group(1)) if m else None


try:
    at = launch()
    record("default load, no exceptions", not at.exception and not at.error)
    record("default rows == test_data rows", rows_shown(at) == len(SAMPLE), f"rows={rows_shown(at)}")
    for mdl in MODELS:
        at.sidebar.selectbox[0].set_value(mdl).run()
        record(f"select '{mdl}': metrics match experiment", not at.exception and matches_ref(at, mdl), str(shown_metrics(at)))
    # re-upload of the exact sample
    at = launch(df=SAMPLE); mdl = MODELS[0]
    record("re-upload sample: same metrics", matches_ref(at, mdl) and rows_shown(at) == len(SAMPLE))
    # subset
    sub = SAMPLE.sample(300, random_state=1)
    at = launch(df=sub)
    record("subset: row count + confusion total", rows_shown(at) == 300 and cm_total(at) == 300, f"rows={rows_shown(at)} cm={cm_total(at)}")
    # reordered columns
    cols = list(SAMPLE.columns); at = launch(df=SAMPLE[cols[::-1]])
    record("reordered columns: same metrics", matches_ref(at, mdl))
    # single class
    cls = META["classes"][0]; single = SAMPLE[SAMPLE[META["target"]] == cls]
    at = launch(df=single)
    record("single-class: no crash, AUC n/a", not at.exception and shown_metrics(at)[1] == "n/a" and len(at.warning) >= 1, str(shown_metrics(at)))
    # invalid uploads
    record("missing column -> visible error", bool(launch(df=SAMPLE.drop(columns=[META["raw_features"][0]])).error))
    bad = SAMPLE.copy(); bad[META["target"]] = "Banana"
    record("invalid labels -> visible error", bool(launch(df=bad).error))
    record("empty file -> visible error", bool(launch(raw=b"").error))
    record("headers only -> visible error", bool(launch(raw=(",".join(SAMPLE.columns) + "\n").encode()).error))
    txt = SAMPLE.copy().astype(object); txt.iloc[0, 0] = "abc"
    record("non-numeric value -> visible error", bool(launch(df=txt).error))
finally:
    (ROOT / "tmp_upload.csv").unlink(missing_ok=True)

fails = [r for r in results if not r[1]]
print(f"\n{len(results) - len(fails)}/{len(results)} checks passed")
print("MANUAL BROWSER CHECKS STILL NEEDED: layout/appearance, file-picker click, download buttons, signed-out cloud access.")
sys.exit(1 if fails else 0)
