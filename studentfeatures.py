"""Domain features derived from the curricular-unit columns of the student dataset."""
import numpy as np
import pandas as pd

_SEMESTERS = (("1st", "sem1"), ("2nd", "sem2"))


def _find_col(columns, semester, tag):
    for name in columns:
        low = name.lower()
        if f"{semester} sem" in low and f"({tag})" in low:
            return name
    return None


def add_progress_features(frame):
    """Append pass-rate per semester, grade shift and total units passed.
    If the expected columns are absent the frame is returned unchanged."""
    if not isinstance(frame, pd.DataFrame):
        return frame
    out = frame.copy()
    cols = list(out.columns)
    slots = {(s, t): _find_col(cols, s, t) for s, _ in _SEMESTERS
             for t in ("approved", "enrolled", "grade")}
    if any(v is None for v in slots.values()):
        return out
    for sem, short in _SEMESTERS:
        enrolled = out[slots[(sem, "enrolled")]].astype(float)
        passed = out[slots[(sem, "approved")]].astype(float)
        out[f"{short}_pass_rate"] = np.where(enrolled > 0, passed / enrolled.where(enrolled > 0, 1.0), 0.0)
    out["grade_shift"] = out[slots[("2nd", "grade")]].astype(float) - out[slots[("1st", "grade")]].astype(float)
    out["units_passed_total"] = out[slots[("1st", "approved")]].astype(float) + out[slots[("2nd", "approved")]].astype(float)
    return out
