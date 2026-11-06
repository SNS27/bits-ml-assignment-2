"""Data audit: shape, classes, missing values, duplicates. Usage: python tools/check_data.py [csv] [target]"""
import sys
import pandas as pd
path = sys.argv[1] if len(sys.argv) > 1 else "data/full_dataset.csv"
target = sys.argv[2] if len(sys.argv) > 2 else "Target"
df = pd.read_csv(path)
print(f"file: {path}\nrows: {len(df)} | predictors: {df.shape[1] - 1} | target: {target}")
print("class counts:\n", df[target].value_counts().to_string())
print("missing cells:", int(df.isna().sum().sum()))
print("duplicate rows:", int(df.duplicated().sum()))
print("non-numeric predictors:", [c for c in df.drop(columns=target).columns if not pd.api.types.is_numeric_dtype(df[c])])
print("identifier-like columns (all values unique):", [c for c in df.columns if df[c].nunique() == len(df)])
print("note: one row per student, no time order column -> stratified random split is appropriate.")
