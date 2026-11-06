"""Write requirements.txt with the exact versions installed in THIS venv (so cloud == local)."""
from importlib.metadata import version
import sys
pkgs = ["streamlit", "scikit-learn", "numpy", "pandas", "matplotlib", "joblib", "ucimlrepo", "tabulate"]
with open("requirements.txt", "w") as f:
    for p in pkgs:
        f.write(f"{p}=={version(p)}\n")
print(open("requirements.txt").read())
print("Python used:", sys.version.split()[0], "-> choose the same minor version in Streamlit Cloud > Advanced settings")
