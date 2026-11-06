"""Six-metric scorecard shared by the trainer and the web app."""
import numpy as np
from sklearn import metrics as skm


def score_predictions(truth, guess, probs):
    k = probs.shape[1]
    avg = "binary" if k == 2 else "weighted"
    try:
        if k == 2:
            area = skm.roc_auc_score(truth, probs[:, 1])
        else:
            area = skm.roc_auc_score(truth, probs, multi_class="ovr", average="weighted",
                                     labels=list(range(k)))
    except ValueError:
        area = float("nan")
    return {
        "Accuracy": skm.accuracy_score(truth, guess),
        "AUC": area,
        "Precision": skm.precision_score(truth, guess, average=avg, zero_division=0),
        "Recall": skm.recall_score(truth, guess, average=avg, zero_division=0),
        "F1": skm.f1_score(truth, guess, average=avg, zero_division=0),
        "MCC": skm.matthews_corrcoef(truth, guess),
    }
