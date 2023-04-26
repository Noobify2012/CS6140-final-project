import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import auc, ConfusionMatrixDisplay, f1_score, precision_recall_curve


img_dir = Path.cwd().parent / "img"
model_dir = img_dir / "model_analysis"
model_dir.mkdir(parents=True, exist_ok=True)


def save_precision_recall_curve(
    model_name: str,
    file_name: str,
    y_test: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray = None,
) -> None:
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    auc_score = auc(recall, precision)
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(int)
    f1 = f1_score(y_test, y_pred)

    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve for {model_name}\n")
    ax.annotate(
        f"AUC ={auc_score:>7.2f}\nF1   ={f1:>7.2f}",
        xy=(30, 30),
        xycoords="axes points",
        size=12,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round", fc="w"),
    )
    plt.savefig(model_dir / file_name)

def save_confusion_matrix(
        model_name: str,
        file_name: str,
        y_test: np.ndarray,
        y_pred: np.ndarray
) -> None:
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.savefig(model_dir/ file_name)

