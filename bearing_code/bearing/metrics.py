"""Five-class metrics, seed summaries and column-normalized confusion matrices.

Rows are true classes; columns are predicted classes. Diagonal entries of a
column-normalized matrix are class precision, NOT overall accuracy or recall.
"""
from pathlib import Path
import csv
import json
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

LABELS = np.arange(5)
CLASS_NAMES = ["D0 Ball", "D1 Compound", "D2 Healthy", "D3 Inner", "D4 Outer"]


def scores(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("Predictions must have the same nonempty shape as targets.")
    if not np.isin(y_true, LABELS).all() or not np.isin(y_pred, LABELS).all():
        raise ValueError("Class labels must be integer IDs 0 through 4.")
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS,
                                       average="macro", zero_division=0)),
            "n_windows": int(y_true.size)}


def seed_summary(rows):
    """Aggregate per-seed scores, rather than recomputing F1 from pooled labels."""
    if len(rows) < 2:
        raise ValueError("Sample standard deviation requires at least two seeds.")
    out = {"n_seeds": len(rows), "metric_scale": "fraction_0_to_1",
           "standard_deviation": "sample, ddof=1"}
    for key in ("accuracy", "macro_f1"):
        values = np.array([r[key] for r in rows], dtype=float)
        out[key + "_mean"] = float(values.mean())
        out[key + "_std"] = float(values.std(ddof=1))
    return out


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty result table.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def column_normalize(counts):
    counts = np.asarray(counts, dtype=float)
    if counts.shape != (5, 5):
        raise ValueError("Expected a 5 by 5 confusion matrix.")
    denom = counts.sum(axis=0, keepdims=True)
    # A class with no predictions has undefined precision, displayed as zeros.
    return np.divide(counts, denom, out=np.zeros_like(counts), where=denom != 0)


def save_confusion(y_true, y_pred, directory, title, plot=True):
    counts = confusion_matrix(y_true, y_pred, labels=LABELS)
    return save_confusion_counts(counts, directory, title, plot)


def save_confusion_counts(counts, directory, title, plot=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    counts = np.asarray(counts, dtype=np.int64)
    normalized = column_normalize(counts)
    np.savetxt(directory / "counts.csv", counts, delimiter=",", fmt="%d")
    np.savetxt(directory / "column_normalized.csv", normalized, delimiter=",", fmt="%.8f")
    metadata = {"rows": "true labels", "columns": "predicted labels",
                "class_order": CLASS_NAMES, "normalization": "each predicted-class column",
                "diagonal": "per-class precision, not accuracy",
                "empty_predicted_columns": np.flatnonzero(counts.sum(0) == 0).tolist(),
                "empty_column_display": "zeros; precision undefined",
                "title": title}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.4, 6.5))
        im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for i in range(5):
            for j in range(5):
                text = "N/A" if counts[:, j].sum() == 0 else f"{normalized[i,j]:.2f}"
                ax.text(j, i, text, ha="center", va="center",
                        color="white" if normalized[i, j] > .55 else "black")
        ax.set(xticks=range(5), yticks=range(5), xticklabels=CLASS_NAMES,
               yticklabels=CLASS_NAMES, xlabel="Predicted class", ylabel="True class", title=title)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.colorbar(im, ax=ax, label="Proportion within predicted class")
        fig.tight_layout()
        fig.savefig(directory / "column_normalized.png", dpi=220)
        fig.savefig(directory / "column_normalized.pdf")
        plt.close(fig)
    return counts
