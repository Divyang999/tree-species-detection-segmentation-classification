"""
Visualise classification results:
  - Confusion matrices (per dataset × combination)
  - F1-score comparison bar charts (4-class and 7-class)
  - Feature importance plots
  - PCA vs SMOTE comparison

Usage:
    python visualize_results.py \\
        --metrics  model_outputs/all_metrics.csv \\
        --gpkg     model_outputs/4_class_smote/pca/test_predictions_4_class_smote_pca.gpkg \\
        --model    model_outputs/4_class_smote/spectral_temporal_glcm/model_4_class_smote_spectral_temporal_glcm.joblib \\
        --outdir   results/plots/
"""

import argparse
from pathlib import Path

import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

plt.rcParams.update({"figure.dpi": 150, "font.size": 11})


# ── confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(gpkg_path: str, outdir: Path, label: str):
    gdf = gpd.read_file(gpkg_path)
    if "Family" not in gdf.columns or "predicted_Family" not in gdf.columns:
        print(f"  Skipping confusion matrix — missing columns in {gpkg_path}")
        return

    classes = sorted(gdf["Family"].dropna().unique())
    cm = confusion_matrix(gdf["Family"], gdf["predicted_Family"], labels=classes)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted Family")
    ax.set_ylabel("True Family")
    ax.set_title(f"Confusion Matrix — {label}")
    plt.tight_layout()
    out = outdir / f"cm_{label}.png"
    fig.savefig(out)
    plt.close()
    print(f"  Saved {out.name}")


# ── F1 comparison bar charts ──────────────────────────────────────────────────

def plot_f1_comparison(metrics_csv: str, n_classes: int, outdir: Path):
    df = pd.read_csv(metrics_csv)
    tag = f"{n_classes}_class"
    subset = df[df["dataset"].str.startswith(tag)].copy()
    if subset.empty:
        print(f"  No data for {tag}")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x  = np.arange(len(subset["combination"].unique()))
    w  = 0.35
    datasets = subset["dataset"].unique()
    offsets  = np.linspace(-w * (len(datasets)-1)/2, w * (len(datasets)-1)/2,
                            len(datasets))

    for ds, offset in zip(datasets, offsets):
        vals = subset[subset["dataset"] == ds].set_index("combination")["test_f1"]
        combs = sorted(vals.index)
        ax.bar(x + offset, [vals.get(c, 0) for c in combs], w, label=ds)

    ax.set_xticks(x)
    ax.set_xticklabels(sorted(subset["combination"].unique()), rotation=15)
    ax.set_ylabel("Test F1 (weighted)")
    ax.set_title(f"{n_classes}-Class F1 Comparison")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    out = outdir / f"{n_classes}_species_f1_comparison.png"
    fig.savefig(out)
    plt.close()
    print(f"  Saved {out.name}")


# ── feature importance ────────────────────────────────────────────────────────

def plot_feature_importance(model_path: str, feat_cols: list,
                             outdir: Path, label: str, top_n: int = 20):
    model = joblib.load(model_path)
    if not hasattr(model, "feature_importances_"):
        print("  Model has no feature_importances_")
        return

    importances = pd.Series(model.feature_importances_, index=feat_cols)
    top = importances.nlargest(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    top.sort_values().plot.barh(ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances — {label}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    out = outdir / f"feature_importance_{label}.png"
    fig.savefig(out)
    plt.close()
    print(f"  Saved {out.name}")


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary_table(metrics_csv: str):
    df = pd.read_csv(metrics_csv)
    cols = ["dataset", "combination", "train_acc", "val_acc", "test_acc",
            "test_f1", "kappa"]
    available = [c for c in cols if c in df.columns]
    print("\n── Classification Results ──")
    print(df[available].sort_values(["dataset", "combination"]).to_string(index=False))


# ── main ──────────────────────────────────────────────────────────────────────

def run(metrics_csv: str, gpkg: str, model_path: str, outdir: str):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    if metrics_csv and Path(metrics_csv).exists():
        print_summary_table(metrics_csv)
        plot_f1_comparison(metrics_csv, 4, out)
        plot_f1_comparison(metrics_csv, 7, out)

    if gpkg and Path(gpkg).exists():
        label = Path(gpkg).stem
        plot_confusion_matrix(gpkg, out, label)

    if model_path and Path(model_path).exists():
        # Feature columns would need to match the training run
        print("  Feature importance plot skipped — pass feat_cols explicitly if needed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise RF classification results")
    parser.add_argument("--metrics", default=None, help="all_metrics.csv path")
    parser.add_argument("--gpkg",    default=None, help="Test prediction GeoPackage")
    parser.add_argument("--model",   default=None, help="Trained RF model (.joblib)")
    parser.add_argument("--outdir",  required=True, help="Output directory for plots")
    args = parser.parse_args()

    run(args.metrics, args.gpkg, args.model, args.outdir)
