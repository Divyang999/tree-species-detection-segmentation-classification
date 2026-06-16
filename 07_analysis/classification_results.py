"""
Phase 6: Classification Result Evaluation & Chart Generation

Reads the outputs from Phase 3 (Random Forest Training).
Generates Master Metric comparisons, Confusion Matrices for the best models,
Feature Importance rankings, and F1-score charts (overall + per-class,
thesis Figures 31-32) for thesis inclusion.

Usage:
    python classification_results.py \\
        --input-dir "phase3_model_outputs/" \\
        --outdir "thesis_figures/classification/"
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Set publication-quality styling
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def plot_metrics_comparison(summary_df, outdir):
    """
    Generates a grouped bar chart comparing Test Accuracy and Cohen's Kappa
    across all tested combinations.
    """
    print("Generating Model Comparison Chart...")
    
    # Sort for visual hierarchy
    summary_df = summary_df.sort_values(by="Test_Accuracy", ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y = np.arange(len(summary_df))
    height = 0.35
    
    # Plot Accuracy and Kappa side-by-side horizontally
    has_f1 = 'Test_F1_Score' in summary_df.columns
    n_bars = 3 if has_f1 else 2
    width = 0.8 / n_bars

    rects1 = ax.barh(y - width, summary_df['Test_Accuracy'], width,
                     label='Test Accuracy', color='#2ca02c')
    rects2 = ax.barh(y, summary_df['Cohen_Kappa'], width,
                     label="Cohen's Kappa", color='#1f77b4')
    bars_for_labels = [rects1, rects2]
    if has_f1:
        rects3 = ax.barh(y + width, summary_df['Test_F1_Score'], width,
                         label='Test F1-Score', color='#ff7f0e')
        bars_for_labels.append(rects3)

    ax.set_xlabel('Score')
    ax.set_title('Random Forest Performance by Feature Combination', pad=20, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(summary_df['Dataset_Combination'])
    ax.legend(loc='lower right')

    # Add exact text labels to the ends of the bars
    for rects in bars_for_labels:
        ax.bar_label(rects, padding=3, fmt='%.3f')

    # Set limit to 1.05 to leave room for the text labels
    ax.set_xlim(0, 1.1)

    plt.tight_layout()
    out_path = outdir / "Classification_Metric_Comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {out_path.name}")


def plot_overall_f1(summary_df, outdir):
    """Thesis Figure 31: overall (macro) test F1-score for every combination,
    grouped by SMOTE / non-SMOTE (matches Fig 31's two-colour bar grouping)."""
    if 'Test_F1_Score' not in summary_df.columns:
        print("  [WARNING] No Test_F1_Score column found; skipping overall F1 chart.")
        return

    print("Generating Overall F1-Score Chart...")
    df = summary_df.copy()
    df['is_smote'] = df['Dataset_Combination'].str.contains('smote', case=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = df['is_smote'].map({True: '#ff7f0e', False: '#1f77b4'})
    ax.bar(df['Dataset_Combination'], df['Test_F1_Score'], color=colors)
    ax.set_ylabel('Test F1-Score (macro)')
    ax.set_title('Overall F1-Score for All Combinations', pad=20, fontweight='bold')
    plt.xticks(rotation=45, ha='right')

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#1f77b4', label='non-SMOTE'),
                       Patch(color='#ff7f0e', label='SMOTE')])

    plt.tight_layout()
    out_path = outdir / "Overall_F1_Score.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {out_path.name}")


def plot_per_class_f1(model_dir, dataset_name, outdir):
    """Thesis Figure 32: per-family F1-score for one (best) model, read from
    that model's classification_report.csv."""
    report_path = model_dir / "classification_report.csv"
    if not report_path.exists():
        print(f"  [WARNING] No classification_report.csv found in {model_dir.name}")
        return

    print(f"Generating Per-Class F1-Score chart for {dataset_name}...")
    report_df = pd.read_csv(report_path, index_col=0)
    # Drop the aggregate rows (accuracy / macro avg / weighted avg) added by
    # sklearn's classification_report, keep only per-class rows.
    class_rows = report_df.drop(
        index=[i for i in ("accuracy", "macro avg", "weighted avg") if i in report_df.index]
    )
    if "f1-score" not in class_rows.columns or class_rows.empty:
        print(f"  [WARNING] No per-class f1-score rows in {report_path}")
        return

    plt.figure(figsize=(8, 6))
    plt.bar(class_rows.index, class_rows["f1-score"], color='#2ca02c')
    plt.ylabel('Test F1-Score')
    plt.ylim(0, 1.0)
    plt.title(f'Per-Class F1-Score: {dataset_name}', pad=20, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    out_path = outdir / f"Per_Class_F1_{dataset_name}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {out_path.name}")

def plot_confusion_matrix(model_dir, dataset_name, outdir):
    """
    Generates a styled heatmap for the confusion matrix.
    """
    cm_path = model_dir / "confusion_matrix.csv"
    if not cm_path.exists():
        print(f"  [WARNING] No confusion matrix found in {model_dir.name}")
        return
        
    print(f"Generating Confusion Matrix for {dataset_name}...")
    cm_df = pd.read_csv(cm_path, index_col=0)
    
    plt.figure(figsize=(8, 6))
    
    # Draw heatmap with integer formatting and a clean colormap
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True,
                square=True, linewidths=.5, annot_kws={"size": 12})
                
    plt.title(f'Confusion Matrix: {dataset_name}', pad=20, fontweight='bold')
    plt.ylabel('True Family')
    plt.xlabel('Predicted Family')
    
    # Ensure labels aren't cut off
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    out_path = outdir / f"Confusion_Matrix_{dataset_name}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {out_path.name}")

def plot_feature_importances(model_dir, dataset_name, outdir, top_n=20):
    """
    Generates a horizontal bar chart of the Top N most important features.
    """
    fi_path = model_dir / "feature_importances.csv"
    if not fi_path.exists():
        print(f"  [WARNING] No feature importances found in {model_dir.name}")
        return
        
    print(f"Generating Feature Importance chart for {dataset_name}...")
    fi_df = pd.read_csv(fi_path)
    
    # Take only the top N for readability
    top_fi = fi_df.head(top_n).sort_values(by="Importance", ascending=True)
    
    plt.figure(figsize=(10, 8))
    
    bars = plt.barh(top_fi['Feature'], top_fi['Importance'], color='#ff7f0e')
    
    plt.xlabel('Mean Decrease in Impurity (Importance)')
    plt.title(f'Top {min(top_n, len(fi_df))} Feature Importances: {dataset_name}', 
              pad=20, fontweight='bold')
              
    plt.tight_layout()
    out_path = outdir / f"Feature_Importances_{dataset_name}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {out_path.name}")

def evaluate_classification(args):
    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load the Master Summary generated by Phase 3
    summary_path = input_dir / "master_metrics_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Cannot find master_metrics_summary.csv in {input_dir}")
        
    summary_df = pd.read_csv(summary_path)
    print(f"Loaded Master Metrics for {len(summary_df)} models.")
    
    # Generate the global comparison charts
    plot_metrics_comparison(summary_df, outdir)
    plot_overall_f1(summary_df, outdir)
    
    # 2. Identify the absolute BEST models for 4-class and 7-class subsets
    print("\n── Locating Best Models for Detail Analysis ──")
    
    # Filter subsets
    df_4 = summary_df[summary_df['Dataset_Combination'].str.contains('4_class')]
    df_7 = summary_df[summary_df['Dataset_Combination'].str.contains('7_class')]
    
    best_models = []
    if not df_4.empty:
        best_4 = df_4.iloc[0]['Dataset_Combination'] # Already sorted descending in Phase 3
        best_models.append(best_4)
        print(f"   Best 4-Class Model: {best_4}")
        
    if not df_7.empty:
        best_7 = df_7.iloc[0]['Dataset_Combination']
        best_models.append(best_7)
        print(f"   Best 7-Class Model: {best_7}")

    # 3. Generate detailed artifacts (CM & Feature Importances) for the winners
    print("\n── Generating Detailed Artifacts ──")
    for model_name in best_models:
        model_dir = input_dir / model_name
        if not model_dir.exists():
            print(f"  [ERROR] Directory {model_dir} missing. Skipping detail charts.")
            continue
            
        plot_confusion_matrix(model_dir, model_name, outdir)
        plot_per_class_f1(model_dir, model_name, outdir)

        # If the best model used PCA, the features will just be PC_1, PC_2.
        plot_feature_importances(model_dir, model_name, outdir, top_n=20)
        
    print(f"\n[SUCCESS] All Classification evaluation charts exported to {outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RF Classification and Generate Thesis Charts")
    parser.add_argument("--input-dir", required=True, help="Directory containing Phase 3 outputs (must contain master_metrics_summary.csv and model folders)")
    parser.add_argument("--outdir", required=True, help="Directory to save the generated charts")
    args = parser.parse_args()

    evaluate_classification(args)