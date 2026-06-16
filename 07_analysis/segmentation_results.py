"""
Phase 5: Segmentation Result Evaluation & Chart Generation

Reads the YOLOv11-Seg transfer learning results.csv.
Extracts mask-specific (M) metrics and classification/segmentation losses.
Generates and saves the 2x2 smoothed subplots.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Styling to match typical academic/thesis formats
sns.set_theme(style="whitegrid")

def smooth_curve(scalars, weight=0.6):
    """
    Applies an exponential moving average to replicate the 
    'smooth' dotted lines seen in the thesis plots.
    """
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def generate_loss_charts(df, outdir):
    """
    Generates Figure 24 from the thesis: 
    train/seg_loss, train/cls_loss, val/seg_loss, val/cls_loss over epochs.
    """
    print("Generating Training & Validation Loss charts...")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    loss_metrics = [
        ('train/seg_loss', axes[0, 0]),
        ('train/cls_loss', axes[0, 1]),
        ('val/seg_loss', axes[1, 0]),
        ('val/cls_loss', axes[1, 1])
    ]
    
    epochs = df['epoch']
    
    for metric, ax in loss_metrics:
        if metric not in df.columns:
            print(f"  [WARNING] Metric {metric} not found. Skipping plot.")
            continue
            
        raw_values = df[metric].values
        smoothed_values = smooth_curve(raw_values)
        
        # Plot raw results (solid line with dots)
        ax.plot(epochs, raw_values, marker='o', markersize=4, linestyle='-', 
                linewidth=1.5, label='results', color='#1f77b4')
        # Plot smoothed results (dotted line)
        ax.plot(epochs, smoothed_values, linestyle=':', 
                linewidth=2, label='smooth', color='#ff7f0e')
        
        ax.set_title(metric, fontsize=12, fontweight='bold')
        # Only set legend on the first subplot to match thesis style
        if metric == 'train/seg_loss':
            ax.legend(loc="upper right")
            
    plt.tight_layout()
    out_path = outdir / "Figure_24_Segmentation_Losses.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  -> Saved {out_path.name}")

def generate_metric_charts(df, outdir):
    """
    Generates Figure 25 from the thesis:
    metrics/precision(M), metrics/recall(M), metrics/mAP50(M), metrics/mAP50-95(M).
    """
    print("Generating Precision, Recall, and mAP charts...")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    eval_metrics = [
        ('metrics/precision(M)', axes[0, 0]),
        ('metrics/recall(M)', axes[0, 1]),
        ('metrics/mAP50(M)', axes[1, 0]),
        ('metrics/mAP50-95(M)', axes[1, 1])
    ]
    
    epochs = df['epoch']
    
    for metric, ax in eval_metrics:
        # Fallback to (B) if (M) is missing, but warn the user since thesis mandates masks
        plot_metric = metric
        if metric not in df.columns:
            fallback = metric.replace('(M)', '(B)')
            if fallback in df.columns:
                print(f"  [WARNING] Mask metric {metric} not found. Using Box metric {fallback}.")
                plot_metric = fallback
            else:
                print(f"  [ERROR] Neither {metric} nor {fallback} found. Skipping.")
                continue
                
        raw_values = df[plot_metric].values
        smoothed_values = smooth_curve(raw_values)
        
        # Plot raw results
        ax.plot(epochs, raw_values, marker='o', markersize=4, linestyle='-', 
                linewidth=1.5, color='#1f77b4')
        # Plot smoothed results
        ax.plot(epochs, smoothed_values, linestyle=':', 
                linewidth=2, color='#ff7f0e')
        
        ax.set_title(plot_metric, fontsize=12, fontweight='bold')
            
    plt.tight_layout()
    out_path = outdir / "Figure_25_Segmentation_Metrics.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  -> Saved {out_path.name}")

def evaluate_segmentation(args):
    results_path = Path(args.results_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Cannot find YOLO results file at {results_path}")
        
    print(f"Loading YOLO training logs from {results_path.name}...")
    df = pd.read_csv(results_path)
    
    # Ultralytics results.csv often has leading/trailing whitespaces in column headers
    df.columns = df.columns.str.strip()
    
    # Thesis specifies plotting the transfer learning stage over 15 epochs. 
    # If the file contains more, we limit it to the first 15 to match the document exactly.
    if len(df) >= 15:
        df = df.head(15)
        print("  Isolated the first 15 epochs for transfer learning evaluation.")
    else:
        print(f"  [NOTE] Log contains only {len(df)} epochs. Plotting available data.")
    
    # Check for F1-Score calculation at best epoch (mAP@0.5 maximization)
    if 'metrics/precision(M)' in df.columns and 'metrics/recall(M)' in df.columns:
        best_epoch_idx = df['metrics/mAP50(M)'].idxmax()
        p = df.loc[best_epoch_idx, 'metrics/precision(M)']
        r = df.loc[best_epoch_idx, 'metrics/recall(M)']
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        print(f"\n── Validation Snapshot (Best Epoch: {int(df.loc[best_epoch_idx, 'epoch'])}) ──")
        print(f"   mAP@0.5:       {df.loc[best_epoch_idx, 'metrics/mAP50(M)']:.3f}")
        print(f"   mAP@0.5:0.95:  {df.loc[best_epoch_idx, 'metrics/mAP50-95(M)']:.3f}")
        print(f"   Precision:     {p:.3f}")
        print(f"   Recall:        {r:.3f}")
        print(f"   F1-Score:      {f1:.3f}")
        print("───────────────────────────────────────────────\n")

    generate_loss_charts(df, outdir)
    generate_metric_charts(df, outdir)
    
    print(f"\n[SUCCESS] Segmentation evaluation charts saved to {outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv11-Seg Results and Generate Thesis Charts")
    parser.add_argument("--results-csv", required=True, help="Path to the results.csv generated by YOLO training")
    parser.add_argument("--outdir", required=True, help="Directory to save the generated charts")
    args = parser.parse_args()
    
    evaluate_segmentation(args)