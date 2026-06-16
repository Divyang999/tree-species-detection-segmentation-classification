"""
Phase 3: Random Forest Training & Hyperparameter Tuning

Reads the preprocessed .parquet datasets, performs Grid Search hyperparameter
tuning using Stratified K-Fold (K=5) cross-validation on the TRAINING split,
evaluates on the held-out Validation and Test splits, and exports all models,
metrics, and feature importances for analysis.

Thesis methodology (section 3.5.3) implemented here:
  - Stratified K-Fold (default K=5) cross-validation with hyperparameter grid
    search. CV folds are drawn from the training split only; the 15%
    validation split is a true hold-out used purely for reporting (Table 12's
    "Validation Accuracy"/"Validation F1-Score" columns), not for tuning.
  - Grid search ranges: n_estimators 100-500, max_depth 10-30,
    min_samples_split 5-10, min_samples_leaf 1-4.
  - RF `class_weight='balanced'` is applied ONLY to non-SMOTE subsets
    (SMOTE already rebalances the training distribution, so combining both
    would double-compensate for minority classes).
  - A feature-importance threshold of 0.8 cumulative (mean decrease in Gini
    impurity) is applied to the non-PCA combinations (S, S_T, S_T_G) as an
    alternative dimensionality-reduction strategy to compare against PCA:
    the full-feature model's importances are ranked descending, and the
    smallest prefix whose cumulative importance reaches 0.8 is kept; the
    final model is refit on just those columns. The PCA combination is left
    untouched (PCA is itself the reduction step for that combination).
"""

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, cohen_kappa_score,
    accuracy_score, f1_score,
)

warnings.filterwarnings("ignore")

# Thesis-dictated Hyperparameter Grid (section 3.5.3)
PARAM_GRID = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 20, 30],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [1, 2, 4]
}

N_SPLITS = 5                      # Stratified K-Fold, thesis default
FEATURE_IMPORTANCE_THRESHOLD = 0.8  # cumulative Gini-importance cutoff


def load_data(folder_path):
    """Loads Train, Val, and Test data from a given combination directory."""
    X_train = pd.read_parquet(folder_path / "X_train.parquet")
    y_train = pd.read_parquet(folder_path / "y_train.parquet").values.ravel()

    X_val = pd.read_parquet(folder_path / "X_val.parquet")
    y_val = pd.read_parquet(folder_path / "y_val.parquet").values.ravel()

    X_test = pd.read_parquet(folder_path / "X_test.parquet")
    y_test = pd.read_parquet(folder_path / "y_test.parquet").values.ravel()

    return X_train, y_train, X_val, y_val, X_test, y_test


def is_smote_dataset(dataset_name: str) -> bool:
    return "smote" in dataset_name.lower()


def is_pca_combination(dataset_name: str) -> bool:
    return dataset_name.lower().endswith("pca")


def select_features_by_importance(model, X_train, threshold: float):
    """Smallest prefix of features (ranked by importance) whose cumulative
    importance reaches `threshold`. Returns the selected column list."""
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    ranked = importances.sort_values(ascending=False)
    cum = ranked.cumsum() / ranked.sum()
    n_keep = int(np.searchsorted(cum.values, threshold) + 1)
    n_keep = max(1, min(n_keep, len(ranked)))
    return list(ranked.index[:n_keep])


def fit_grid_search(X_train, y_train, class_weight):
    rf = RandomForestClassifier(random_state=42, n_jobs=1, class_weight=class_weight)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=PARAM_GRID,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def evaluate_split(model, X, y):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average='macro')
    return acc, f1_macro, y_pred


def train_and_evaluate(input_dir, outdir):
    input_path = Path(input_dir)
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    # Locate all subset directories (e.g., 4_class_S, 7_class_smote_S_T_G)
    combinations = [d for d in input_path.iterdir() if d.is_dir()]

    if not combinations:
        raise ValueError(f"No valid dataset directories found in {input_dir}")

    # Load master encoders to recover actual class names for reporting
    encoders = {}
    for p in input_path.glob("label_encoder_*.pkl"):
        # e.g., label_encoder_4_class.pkl -> '4_class'
        subset_name = p.stem.replace("label_encoder_", "")
        encoders[subset_name] = joblib.load(p)

    all_metrics = []

    for comb_path in sorted(combinations):
        dataset_name = comb_path.name
        print(f"\n{'='*60}\nTraining Model for: {dataset_name}\n{'='*60}")

        # Determine which encoder to use based on the folder name
        subset_key = "4_class" if "4_class" in dataset_name else "7_class"
        le = encoders.get(subset_key)
        target_names = le.classes_ if le else None

        # 1. Load Parquet Data
        try:
            X_train, y_train, X_val, y_val, X_test, y_test = load_data(comb_path)
        except Exception as e:
            print(f"  [SKIP] Missing data files in {dataset_name}: {e}")
            continue

        print(f"   Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")

        # 2. class_weight='balanced' only for non-SMOTE subsets
        smote_dataset = is_smote_dataset(dataset_name)
        class_weight = None if smote_dataset else 'balanced'
        print(f"   SMOTE dataset: {smote_dataset} | class_weight: {class_weight}")

        # 3. Stratified K-Fold (K=5) Grid Search on the TRAINING split only.
        print(f"   Running Grid Search (StratifiedKFold, K={N_SPLITS})...")
        grid_search = fit_grid_search(X_train, y_train, class_weight)
        model = grid_search.best_estimator_
        print(f"   Best Params: {grid_search.best_params_}")

        # 4. Feature-importance threshold (0.8 cumulative) for non-PCA combos,
        #    as an alternative dimensionality-reduction strategy to compare
        #    against the PCA combination.
        selected_features = list(X_train.columns)
        if not is_pca_combination(dataset_name):
            selected_features = select_features_by_importance(
                model, X_train, FEATURE_IMPORTANCE_THRESHOLD)
            print(f"   Feature importance threshold ({FEATURE_IMPORTANCE_THRESHOLD}): "
                  f"kept {len(selected_features)}/{len(X_train.columns)} features")
            if len(selected_features) < len(X_train.columns):
                grid_search = fit_grid_search(X_train[selected_features], y_train, class_weight)
                model = grid_search.best_estimator_
                print(f"   Refit best params on reduced features: {grid_search.best_params_}")

        X_train_f = X_train[selected_features]
        X_val_f = X_val[selected_features]
        X_test_f = X_test[selected_features]

        # 5. Evaluate on Train / Validation / Test
        train_acc, train_f1, _ = evaluate_split(model, X_train_f, y_train)
        val_acc, val_f1, _ = evaluate_split(model, X_val_f, y_val)
        test_acc, test_f1, y_pred_test = evaluate_split(model, X_test_f, y_test)
        kappa = cohen_kappa_score(y_test, y_pred_test)

        print(f"   Train Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
              f"Val Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
              f"Test Acc: {test_acc:.4f} F1: {test_f1:.4f} | Kappa: {kappa:.4f}")

        # 6. Save Artifacts
        model_out = outdir_path / dataset_name
        model_out.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, model_out / "best_rf_model.pkl")
        with open(model_out / "best_params.json", "w") as f:
            json.dump(grid_search.best_params_, f, indent=4)
        with open(model_out / "selected_features.json", "w") as f:
            json.dump(selected_features, f, indent=2)

        # Save Classification Report (test set)
        report_dict = classification_report(y_test, y_pred_test, target_names=target_names, output_dict=True)
        pd.DataFrame(report_dict).transpose().to_csv(model_out / "classification_report.csv")

        # Save Confusion Matrix (test set)
        cm = confusion_matrix(y_test, y_pred_test)
        cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
        cm_df.to_csv(model_out / "confusion_matrix.csv")

        # Save Feature Importances (for the final, possibly-reduced model)
        importances = model.feature_importances_
        feature_names = X_train_f.columns if isinstance(X_train_f.columns[0], str) else [f"PC_{i+1}" for i in range(len(importances))]
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        fi_df.to_csv(model_out / "feature_importances.csv", index=False)

        # Append to master metrics tracker (Table 12 columns)
        all_metrics.append({
            "Dataset_Combination": dataset_name,
            "Training_F1_Score": round(train_f1, 4),
            "Validation_Accuracy": round(val_acc, 4),
            "Validation_F1_Score": round(val_f1, 4),
            "Test_Accuracy": round(test_acc, 4),
            "Test_F1_Score": round(test_f1, 4),
            "Cohen_Kappa": round(kappa, 4),
            "N_Features_Used": len(selected_features),
            "Best_n_estimators": grid_search.best_params_['n_estimators'],
            "Best_max_depth": grid_search.best_params_['max_depth'],
            "Best_min_samples_split": grid_search.best_params_['min_samples_split'],
            "Best_min_samples_leaf": grid_search.best_params_['min_samples_leaf']
        })

    # 7. Generate Master Summary
    summary_df = pd.DataFrame(all_metrics).sort_values(by="Test_Accuracy", ascending=False)
    summary_df.to_csv(outdir_path / "master_metrics_summary.csv", index=False)

    print(f"\n{'='*60}\nTRAINING COMPLETE. MASTER SUMMARY:\n{'='*60}")
    print(summary_df.to_string(index=False))
    print(f"\nAll models and evaluation artefacts saved to: {outdir_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest with Grid Search Tuning")
    parser.add_argument("--input", required=True, help="Directory containing preprocessed subsets (output from Phase 2)")
    parser.add_argument("--outdir", required=True, help="Directory to save trained models and evaluation metrics")
    args = parser.parse_args()

    train_and_evaluate(args.input, args.outdir)
