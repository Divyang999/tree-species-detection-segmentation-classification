"""
Train Random Forest classifiers for individual tree species classification.

Runs four feature combinations × two class sets (4-class / 7-class),
with optional SMOTE oversampling and PCA dimensionality reduction.

Feature combinations:
  spectral_only         — 8 bands × 5 months × 3 stats = 120 features
  spectral_temporal     — spectral + temporal differences (278 features)
  spectral_temporal_glcm — + GLCM texture (284 features)
  pca                   — PCA on spectral_temporal_glcm (35 components)

Best result (thesis): 4-class SMOTE + PCA
  Train/Val/Test accuracy: 70.1 / 65.1 / 62.5 %   Cohen's Kappa: 0.80

Usage:
    python train_random_forest.py \\
        --input  data/preprocessed.gpkg \\
        --outdir model_outputs/
"""

import argparse
import json
import warnings
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BAND_NAMES  = ["CoastalBlue", "Blue", "GreenI", "Green",
               "Yellow", "Red", "RedEdge", "NIR"]
INDEX_NAMES = ["NDVI", "NDVI2", "NDVI_RE", "EVI", "SAVI", "GNDVI"]
STATS       = ["mean", "std", "median"]
DATES_5     = ["May2024", "Sept2024", "Dec2024", "Jan2025", "Feb2025"]
GLCM_PROPS  = ["contrast", "homogeneity", "correlation"]
GLCM_BANDS  = ["Red", "NIR"]
GLCM_DATE   = "Dec2024"

# 4-class grouping (Fabaceae, Meliaceae, Arecaceae, Other)
CLASS_4_MAP = {
    "Fabaceae":     "Fabaceae",
    "Meliaceae":    "Meliaceae",
    "Arecaceae":    "Arecaceae",
    "Bignoniaceae": "Other",
    "Combretaceae": "Other",
    "Moraceae":     "Other",
    "Anacardiaceae":"Other",
}

PARAM_GRID = {
    "n_estimators":    [100, 200],
    "max_depth":       [10, 20],
    "max_features":    ["sqrt"],
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [1, 2],
}


def get_spectral_cols(gdf: pd.DataFrame) -> list:
    cols = []
    for feat in BAND_NAMES:
        for suffix in DATES_5:
            for stat in STATS:
                c = f"{feat}_{suffix}_{stat}"
                if c in gdf.columns:
                    cols.append(c)
    return cols


def get_temporal_diff_cols(gdf: pd.DataFrame) -> list:
    intervals = list(zip(DATES_5[:-1], DATES_5[1:]))
    cols = []
    for feat in BAND_NAMES + INDEX_NAMES:
        for s1, s2 in intervals:
            for stat in STATS:
                c = f"{feat}_{s1}_{s2}_diff_{stat}"
                if c in gdf.columns:
                    cols.append(c)
    return cols


def get_index_cols(gdf: pd.DataFrame) -> list:
    cols = []
    for feat in INDEX_NAMES:
        for suffix in DATES_5:
            for stat in STATS:
                c = f"{feat}_{suffix}_{stat}"
                if c in gdf.columns:
                    cols.append(c)
    return cols


def get_glcm_cols(gdf: pd.DataFrame) -> list:
    cols = []
    for band in GLCM_BANDS:
        for prop in GLCM_PROPS:
            c = f"{band}_{prop}_{GLCM_DATE}"
            if c in gdf.columns:
                cols.append(c)
    return cols


def select_features(gdf: pd.DataFrame, combination: str) -> list:
    spectral = get_spectral_cols(gdf) + get_index_cols(gdf)
    temporal = get_temporal_diff_cols(gdf)
    glcm     = get_glcm_cols(gdf)

    if combination == "spectral_only":
        return spectral
    if combination == "spectral_temporal":
        return spectral + temporal
    if combination == "spectral_temporal_glcm":
        return spectral + temporal + glcm
    if combination == "pca":
        return spectral + temporal + glcm  # PCA applied later
    raise ValueError(f"Unknown combination: {combination}")


def train_and_evaluate(gdf: pd.DataFrame, label_col: str,
                       combination: str, use_smote: bool,
                       outdir: Path, dataset_name: str):

    feat_cols = select_features(gdf, combination)
    labeled   = gdf[gdf[label_col].notna()].copy()
    X = labeled[feat_cols].fillna(labeled[feat_cols].mean())
    y = labeled[label_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    if use_smote:
        sm = SMOTE(random_state=42)
        X_train_s, y_train = sm.fit_resample(X_train_s, y_train)
        print(f"  After SMOTE: {len(X_train_s)} train samples")

    if combination == "pca":
        pca = PCA(n_components=35, random_state=42)
        X_train_s = pca.fit_transform(X_train_s)
        X_val_s   = pca.transform(X_val_s)
        X_test_s  = pca.transform(X_test_s)
    else:
        pca = None

    rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    gs = GridSearchCV(rf, PARAM_GRID, cv=5, scoring="f1_weighted",
                      n_jobs=-1, verbose=0)
    gs.fit(X_train_s, y_train)
    best = gs.best_estimator_

    # Metrics
    def metrics(X, y, split):
        yp    = best.predict(X)
        acc   = (yp == y).mean()
        kappa = cohen_kappa_score(y, yp)
        rep   = classification_report(y, yp, output_dict=True)
        f1    = rep["weighted avg"]["f1-score"]
        print(f"  {split}: acc={acc:.3f}  f1={f1:.3f}  kappa={kappa:.3f}")
        return acc, f1, kappa, yp

    train_acc, train_f1, _, _     = metrics(X_train_s, y_train, "Train")
    val_acc,   val_f1,   _, _     = metrics(X_val_s,   y_val,   "Val  ")
    test_acc,  test_f1,  kappa, yp_test = metrics(X_test_s, y_test, "Test ")

    # Save artefacts
    sub = outdir / dataset_name / combination
    sub.mkdir(parents=True, exist_ok=True)

    joblib.dump(best,   sub / f"model_{dataset_name}_{combination}.joblib")
    joblib.dump(scaler, sub / f"scaler_{dataset_name}_{combination}.joblib")
    if pca:
        joblib.dump(pca, sub / f"pca_{dataset_name}_{combination}.joblib")

    with open(sub / f"best_params_{dataset_name}_{combination}.json", "w") as f:
        json.dump(gs.best_params_, f, indent=2)

    # Save test predictions GeoPackage
    test_gdf = labeled.iloc[
        [i for i in range(len(labeled)) if labeled.index[i] in X_test.index]
    ].copy() if False else labeled.loc[X_test.index].copy()
    test_gdf["predicted_Family"] = yp_test
    test_gdf[["geometry", "Family", "predicted_Family"]].to_file(
        str(sub / f"test_predictions_{dataset_name}_{combination}.gpkg"),
        driver="GPKG"
    )

    return {
        "dataset":      dataset_name,
        "combination":  combination,
        "train_acc":    round(train_acc, 4),
        "val_acc":      round(val_acc,   4),
        "test_acc":     round(test_acc,  4),
        "test_f1":      round(test_f1,   4),
        "kappa":        round(kappa,     4),
        "best_params":  gs.best_params_,
    }


def run(input_gpkg: str, outdir: str):
    gdf = gpd.read_file(input_gpkg)
    out = Path(outdir)

    # 4-class mapping
    if "Family" in gdf.columns:
        gdf["Family_4class"] = gdf["Family"].map(CLASS_4_MAP)

    combinations = ["spectral_only", "spectral_temporal",
                    "spectral_temporal_glcm", "pca"]
    datasets = [
        ("7_class",      "Family",        False),
        ("7_class_smote","Family",        True),
        ("4_class",      "Family_4class", False),
        ("4_class_smote","Family_4class", True),
    ]

    all_results = []
    for ds_name, label_col, smote in datasets:
        for comb in combinations:
            print(f"\n── {ds_name} / {comb} ──")
            try:
                result = train_and_evaluate(
                    gdf, label_col, comb, smote, out, ds_name)
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")

    summary = pd.DataFrame(all_results)
    summary.to_csv(out / "all_metrics.csv", index=False)
    print(f"\nAll metrics saved → {out / 'all_metrics.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest tree species classifiers")
    parser.add_argument("--input",  required=True, help="Preprocessed GeoPackage")
    parser.add_argument("--outdir", required=True, help="Output directory for models and results")
    args = parser.parse_args()

    run(args.input, args.outdir)
