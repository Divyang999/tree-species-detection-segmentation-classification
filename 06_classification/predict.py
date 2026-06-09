"""
Apply a trained Random Forest model to classify all tree crown polygons
(including unlabelled ones) and save a full-coverage prediction GeoPackage.

Usage:
    python predict.py \\
        --input    data/preprocessed.gpkg \\
        --model    model_outputs/4_class_smote/pca/model_4_class_smote_pca.joblib \\
        --scaler   model_outputs/4_class_smote/pca/scaler_4_class_smote_pca.joblib \\
        --pca      model_outputs/4_class_smote/pca/pca_4_class_smote_pca.joblib \\
        --output   inference/4_class_smote_pca_predictions.gpkg \\
        --combination pca
"""

import argparse
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd

BAND_NAMES  = ["CoastalBlue", "Blue", "GreenI", "Green",
               "Yellow", "Red", "RedEdge", "NIR"]
INDEX_NAMES = ["NDVI", "NDVI2", "NDVI_RE", "EVI", "SAVI", "GNDVI"]
STATS       = ["mean", "std", "median"]
DATES_5     = ["May2024", "Sept2024", "Dec2024", "Jan2025", "Feb2025"]
GLCM_BANDS  = ["Red", "NIR"]
GLCM_PROPS  = ["contrast", "homogeneity", "correlation"]
GLCM_DATE   = "Dec2024"


def get_feature_columns(gdf: pd.DataFrame, combination: str) -> list:
    spectral = [f"{f}_{s}_{st}" for f in BAND_NAMES + INDEX_NAMES
                for s in DATES_5 for st in STATS if f"{f}_{s}_{st}" in gdf.columns]
    temporal = [c for c in gdf.columns if "_diff_" in c]
    glcm     = [f"{b}_{p}_{GLCM_DATE}" for b in GLCM_BANDS
                for p in GLCM_PROPS if f"{b}_{p}_{GLCM_DATE}" in gdf.columns]

    if combination == "spectral_only":
        return [c for c in spectral
                if not any(c.startswith(idx) for idx in INDEX_NAMES)]
    if combination in ("spectral_temporal", "spectral_temporal_glcm", "pca"):
        base = spectral + temporal
        if combination in ("spectral_temporal_glcm", "pca"):
            base += glcm
        return base
    raise ValueError(f"Unknown combination: {combination}")


def predict(input_gpkg: str, model_path: str, scaler_path: str,
            output_gpkg: str, combination: str, pca_path: str = None):

    gdf  = gpd.read_file(input_gpkg)
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    pca    = joblib.load(pca_path) if pca_path else None

    feat_cols = get_feature_columns(gdf, combination)
    # Use only columns that exist
    feat_cols = [c for c in feat_cols if c in gdf.columns]
    print(f"Using {len(feat_cols)} feature columns")

    X = gdf[feat_cols].copy()
    X.fillna(X.mean(), inplace=True)

    X_s = scaler.transform(X)
    if pca:
        X_s = pca.transform(X_s)

    gdf["predicted_Family"] = model.predict(X_s)
    proba = model.predict_proba(X_s)
    gdf["confidence"] = proba.max(axis=1)

    # Add per-class probability columns
    for i, cls in enumerate(model.classes_):
        gdf[f"prob_{cls}"] = proba[:, i]

    Path(output_gpkg).parent.mkdir(parents=True, exist_ok=True)
    out_cols = ["geometry", "predicted_Family", "confidence"] + \
               [f"prob_{c}" for c in model.classes_]
    if "Family" in gdf.columns:
        out_cols.insert(1, "Family")

    gdf[out_cols].to_file(output_gpkg, driver="GPKG")
    print(f"Predictions saved → {output_gpkg}  ({len(gdf)} polygons)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict tree species for all polygons")
    parser.add_argument("--input",       required=True)
    parser.add_argument("--model",       required=True)
    parser.add_argument("--scaler",      required=True)
    parser.add_argument("--pca",         default=None, help="Optional PCA joblib path")
    parser.add_argument("--output",      required=True)
    parser.add_argument("--combination", required=True,
                        choices=["spectral_only", "spectral_temporal",
                                 "spectral_temporal_glcm", "pca"])
    args = parser.parse_args()

    predict(args.input, args.model, args.scaler, args.output,
            args.combination, pca_path=args.pca)
