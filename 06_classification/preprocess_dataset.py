"""
Prepare the feature GeoPackage for classification.

Steps:
  1. Load zonal_stats.gpkg
  2. Filter: NDVI_Dec2024_mean > 0.3  AND  SAVI_Dec2024_mean > 0.2
  3. Drop rows with > 50 % null values in feature columns
  4. KNN-impute remaining nulls (mean/std/median features only)
  5. Label-encode the 'Family' column; save mapping CSV
  6. Save processed GeoPackage (features are NOT yet scaled — done in training)

Usage:
    python preprocess_dataset.py \\
        --input  data/zonal_stats.gpkg \\
        --output data/preprocessed.gpkg \\
        --label-map data/family_label_mapping.csv
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder


def get_feature_columns(gdf: gpd.GeoDataFrame) -> list:
    """Return all numeric columns that are features (exclude geometry, labels, ids)."""
    exclude = {"geometry", "Family", "Common Nam", "Family_encoded", "fid"}
    return [c for c in gdf.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(gdf[c])]


def preprocess(input_gpkg: str, output_gpkg: str, label_map_csv: str,
               ndvi_col: str = "NDVI_Dec2024_mean",
               savi_col: str = "SAVI_Dec2024_mean"):

    gdf = gpd.read_file(input_gpkg)
    print(f"Initial polygons: {len(gdf)}")

    # ── vegetation filter ───────────────────────────────────────────────────
    if ndvi_col in gdf.columns and savi_col in gdf.columns:
        gdf = gdf[(gdf[ndvi_col] >= 0.3) & (gdf[savi_col] >= 0.2)]
        print(f"After vegetation filter: {len(gdf)}")
    else:
        print(f"Warning: {ndvi_col} or {savi_col} not found — skipping filter")

    feat_cols = get_feature_columns(gdf)
    print(f"Feature columns: {len(feat_cols)}")

    # ── drop rows with > 50 % nulls ────────────────────────────────────────
    null_frac = gdf[feat_cols].isnull().mean(axis=1)
    gdf = gdf[null_frac <= 0.5].copy()
    print(f"After dropping high-null rows: {len(gdf)}")

    # ── KNN imputation on mean/std/median features ─────────────────────────
    impute_cols = [c for c in feat_cols
                   if c.endswith(("_mean", "_std", "_median"))]
    if impute_cols:
        imputer = KNNImputer(n_neighbors=5)
        gdf[impute_cols] = imputer.fit_transform(gdf[impute_cols])
        print(f"KNN-imputed {len(impute_cols)} columns")

    # ── label encoding ─────────────────────────────────────────────────────
    if "Family" in gdf.columns:
        gdf = gdf[gdf["Family"].notna()].copy()
        le = LabelEncoder()
        gdf["Family_encoded"] = le.fit_transform(gdf["Family"])
        mapping = pd.DataFrame({
            "Family":         le.classes_,
            "Family_encoded": le.transform(le.classes_),
        })
        Path(label_map_csv).parent.mkdir(parents=True, exist_ok=True)
        mapping.to_csv(label_map_csv, index=False)
        print(f"Label map saved → {label_map_csv}")
        print(f"Classes: {list(le.classes_)}")
        print(f"Labeled polygons: {len(gdf)}")

    Path(output_gpkg).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_gpkg, driver="GPKG")
    print(f"Saved → {output_gpkg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess feature GeoPackage for RF classification")
    parser.add_argument("--input",     required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--label-map", required=True)
    args = parser.parse_args()

    preprocess(args.input, args.output, args.label_map)
