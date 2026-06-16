"""
Thesis methodology:
1. Sentinel replacement & QA Abort Gate (Requires exactly 384 features).
2. December NDVI > 0.3 Filtering.
3. Finite Gate (Drop NaN/-9999/inf rows).
4. Inference Separation (Reserved unlabelled polygons).
5. Global Label Encoding & Z-Score Standardization.
6. Post-standardization subsetting (Top-4 and Top-7 Families).
7. Stratified Splitting (70/15/15).
8. Feature Combination Slicing (S, S+T, S+T+G).
9. SMOTE Application (Train only).
10. PCA 95% Variance Application.
"""

import argparse
import sys
import joblib
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ─── CONSTANTS & CONFIGURATION ──────────────────────────────────────────
SENTINELS = [-9999, -9999.0, -999, 9999, np.inf, -np.inf]

NON_FEATURE_TOKENS = {
    "geometry", "family", "family_encoded", "fid", "id", "poly_id",
    "common nam", "common name", "commonname",
    "scientific", "species", "genus", "tree_id", "treeid",
    "confidence", "class_id", "classid", "area", "geo_polygon", "wkt",
}

INDEX_TOKENS    = ["NDVI-RE", "NDVIRE", "NDVI2", "NDVI", "SAVI", "GNDVI", "EVI"]
GLCM_TOKENS     = ["GLCM", "CONTRAST", "HOMOGEN", "CORREL"]
TEMPORAL_TOKENS = ["DIFF", "TEMPORAL", "DELTA", "PERIOD"]

TOP_4_FAMILIES = ["Fabaceae", "Meliaceae", "Arecaceae", "Bignoniaceae"]
TOP_7_FAMILIES = TOP_4_FAMILIES + ["Combretaceae", "Moraceae", "Anacardiaceae"]

EXPECTED_FEATURES = {"band": 120, "index": 90, "glcm": 6, "temporal": 168, "total": 384}


# ─── HELPER FUNCTIONS ───────────────────────────────────────────────────
def get_feature_columns(gdf: gpd.GeoDataFrame) -> list:
    feats = []
    for c in gdf.columns:
        if any(tok in c.lower() for tok in NON_FEATURE_TOKENS): continue
        if c == gdf.geometry.name: continue
        if pd.api.types.is_numeric_dtype(gdf[c]): feats.append(c)
    return feats

def classify_feature(name: str) -> str:
    u = name.upper()
    if any(t in u for t in TEMPORAL_TOKENS): return "temporal"
    if any(t in u for t in GLCM_TOKENS): return "glcm"
    if any(t in u for t in INDEX_TOKENS): return "index"
    return "band"

def group_features(feat_cols: list) -> dict:
    g = {"band": [], "index": [], "glcm": [], "temporal": []}
    for c in feat_cols: g[classify_feature(c)].append(c)
    return g

# ─── ERROR HANDLING & GATES ─────────────────────────────────────────────
def qa_and_group_check(gdf, feat_cols):
    """Hard abort if feature breakdown deviates from the 384-feature layout."""
    print("\n── QA: Feature Layout Check ──")
    groups = group_features(feat_cols)
    detected = {g: len(c) for g, c in groups.items()}
    total = len(feat_cols)

    mism = []
    rows = [("band", detected["band"], EXPECTED_FEATURES["band"]),
            ("index", detected["index"], EXPECTED_FEATURES["index"]),
            ("glcm", detected["glcm"], EXPECTED_FEATURES["glcm"]),
            ("temporal", detected["temporal"], EXPECTED_FEATURES["temporal"]),
            ("TOTAL", total, EXPECTED_FEATURES["total"])]
            
    print(f"   {'Group':9s} {'Detected':>9s} {'Expected':>9s}")
    for name, det, exp in rows:
        flag = "" if det == exp else "  <-- MISMATCH"
        if det != exp: mism.append(name)
        print(f"   {name:9s} {det:9d} {exp:9d}{flag}")

    if mism:
        raise SystemExit(f"\n[ABORT] Feature breakdown mismatch: {mism}. Script aborted to enforce 384-feature requirement.")
    print("   OK: Feature layout matches 384 expectations.")
    return groups

def finite_gate(gdf, feat_cols, rejected_path):
    """Drops rows with NaN, -9999, or inf and warns the user."""
    print("\n── QA: Finite Data Gate ──")
    finite = np.isfinite(gdf[feat_cols].to_numpy(dtype="float64"))
    row_bad = ~finite.all(axis=1)
    n_bad = int(row_bad.sum())
    
    if n_bad == 0:
        print("   OK: All feature values are finite.")
        return gdf

    bad_cols = [c for c, ok in zip(feat_cols, finite.all(axis=0)) if not ok]
    print(f"   [WARNING] {n_bad} row(s) contain NaN/-9999/inf across {len(bad_cols)} columns.")
    
    gdf.loc[row_bad].to_file(rejected_path, driver="GPKG")
    print(f"   Offending rows saved -> {rejected_path}")
    
    kept = gdf.loc[~row_bad].copy()
    print(f"   Dropped {n_bad} row(s); {len(kept)} remain.")
    return kept


# ─── MAIN PIPELINE ──────────────────────────────────────────────────────
def run_pipeline(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("Loading Zonal Statistics...")
    gdf = gpd.read_file(args.input)
    
    # 1. Feature Detection & Sentinel Replacement
    feat_cols = get_feature_columns(gdf)
    gdf[feat_cols] = gdf[feat_cols].replace(SENTINELS, np.nan)
    
    # 2. Strict Layout Check (Abort Gate)
    feat_groups = qa_and_group_check(gdf, feat_cols)
    
    # 3. NDVI Vegetation Filter
    dec_ndvi_cols = [c for c in gdf.columns if "NDVI" in c.upper() and "DEC" in c.upper()]
    if not dec_ndvi_cols:
        raise SystemExit("[ABORT] Could not locate a December NDVI column for filtering.")
    
    ndvi_col = dec_ndvi_cols[0]
    initial_len = len(gdf)
    gdf = gdf[gdf[ndvi_col] > 0.3].copy()
    print(f"\n── Filter: NDVI ──\n   Applied {ndvi_col} > 0.3. Retained {len(gdf)}/{initial_len} samples.")

    # 4. Finite Gate (Data Cleaning)
    gdf = finite_gate(gdf, feat_cols, outdir / "rejected_nonfinite.gpkg")
    
    # Assign unique IDs
    gdf.insert(0, "poly_id", np.arange(len(gdf), dtype="int64"))

    print("\n── Inference Separation & Global Standardization ──")
    is_unlabeled = gdf["Family"].isna() | (gdf["Family"].astype(str).str.strip() == "")
    inference_df = gdf[is_unlabeled].copy()
    labeled_df = gdf[~is_unlabeled].copy()
    
    # Label Encoding (Global on Labeled Data)
    le = LabelEncoder()
    labeled_df["Family_encoded"] = le.fit_transform(labeled_df["Family"])
    joblib.dump(le, outdir / "master_label_encoder.pkl")
    
    # Z-Score Standardization (Fit on Labeled, transform both)
    scaler = StandardScaler()
    labeled_df[feat_cols] = scaler.fit_transform(labeled_df[feat_cols])
    inference_df[feat_cols] = scaler.transform(inference_df[feat_cols])
    joblib.dump(scaler, outdir / "master_scaler.pkl")
    
    # Export Inference Data
    inference_df.to_parquet(outdir / "inference_data_scaled.parquet")
    print(f"   Reserved {len(inference_df)} unlabelled samples for inference.")

    # 5. Subset Generation & Splitting
    combinations = {
        "S": feat_groups["band"] + feat_groups["index"],
        "S_T": feat_groups["band"] + feat_groups["index"] + feat_groups["temporal"],
        "S_T_G": feat_cols  # All features (Spectral + Temporal + GLCM)
    }

    datasets = {"4_class": TOP_4_FAMILIES, "7_class": TOP_7_FAMILIES}

    print("\n── Subset Generation, Splitting, and Class Balancing ──")
    for ds_name, target_families in datasets.items():
        print(f"\nProcessing {ds_name}...")
        
        # Isolate Families Post-Standardization
        df_subset = labeled_df[labeled_df["Family"].isin(target_families)].copy()
        
        # We re-encode per subset so classes are contiguous integers (0-3, 0-6)
        le_subset = LabelEncoder()
        y = le_subset.fit_transform(df_subset["Family"])
        joblib.dump(le_subset, outdir / f"label_encoder_{ds_name}.pkl")
        
        print(f"   Total valid samples for {ds_name}: {len(df_subset)}")
        
        X = df_subset[feat_cols]
        
        # Stratified Split: 70% Train, 15% Val, 15% Test
        X_train_temp, X_test, y_train_temp, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_temp, y_train_temp, test_size=0.17647, stratify=y_train_temp, random_state=42
        )
        
        print(f"   Split sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        for comb_name, cols in combinations.items():
            print(f"   -> Combination: {comb_name} ({len(cols)} features)")
            
            X_train_comb, X_val_comb, X_test_comb = X_train[cols], X_val[cols], X_test[cols]
            
            # A. Export Non-SMOTE
            base_path = outdir / f"{ds_name}_{comb_name}"
            base_path.mkdir(parents=True, exist_ok=True)
            X_train_comb.to_parquet(base_path / "X_train.parquet")
            pd.DataFrame(y_train, columns=["Family_encoded"]).to_parquet(base_path / "y_train.parquet")
            X_val_comb.to_parquet(base_path / "X_val.parquet")
            pd.DataFrame(y_val, columns=["Family_encoded"]).to_parquet(base_path / "y_val.parquet")
            X_test_comb.to_parquet(base_path / "X_test.parquet")
            pd.DataFrame(y_test, columns=["Family_encoded"]).to_parquet(base_path / "y_test.parquet")

            # B. Apply SMOTE (Train only) & Export
            smote = SMOTE(random_state=42)
            X_train_sm, y_train_sm = smote.fit_resample(X_train_comb, y_train)
            
            smote_path = outdir / f"{ds_name}_smote_{comb_name}"
            smote_path.mkdir(parents=True, exist_ok=True)
            X_train_sm.to_parquet(smote_path / "X_train.parquet")
            pd.DataFrame(y_train_sm, columns=["Family_encoded"]).to_parquet(smote_path / "y_train.parquet")
            X_val_comb.to_parquet(smote_path / "X_val.parquet")
            pd.DataFrame(y_val, columns=["Family_encoded"]).to_parquet(smote_path / "y_val.parquet")
            X_test_comb.to_parquet(smote_path / "X_test.parquet")
            pd.DataFrame(y_test, columns=["Family_encoded"]).to_parquet(smote_path / "y_test.parquet")

            # C. PCA 95% Variance Application (S_T_G / full feature set only, per thesis)
            if comb_name == "S_T_G":
                pca = PCA(n_components=0.95, random_state=42)
                X_train_pca = pd.DataFrame(pca.fit_transform(X_train_sm))
                X_val_pca = pd.DataFrame(pca.transform(X_val_comb))
                X_test_pca = pd.DataFrame(pca.transform(X_test_comb))
                
                print(f"      PCA reduced dimensions to {pca.n_components_} components.")
                joblib.dump(pca, outdir / f"pca_transformer_{ds_name}.pkl")
                
                pca_path = outdir / f"{ds_name}_smote_pca"
                pca_path.mkdir(parents=True, exist_ok=True)
                X_train_pca.to_parquet(pca_path / "X_train.parquet")
                pd.DataFrame(y_train_sm, columns=["Family_encoded"]).to_parquet(pca_path / "y_train.parquet")
                X_val_pca.to_parquet(pca_path / "X_val.parquet")
                pd.DataFrame(y_val, columns=["Family_encoded"]).to_parquet(pca_path / "y_val.parquet")
                X_test_pca.to_parquet(pca_path / "X_test.parquet")
                pd.DataFrame(y_test, columns=["Family_encoded"]).to_parquet(pca_path / "y_test.parquet")

    print("\n[SUCCESS] Pipeline Complete. Data is fully preprocessed and ready for Random Forest.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Preprocessing & Data Splitting Pipeline")
    parser.add_argument("--input", required=True, help="Input raw Zonal Statistics GeoPackage")
    parser.add_argument("--outdir", required=True, help="Directory to save output parquets and models")
    args = parser.parse_args()
    
    run_pipeline(args)