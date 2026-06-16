"""
Phase 4: Final Inference and QGIS Mapping

Deploys the best trained 4-class and 7-class Random Forest models onto the
unlabelled inference dataset. Decodes the predictions back to string names,
merges them with the original spatial geometries, and exports a QGIS-ready map.

Usage:
    python predict.py \\
        --input-gpkg "data/zonal_stats.gpkg" \\
        --prep-dir   "phase2_preprocessed_data/" \\
        --model-4    "phase3_model_outputs/4_class_smote_pca" \\
        --model-7    "phase3_model_outputs/7_class_smote_pca" \\
        --outdir     "final_predictions/"
"""

import argparse
import joblib
import warnings
from pathlib import Path

import pandas as pd
import geopandas as gpd
import numpy as np

warnings.filterwarnings("ignore")

def load_artefact(path, artefact_name):
    """Helper to load a pickle file securely."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required artefact: {path} ({artefact_name})")
    return joblib.load(path)

def generate_predictions(model_dir, inference_df, prep_dir, subset_name):
    """
    Dynamically loads the model, applies PCA if required, subsets features, 
    predicts, and decodes the labels.
    """
    model_dir = Path(model_dir)
    prep_dir = Path(prep_dir)
    
    print(f"\n   Processing {subset_name.upper()} Inference...")
    
    # 1. Load Model & Label Encoder
    rf_model = load_artefact(model_dir / "best_rf_model.pkl", "Random Forest Model")
    label_encoder = load_artefact(prep_dir / f"label_encoder_{subset_name}.pkl", "Label Encoder")
    
    X_inf = inference_df.copy()
    
    # 2. Handle Dimensionality Reduction (PCA) if the model folder implies it
    if "pca" in model_dir.name.lower():
        print("      Detected PCA model. Applying PCA transformation...")
        pca_transformer = load_artefact(prep_dir / f"pca_transformer_{subset_name}.pkl", "PCA Transformer")
        
        # PCA needs the exact features it was trained on. 
        # (sklearn >= 1.0 stores feature names in feature_names_in_)
        if hasattr(pca_transformer, "feature_names_in_"):
            expected_features = pca_transformer.feature_names_in_
            X_inf_subset = X_inf[expected_features]
        else:
            raise ValueError("PCA Transformer does not contain feature_names_in_. Check scikit-learn version.")
            
        # Transform features into Principal Components
        X_inf_transformed = pca_transformer.transform(X_inf_subset)
        X_final = pd.DataFrame(X_inf_transformed, columns=[f"PC_{i+1}" for i in range(X_inf_transformed.shape[1])])
        
    else:
        # 3. Handle Standard Features (S, S_T, S_T_G)
        print("      Detected standard feature model. Slicing features...")
        if hasattr(rf_model, "feature_names_in_"):
            expected_features = rf_model.feature_names_in_
            X_final = X_inf[expected_features]
        else:
            raise ValueError("RF Model does not contain feature_names_in_.")

    # 4. Execute Prediction
    print("      Executing Random Forest predictions...")
    numeric_predictions = rf_model.predict(X_final)
    
    # 5. Decode Labels
    string_predictions = label_encoder.inverse_transform(numeric_predictions)
    
    return string_predictions

def run_inference(args):
    prep_dir = Path(args.prep_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("── 1. Loading Datasets ──")
    # Load standardized inference features (from Phase 2)
    inf_parquet_path = prep_dir / "inference_data_scaled.parquet"
    if not inf_parquet_path.exists():
        raise FileNotFoundError(f"Inference data not found at {inf_parquet_path}")
    
    inference_df = pd.read_parquet(inf_parquet_path)
    print(f"   Loaded {len(inference_df)} unlabelled polygons for prediction.")

    # Generate Predictions
    print("\n── 2. Running Inference Engines ──")
    pred_4_class = generate_predictions(args.model_4, inference_df, prep_dir, "4_class")
    pred_7_class = generate_predictions(args.model_7, inference_df, prep_dir, "7_class")

    # Append to the inference dataframe
    inference_df["Predicted_Family_4Class"] = pred_4_class
    inference_df["Predicted_Family_7Class"] = pred_7_class

    # Save a lightweight CSV containing just IDs and predictions
    csv_out = outdir / "inference_predictions_only.csv"
    inference_df[["poly_id", "Predicted_Family_4Class", "Predicted_Family_7Class"]].to_csv(csv_out, index=False)
    print(f"\n   Saved tabular predictions -> {csv_out}")

    print("\n── 3. Merging with Spatial Data for QGIS ──")
    # Load the raw Zonal Statistics to retrieve geometries
    print("   Loading original spatial GeoPackage...")
    raw_gdf = gpd.read_file(args.input_gpkg)
    
    # Replicate the exact 'poly_id' assignment logic from Phase 2
    raw_gdf.insert(0, "poly_id", np.arange(len(raw_gdf), dtype="int64"))
    
    # Isolate the predictions dataframe for merging
    predictions_to_merge = inference_df[["poly_id", "Predicted_Family_4Class", "Predicted_Family_7Class"]]
    
    # Left merge the predictions onto the original GeoDataFrame based on poly_id
    final_gdf = raw_gdf.merge(predictions_to_merge, on="poly_id", how="left")
    
    # Optional: Fill empty prediction columns for the labeled set with their actual ground truth
    mask_labeled = final_gdf["Predicted_Family_7Class"].isna() & final_gdf["Family"].notna()
    final_gdf.loc[mask_labeled, "Predicted_Family_4Class"] = final_gdf.loc[mask_labeled, "Family"]
    final_gdf.loc[mask_labeled, "Predicted_Family_7Class"] = final_gdf.loc[mask_labeled, "Family"]

    # Export final map
    map_out = outdir / "final_predicted_map.gpkg"
    print(f"   Exporting spatial dataset...")
    final_gdf.to_file(map_out, driver="GPKG")
    print(f"\n[SUCCESS] Pipeline Complete. Drag and drop '{map_out.name}' into QGIS to visualize.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RF inference and export QGIS map")
    parser.add_argument("--input-gpkg", required=True, help="Path to original raw Zonal Statistics .gpkg")
    parser.add_argument("--prep-dir", required=True, help="Directory containing Phase 2 preprocessed outputs (encoders, PCA, parquet)")
    parser.add_argument("--model-4", required=True, help="Path to the directory of your BEST 4-class model (Phase 3 output)")
    parser.add_argument("--model-7", required=True, help="Path to the directory of your BEST 7-class model (Phase 3 output)")
    parser.add_argument("--outdir", required=True, help="Directory to save the final QGIS maps and CSVs")
    
    args = parser.parse_args()
    run_inference(args)