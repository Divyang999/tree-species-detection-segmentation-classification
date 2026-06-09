"""
Extract per-crown spectral, temporal, and GLCM texture features from
PlanetScope imagery for all tree crown polygons in a GeoPackage/Shapefile.

Feature set (384 total):
  Spectral + Indices zonal stats : 5 months × 14 features × 3 stats = 210
  GLCM texture (Dec 2024 only)   : 2 bands (Red, NIR) × 3 props    =   6
  Temporal differences           : 4 intervals × 14 features × 3    = 168

Usage:
    python zonal_statistics_glcm.py --config config.yaml

See config.yaml for path and date-suffix configuration.
"""

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from joblib import Parallel, delayed
from rasterstats import zonal_stats
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

BAND_NAMES  = ["CoastalBlue", "Blue", "GreenI", "Green",
               "Yellow", "Red", "RedEdge", "NIR"]
INDEX_NAMES = ["NDVI", "NDVI2", "NDVI_RE", "EVI", "SAVI", "GNDVI"]
STATS       = ["mean", "std", "median"]
GLCM_PROPS  = ["contrast", "homogeneity", "correlation"]
GLCM_BANDS  = ["Red", "NIR"]  # band indices in composite (0-based): Red=5, NIR=7
GLCM_BAND_IDX = {"Red": 5, "NIR": 7}


# ── zonal statistics ──────────────────────────────────────────────────────────

def _zonal_stats_one(geometries, raster_path: str, prefix: str) -> pd.DataFrame:
    with rasterio.open(raster_path) as src:
        data      = src.read(1).astype(float)
        transform = src.transform
        nodata    = src.nodata if src.nodata is not None else np.nan
    result = zonal_stats(geometries, data, affine=transform,
                         stats=STATS, nodata=nodata, geojson_out=False)
    df = pd.DataFrame(result)
    df.columns = [f"{prefix}_{s}" for s in STATS]
    return df


def compute_zonal_stats_for_image(geometries, composite_path: str,
                                   index_dir: str, suffix: str,
                                   n_jobs: int = 8) -> pd.DataFrame:
    """Parallel zonal stats for 8 bands + 6 indices for one date."""
    tasks = []

    # Spectral bands from composite
    tasks += [(composite_path, band_idx, f"{name}_{suffix}")
              for band_idx, name in enumerate(BAND_NAMES, start=1)]

    # Vegetation index rasters
    stem = Path(composite_path).stem
    for idx_name in INDEX_NAMES:
        idx_path = str(Path(index_dir) / f"{stem}_{idx_name}.tif")
        tasks.append((idx_path, None, f"{idx_name}_{suffix}"))

    def _run(tif_path, band_idx, prefix):
        with rasterio.open(tif_path) as src:
            data      = src.read(band_idx if band_idx else 1).astype(float)
            transform = src.transform
            nodata    = src.nodata if src.nodata is not None else np.nan
        result = zonal_stats(geometries, data, affine=transform,
                             stats=STATS, nodata=nodata, geojson_out=False)
        df = pd.DataFrame(result)
        df.columns = [f"{prefix}_{s}" for s in STATS]
        return df

    dfs = Parallel(n_jobs=n_jobs)(
        delayed(_run)(tp, bi, pf) for tp, bi, pf in tasks
    )
    return pd.concat(dfs, axis=1)


# ── GLCM features ─────────────────────────────────────────────────────────────

def compute_glcm_for_polygon(poly_geom, raster_data: np.ndarray,
                              transform, band_name: str, suffix: str) -> dict:
    """Compute GLCM contrast / homogeneity / correlation for one polygon."""
    from rasterio.features import geometry_mask
    from rasterio.windows import from_bounds

    bounds = poly_geom.bounds
    win = from_bounds(*bounds, transform=transform)
    row_off = max(0, int(win.row_off))
    col_off = max(0, int(win.col_off))
    h       = max(1, int(win.height))
    w       = max(1, int(win.width))

    crop = raster_data[row_off:row_off+h, col_off:col_off+w]
    if crop.size == 0 or np.all(np.isnan(crop)):
        return {f"{band_name}_{p}_{suffix}": np.nan for p in GLCM_PROPS}

    valid = crop[~np.isnan(crop)]
    mn, mx = valid.min(), valid.max()
    if mx == mn:
        return {f"{band_name}_{p}_{suffix}": np.nan for p in GLCM_PROPS}

    img8 = ((crop - mn) / (mx - mn) * 255).astype(np.uint8)
    img8[np.isnan(crop)] = 0

    glcm = graycomatrix(img8, distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)
    return {
        f"{band_name}_contrast_{suffix}":    float(graycoprops(glcm, "contrast")[0, 0]),
        f"{band_name}_homogeneity_{suffix}": float(graycoprops(glcm, "homogeneity")[0, 0]),
        f"{band_name}_correlation_{suffix}": float(graycoprops(glcm, "correlation")[0, 0]),
    }


def compute_glcm_features(geometries, composite_path: str,
                           suffix: str, n_jobs: int = 8) -> pd.DataFrame:
    """Compute GLCM stats for Red and NIR bands across all polygons."""
    with rasterio.open(composite_path) as src:
        bands     = {b: src.read(GLCM_BAND_IDX[b] + 1).astype(float)
                     for b in GLCM_BANDS}
        transform = src.transform

    records = Parallel(n_jobs=n_jobs)(
        delayed(lambda geom: {
            k: v
            for bn in GLCM_BANDS
            for k, v in compute_glcm_for_polygon(
                geom, bands[bn], transform, bn, suffix).items()
        })(g)
        for g in tqdm(geometries, desc=f"GLCM {suffix}")
    )
    return pd.DataFrame(records)


# ── temporal differences ──────────────────────────────────────────────────────

def compute_temporal_differences(df: pd.DataFrame, suffixes: list) -> pd.DataFrame:
    """
    For consecutive date pairs, compute difference of mean/std/median for all
    spectral bands and indices.
    """
    intervals = list(zip(suffixes[:-1], suffixes[1:]))
    diff_cols = {}
    for s1, s2 in intervals:
        for feat in BAND_NAMES + INDEX_NAMES:
            for stat in STATS:
                c1 = f"{feat}_{s1}_{stat}"
                c2 = f"{feat}_{s2}_{stat}"
                if c1 in df.columns and c2 in df.columns:
                    diff_cols[f"{feat}_{s1}_{s2}_diff_{stat}"] = df[c2] - df[c1]
    return pd.concat([df, pd.DataFrame(diff_cols, index=df.index)], axis=1)


# ── main ──────────────────────────────────────────────────────────────────────

def run(config: dict):
    mask_path  = config["mask_gpkg"]
    out_path   = config["output_gpkg"]
    dates      = config["dates"]       # list of {suffix, composite, index_dir}
    glcm_date  = config["glcm_date"]   # suffix of the date to use for GLCM
    n_jobs     = config.get("n_jobs", 8)

    gdf = gpd.read_file(mask_path)
    print(f"Loaded {len(gdf)} polygons from {mask_path}")

    geometries = list(gdf.geometry)
    all_stats  = []

    for date_cfg in dates:
        suffix    = date_cfg["suffix"]
        composite = date_cfg["composite"]
        idx_dir   = date_cfg["index_dir"]
        print(f"\nComputing zonal stats for {suffix}…")
        stats_df = compute_zonal_stats_for_image(
            geometries, composite, idx_dir, suffix, n_jobs=n_jobs)
        all_stats.append(stats_df)

    feat_df = pd.concat(all_stats, axis=1)

    # GLCM features for the selected date
    glcm_composite = next(
        d["composite"] for d in dates if d["suffix"] == glcm_date)
    print(f"\nComputing GLCM features for {glcm_date}…")
    glcm_df = compute_glcm_features(geometries, glcm_composite,
                                     glcm_date, n_jobs=n_jobs)
    feat_df = pd.concat([feat_df, glcm_df], axis=1)

    # Temporal differences
    suffixes = [d["suffix"] for d in dates]
    feat_df  = compute_temporal_differences(feat_df, suffixes)

    # Merge with polygon GDF
    for col in feat_df.columns:
        gdf[col] = feat_df[col].values

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG")

    feature_cols = [c for c in gdf.columns if c not in ["geometry"]]
    print(f"\nTotal feature columns: {len(feature_cols)}")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute zonal stats + GLCM features")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run(cfg)
