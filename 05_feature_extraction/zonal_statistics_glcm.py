"""
Extract per-crown spectral, temporal, and GLCM texture features from
PlanetScope imagery for all tree-crown polygons in a GeoPackage.

Feature set (384 NEW columns):
  Spectral + Indices zonal stats : 5 months x 14 features x 3 stats = 210
  GLCM texture (Dec 2024, RAW)   : 2 bands (Red, NIR) x 3 props      =   6
  Temporal differences           : 4 intervals x 14 features x 3     = 168

Structure decisions:
  * Spectral bands + indices come from the MASKED reflectance composites;
    masked / NoData pixels are EXCLUDED so each statistic is the average of
    valid pixels only (rasterstats masks by equality, so a sentinel value is
    used instead of NaN, which equality cannot match).
  * GLCM is computed from the RAW (un-masked) December composite, over the
    actual CROWN SHAPE (not the bounding box), so masked-edge artefacts do
    not distort texture.
  * Contrast / Homogeneity / Correlation are computed with the explicit thesis
    Table 3 formulas (see _glcm_props), not a library convenience function.
  * Carry-forward columns (Family, confidence, class_id, geo_polygon, ...)

Usage:
    python zonal_statistics_glcm.py --config feature_config.yaml
"""

import geopandas.geodataframe
import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from joblib import Parallel, delayed
from rasterio.features import geometry_mask
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from rasterstats import zonal_stats
from skimage.feature import graycomatrix
from tqdm import tqdm
from sklearn.impute import KNNImputer

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ── feature definitions (order follows) ────────────────────────
BAND_NAMES  = ["CoastalBlue", "Blue", "GreenI", "Green",
               "Yellow", "Red", "RedEdge", "NIR"]
INDEX_NAMES = ["NDVI", "NDVI2", "NDVI_RE", "EVI", "SAVI", "GNDVI"]
STATS       = ["mean", "std", "median"]

GLCM_PROPS    = ["contrast", "homogeneity", "correlation"]
GLCM_BAND_IDX = {"Red": 5, "NIR": 7}

# Sentinel for "ignore this pixel". Safe: reflectance is 0..1 and indices are
# roughly [-1, 1], so -9999 can never be a real value. Used instead of np.nan
# because rasterstats masks NoData by equality and (nan == nan) is False.
SENTINEL = -9999.0

GLCM_DISTANCE = 1        # pixel offset
GLCM_ANGLE    = 0.0      # radians (0 = horizontal)
GLCM_LEVELS   = 256      # grey levels; level 0 is reserved for background


# ── zonal statistics (spectral bands + indices) ───────────────────────────────

def _read_band_excluding_invalid(tif_path: str, band_idx: int):
    """Read one band as float64, replacing NaN / declared NoData with SENTINEL."""
    with rasterio.open(tif_path) as src:
        arr       = src.read(band_idx).astype("float64")
        transform = src.transform
        nodata    = src.nodata

    invalid = np.isnan(arr)
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        invalid |= (arr == nodata)
    arr[invalid] = SENTINEL
    return arr, transform


def _zonal_layer(tif_path: str, band_idx: int, prefix: str, geometries) -> pd.DataFrame:
    """Zonal mean/std/median for one raster band over all polygons.

    Only valid (non-SENTINEL) pixels contribute to each statistic. A crown with
    no valid pixels yields NaN (to be imputed later in classification prep).
    """
    arr, transform = _read_band_excluding_invalid(tif_path, band_idx)
    result = zonal_stats(geometries, arr, affine=transform,
                         stats=STATS, nodata=SENTINEL, geojson_out=False)
    df = pd.DataFrame(result)
    df.columns = [f"{prefix}_{s}" for s in STATS]
    return df


def compute_zonal_stats_for_image(geometries, composite_path: str,
                                   index_dir: str, suffix: str,
                                   n_jobs: int = 8) -> pd.DataFrame:
    """Parallel zonal stats for 8 bands + 6 indices for one date (masked data)."""
    # Spectral bands from the masked composite
    tasks = [(composite_path, b_idx, f"{name}_{suffix}")
             for b_idx, name in enumerate(BAND_NAMES, start=1)]

    # Vegetation-index rasters. compute_vegetation_indices.py writes them as
    # "<composite_stem>_<INDEX>.tif", so the stem here must match that input.
    stem = Path(composite_path).stem
    for idx_name in INDEX_NAMES:
        idx_path = str(Path(index_dir) / f"{stem}_{idx_name}.tif")
        tasks.append((idx_path, 1, f"{idx_name}_{suffix}"))

    dfs = Parallel(n_jobs=n_jobs)(
        delayed(_zonal_layer)(tp, bi, pf, geometries) for tp, bi, pf in tasks
    )
    return pd.concat(dfs, axis=1)


# ── GLCM features ──────────

def _glcm_props(P: np.ndarray):
    """Contrast / Homogeneity / Correlation from a normalised GLCM P(i, j).

    Implements the thesis Table 3 formulas directly:
        Contrast    = Σ_ij (i - j)^2 · P(i, j)
        Homogeneity = Σ_ij P(i, j) / (1 + |i - j|)
        Correlation = Σ_ij (i - µ_i)(j - µ_j) · P(i, j) / (σ_i · σ_j)
    """
    L   = P.shape[0]
    idx = np.arange(L, dtype="float64")
    i   = idx[:, None]
    j   = idx[None, :]

    contrast    = float(np.sum(((i - j) ** 2) * P))
    homogeneity = float(np.sum(P / (1.0 + np.abs(i - j))))

    mu_i  = float(np.sum(i * P))
    mu_j  = float(np.sum(j * P))
    sig_i = float(np.sqrt(np.sum(((i - mu_i) ** 2) * P)))
    sig_j = float(np.sqrt(np.sum(((j - mu_j) ** 2) * P)))

    if sig_i < 1e-12 or sig_j < 1e-12:
        correlation = np.nan          # undefined for a constant crown
    else:
        correlation = float(np.sum((i - mu_i) * (j - mu_j) * P) / (sig_i * sig_j))

    return contrast, homogeneity, correlation


def _glcm_for_polygon(poly_geom, raster_data: np.ndarray,
                      transform, band_name: str, suffix: str) -> dict:
    """GLCM texture for ONE polygon, restricted to the crown shape."""
    nan_result = {f"{band_name}_{p}_{suffix}": np.nan for p in GLCM_PROPS}

    # Integer window covering the polygon, clipped to the raster.
    minx, miny, maxx, maxy = poly_geom.bounds
    inv = ~transform
    cols, rows = [], []
    for x in (minx, maxx):
        for y in (miny, maxy):
            c, r = inv * (x, y)
            cols.append(c)
            rows.append(r)
    H, W = raster_data.shape
    col_off = max(0, int(np.floor(min(cols))))
    row_off = max(0, int(np.floor(min(rows))))
    col_end = min(W, int(np.ceil(max(cols))))
    row_end = min(H, int(np.ceil(max(rows))))
    if row_end <= row_off or col_end <= col_off:
        return nan_result

    crop = raster_data[row_off:row_end, col_off:col_end]
    win  = Window(col_off, row_off, col_end - col_off, row_end - row_off)
    win_tf = window_transform(win, transform)

    # True = inside crown; combine with validity (exclude NaN raw pixels).
    inside = geometry_mask([poly_geom], out_shape=crop.shape,
                           transform=win_tf, invert=True)
    valid = inside & ~np.isnan(crop)
    if valid.sum() < 2:
        return nan_result

    vals = crop[valid]
    mn, mx = float(vals.min()), float(vals.max())
    if mx <= mn:
        return nan_result

    # Quantise in-crown pixels to 1..(LEVELS-1); background / outside stays 0.
    q = np.zeros(crop.shape, dtype=np.uint8)
    scaled = (crop[valid] - mn) / (mx - mn)                 # 0..1
    q[valid] = 1 + np.floor(scaled * (GLCM_LEVELS - 2)).astype(np.uint8)

    glcm = graycomatrix(q, distances=[GLCM_DISTANCE], angles=[GLCM_ANGLE],
                        levels=GLCM_LEVELS, symmetric=True, normed=False)
    P = glcm[:, :, 0, 0].astype("float64")

    # Drop every co-occurrence that involves the background level (0), so only
    # crown-to-crown pixel pairs remain, then normalise to a probability.
    P[0, :] = 0.0
    P[:, 0] = 0.0
    total = P.sum()
    if total <= 0:
        return nan_result
    P /= total

    contrast, homogeneity, correlation = _glcm_props(P)
    return {
        f"{band_name}_contrast_{suffix}":    contrast,
        f"{band_name}_homogeneity_{suffix}": homogeneity,
        f"{band_name}_correlation_{suffix}": correlation,
    }


def _glcm_worker(poly_geom, red: np.ndarray, nir: np.ndarray,
                 transform, suffix: str) -> dict:
    """Compute GLCM for the Red and NIR bands of one polygon."""
    out = {}
    out.update(_glcm_for_polygon(poly_geom, red, transform, "Red", suffix))
    out.update(_glcm_for_polygon(poly_geom, nir, transform, "NIR", suffix))
    return out


def compute_glcm_features(geometries, raw_composite_path: str,
                          suffix: str, n_jobs: int = 8) -> pd.DataFrame:
    """GLCM stats for Red and NIR from the RAW composite, across all polygons."""
    with rasterio.open(raw_composite_path) as src:
        red       = src.read(GLCM_BAND_IDX["Red"] + 1).astype("float64")
        nir       = src.read(GLCM_BAND_IDX["NIR"] + 1).astype("float64")
        transform = src.transform
        nodata    = src.nodata

    # Exclude any declared NoData in the raw raster (treated as NaN in GLCM).
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        red[red == nodata] = np.nan
        nir[nir == nodata] = np.nan

    # red / nir are passed by reference each iteration, so joblib memory-maps
    # them once and shares them across workers (no per-task copy).
    records = Parallel(n_jobs=n_jobs)(
        delayed(_glcm_worker)(g, red, nir, transform, suffix)
        for g in tqdm(geometries, desc=f"GLCM {suffix}")
    )
    return pd.DataFrame(records)


# ── KNN imputation ──────────────────────────────────────────────────────────

def knn_impute_september(spectral_df: pd.DataFrame, sept_token: str = "Sept2024") -> pd.DataFrame:
    """KNN-impute (K=5) NaNs in the September columns only.

    Must run AFTER zonal-stats extraction but BEFORE temporal differences are
    computed (a Sept2024 NaN would otherwise propagate into two diff columns:
    May->Sept and Sept->Dec). December / GLCM columns are never touched.
    `spectral_df` is expected to contain ONLY the 14-feature x 5-month zonal
    stat columns (no geometry/Family/confidence/etc carry-forward columns).
    """
    sept_cols = [c for c in spectral_df.columns if sept_token in c]
    if not sept_cols:
        raise ValueError(
            f"No September columns matched '{sept_token}'. "
            f"Nothing would be imputed — check your column naming."
        )

    n_missing = int(spectral_df[sept_cols].isna().sum().sum())
    if n_missing == 0:
        print(f"KNN(K=5): no NaNs found in {len(sept_cols)} {sept_token} columns; nothing to impute.")
        return spectral_df

    imputer = KNNImputer(n_neighbors=5)              # K=5
    imputed = pd.DataFrame(
        imputer.fit_transform(spectral_df),
        columns=spectral_df.columns,
        index=spectral_df.index,
    )

    # Write back ONLY September. Every other column keeps its raw value.
    spectral_df = spectral_df.copy()
    spectral_df[sept_cols] = imputed[sept_cols]

    # Sanity check
    remaining = int(spectral_df[sept_cols].isna().sum().sum())
    print(f"KNN(K=5): filled {n_missing} missing value(s) across {len(sept_cols)} "
          f"{sept_token} columns; {remaining} NaNs left.")
    return spectral_df


# ── temporal differences ──────────────────────────────────────────────────────

def compute_temporal_differences(spectral_df: pd.DataFrame,
                                 suffixes: list) -> pd.DataFrame:
    """Difference (later - earlier) of each spectral stat over consecutive dates.

    Returns ONLY the new difference columns (168 of them for 5 dates).
    """
    intervals = list(zip(suffixes[:-1], suffixes[1:]))
    diff_cols = {}
    for s1, s2 in intervals:
        for feat in BAND_NAMES + INDEX_NAMES:
            for stat in STATS:
                c1 = f"{feat}_{s1}_{stat}"
                c2 = f"{feat}_{s2}_{stat}"
                if c1 in spectral_df.columns and c2 in spectral_df.columns:
                    diff_cols[f"{feat}_{s1}_{s2}_diff_{stat}"] = (
                        spectral_df[c2] - spectral_df[c1])
    return pd.DataFrame(diff_cols, index=spectral_df.index)


# ── main ──────────────────────────────────────────────────────────────────────

def run(config: dict):
    mask_path  = config["mask_gpkg"]
    out_path   = config["output_gpkg"]
    dates      = config["dates"]                 # [{suffix, composite, index_dir}, ...]
    glcm_date  = config["glcm_date"]             # suffix used for GLCM column names
    glcm_raw   = config["glcm_raw_composite"]    # RAW raster for GLCM
    n_jobs     = config.get("n_jobs", 8)

    gdf = gpd.read_file(mask_path)
    print(f"Loaded {len(gdf)} polygons from {mask_path}")
    original_cols = list(gdf.columns)            # carry-forward columns, never dropped
    geometries    = list(gdf.geometry)

    # 1) Spectral + index zonal stats (masked composites), per date.
    spectral_parts, suffixes = [], []
    for date_cfg in dates:
        suffix = date_cfg["suffix"]
        suffixes.append(suffix)
        print(f"\nComputing zonal stats for {suffix}…")
        spectral_parts.append(
            compute_zonal_stats_for_image(
                geometries, date_cfg["composite"], date_cfg["index_dir"],
                suffix, n_jobs=n_jobs))
    spectral_df = pd.concat(spectral_parts, axis=1)

    # 1b) KNN-impute (K=5) September cloud-gap NaNs, BEFORE temporal diffs are
    #     computed from spectral_df (otherwise a Sept2024 NaN propagates into
    #     two diff columns: May->Sept and Sept->Dec).
    sept_suffix = next((s for s in suffixes if "sep" in s.lower()), None)
    if sept_suffix is not None:
        print(f"\nKNN-imputing {sept_suffix} cloud-gap NaNs…")
        spectral_df = knn_impute_september(spectral_df, sept_token=sept_suffix)

    # 2) GLCM texture from the RAW composite of the selected month.
    print(f"\nComputing GLCM features for {glcm_date} (raw: {glcm_raw})…")
    glcm_df = compute_glcm_features(geometries, glcm_raw, glcm_date, n_jobs=n_jobs)

    # 3) Temporal differences from the spectral stats only.
    diff_df = compute_temporal_differences(spectral_df, suffixes)

    feat_df = pd.concat([spectral_df, glcm_df, diff_df], axis=1)

    # Count ONLY the newly created columns (carry-forward columns excluded).
    n_new = feat_df.shape[1]
    print(f"\nNew feature columns created: {n_new} (expected 384)")
    print(f"  spectral={spectral_df.shape[1]}  glcm={glcm_df.shape[1]}  "
          f"temporal={diff_df.shape[1]}")
    if n_new != 384:
        warnings.warn(f"Expected 384 new feature columns, produced {n_new}.")

    # Attach features positionally; keep all original columns.
    for col in feat_df.columns:
        gdf[col] = feat_df[col].values

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG")
    print(f"Carried-forward columns kept: {original_cols}")
    print(f"Saved → {out_path}  ({len(gdf.columns)} total columns)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute zonal stats + GLCM features")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run(cfg)