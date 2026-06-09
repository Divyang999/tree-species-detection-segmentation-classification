"""
Compute vegetation indices from a preprocessed 8-band PlanetScope image.

Band order (1-indexed):
  1 CoastalBlue  2 Blue  3 GreenI  4 Green
  5 Yellow       6 Red   7 RedEdge 8 NIR

Indices produced:
  NDVI    (NIR - Red)  / (NIR + Red)
  NDVI2   (NIR - RedEdge) / (NIR + RedEdge)
  NDVI_RE (RedEdge - Red) / (RedEdge + Red)   [red-edge NDVI]
  EVI     2.5 * (NIR-Red) / (NIR + 6*Red - 7.5*Blue + 1)
  SAVI    ((NIR-Red) / (NIR+Red+L)) * (1+L),  L=0.5
  GNDVI   (NIR - Green) / (NIR + Green)

Usage:
    python compute_vegetation_indices.py --input masked.tif --outdir ./indices/
"""

import argparse
import numpy as np
import rasterio
from pathlib import Path

EPS = 1e-6  # avoid division by zero


def read_bands(tif_path: Path):
    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)
        meta = src.meta.copy()
        transform = src.transform
    return data, meta, transform


def save_index(arr: np.ndarray, meta: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    m = meta.copy()
    m.update(count=1, dtype=rasterio.float32, nodata=np.nan)
    with rasterio.open(path, "w", **m) as dst:
        dst.write(arr.astype(np.float32), 1)


def compute_indices(tif_path: Path, outdir: Path):
    data, meta, _ = read_bands(tif_path)

    coastal_blue = data[0]
    blue         = data[1]
    green        = data[3]
    red          = data[5]
    red_edge     = data[6]
    nir          = data[7]

    L = 0.5

    indices = {
        "NDVI":    np.clip((nir - red)      / (nir + red      + EPS), -1, 1),
        "NDVI2":   np.clip((nir - red_edge) / (nir + red_edge + EPS), -1, 1),
        "NDVI_RE": np.clip((red_edge - red) / (red_edge + red + EPS), -1, 1),
        "EVI":     np.clip(
                       2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + EPS),
                       -1.5, 1.5),
        "SAVI":    np.clip(
                       ((nir - red) / (nir + red + L + EPS)) * (1 + L),
                       -1, 1),
        "GNDVI":   np.clip((nir - green) / (nir + green + EPS), -1, 1),
    }

    stem = tif_path.stem
    for name, arr in indices.items():
        out = outdir / f"{stem}_{name}.tif"
        save_index(arr, meta, out)
        print(f"  Saved {name} → {out.name}")

    return indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute vegetation indices from 8-band PlanetScope TIF")
    parser.add_argument("--input",  required=True, help="Preprocessed 8-band TIF")
    parser.add_argument("--outdir", required=True, help="Directory for output index TIFs")
    args = parser.parse_args()

    compute_indices(Path(args.input), Path(args.outdir))
