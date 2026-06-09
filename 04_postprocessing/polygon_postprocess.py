"""
Post-process raw YOLO segmentation polygons.

Steps:
  1. Load polygon WKT from CSV
  2. Reproject to UTM for metric operations
  3. Simplify polygons (Ramer-Douglas-Peucker, tolerance in metres)
  4. Merge overlapping polygons by IoU threshold (Non-Maximum Suppression)
  5. Reproject back to WGS-84 and save

Usage:
    python polygon_postprocess.py \\
        --input  raw_segments.csv \\
        --output processed_segments.csv \\
        --utm-epsg 32643 \\
        --tolerance 0.5 \\
        --iou-threshold 0.3
"""

import argparse
import multiprocessing
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rtree import index as rtree_index
from shapely.geometry import mapping
from shapely.ops import unary_union
from shapely.wkt import loads
from tqdm import tqdm


# ── helpers ───────────────────────────────────────────────────────────────────

def load_polygons(csv_path: str, utm_epsg: int) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path)
    df["geometry"] = df["geo_polygon"].apply(loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    return gdf.to_crs(f"EPSG:{utm_epsg}")


def _simplify_one(args):
    poly, tol = args
    try:
        s = poly.simplify(tol, preserve_topology=True)
        return s if s.is_valid and not s.is_empty else poly
    except Exception:
        return poly


def simplify_polygons(gdf: gpd.GeoDataFrame, tolerance: float) -> gpd.GeoDataFrame:
    """Simplify each polygon in parallel (tolerance in CRS units, i.e. metres for UTM)."""
    with multiprocessing.Pool() as pool:
        simplified = list(tqdm(
            pool.imap(_simplify_one, [(g, tolerance) for g in gdf.geometry]),
            total=len(gdf), desc="Simplifying"
        ))
    gdf = gdf.copy()
    gdf["geometry"] = simplified
    return gdf


def nms_merge(polygons: list, iou_threshold: float) -> list:
    """IoU-based Non-Maximum Suppression merge for a list of Shapely polygons."""
    idx = rtree_index.Index()
    for i, poly in enumerate(polygons):
        idx.insert(i, poly.bounds)

    merged = []
    used = set()

    for i, p1 in enumerate(polygons):
        if i in used:
            continue
        group = [p1]
        for j in idx.intersection(p1.bounds):
            if j == i or j in used:
                continue
            p2 = polygons[j]
            if not p1.intersects(p2):
                continue
            inter = p1.intersection(p2).area
            union = p1.union(p2).area
            if union > 0 and inter / union > iou_threshold:
                group.append(p2)
                used.add(j)
        merged.append(unary_union(group))

    return merged


def postprocess(input_csv: str, output_csv: str, utm_epsg: int,
                tolerance: float, iou_threshold: float):
    print(f"Loading {input_csv}…")
    gdf = load_polygons(input_csv, utm_epsg)
    print(f"  {len(gdf)} polygons loaded (EPSG:{utm_epsg})")

    gdf = simplify_polygons(gdf, tolerance)
    print(f"  After simplification: {len(gdf)}")

    print("Merging overlapping polygons (IoU NMS)…")
    merged_geoms = nms_merge(list(gdf.geometry), iou_threshold)
    gdf_merged = gpd.GeoDataFrame(geometry=merged_geoms, crs=f"EPSG:{utm_epsg}")
    print(f"  After merging: {len(gdf_merged)}")

    gdf_out = gdf_merged.to_crs("EPSG:4326")
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({"geo_polygon": gdf_out.geometry.apply(lambda g: g.wkt)})
    out_df.to_csv(out_path, index=False)
    print(f"Saved → {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process YOLO segmentation polygons")
    parser.add_argument("--input",         required=True)
    parser.add_argument("--output",        required=True)
    parser.add_argument("--utm-epsg",      type=int,   default=32643,
                        help="UTM EPSG for metric operations (default 32643 = WGS84 UTM 43N, Pune/Bangalore)")
    parser.add_argument("--tolerance",     type=float, default=0.5,
                        help="RDP simplification tolerance in metres (default 0.5)")
    parser.add_argument("--iou-threshold", type=float, default=0.3,
                        help="IoU threshold for NMS merge (default 0.3)")
    args = parser.parse_args()

    postprocess(args.input, args.output, args.utm_epsg, args.tolerance, args.iou_threshold)
