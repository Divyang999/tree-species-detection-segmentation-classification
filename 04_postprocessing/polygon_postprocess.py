"""
Post-process raw YOLO segmentation polygons (Part A: filter + clean).

Follows the thesis segmentation post-processing, in this order:
  1. Confidence + area filter
       - drop polygons with confidence < 0.3   (visual-inspection threshold)
       - drop polygons smaller than 18 m²       (< ~2 PlanetScope pixels)
       - drop polygons larger than the max-area ceiling (clusters / fields / non-tree)
  2. IoU merge (Non-Maximum Suppression)
       - polygon pairs overlapping by IoU > 0.8 are treated as duplicates of the
         same tree and merged; the higher-confidence polygon is the anchor.
  3. Convex hull
       - smooth each final polygon to the smallest enclosing convex shape.

All metric operations (area, overlap) are done in UTM.

Max-area ceiling — personal note:
  The widest avenue species here are the rain tree (Samanea saman, Fabaceae) and
  banyan/figs (Moraceae). A typical large rain-tree crown ~30 m across ≈ 707 m²;
  very large specimens ~40 m ≈ 1,250 m². 1,200 m² is used as a
  generous single-crown ceiling; larger polygons are almost certainly merged
  clusters or non-tree objects. Verified against histogram.

Usage:
    python polygon_postprocess.py \\
        --input  raw_segments.csv \\ (geo)
        --output processed_segments.csv \\
        --utm-epsg 32643 \\
        --conf-min 0.3 \\
        --area-min 18 \\
        --area-max 1200 \\
        --iou-threshold 0.8
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from rtree import index as rtree_index
from shapely.ops import unary_union
from shapely.wkt import loads


# ── load ──────────────────────────────────────────────────────────────────────
def load_polygons(csv_path: str, utm_epsg: int) -> gpd.GeoDataFrame:
    """Read the inference CSV (geo_polygon, confidence, class_id) into UTM."""
    df = pd.read_csv(csv_path)
    df["geometry"] = df["geo_polygon"].apply(loads)
    # Keep attributes; default them if a column is missing.
    if "confidence" not in df.columns:
        df["confidence"] = 1.0
    if "class_id" not in df.columns:
        df["class_id"] = 0
    gdf = gpd.GeoDataFrame(
        df[["confidence", "class_id", "geometry"]],
        geometry="geometry", crs="EPSG:4326",
    )
    return gdf.to_crs(f"EPSG:{utm_epsg}")


# ── step 1: confidence + area filter ───────────────────────────────────────────
def filter_confidence_area(gdf: gpd.GeoDataFrame, conf_min: float,
                           area_min: float, area_max: float) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    # Drop invalid/empty geometries first so .area is meaningful.
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid & ~gdf.geometry.is_empty]
    gdf["area_m2"] = gdf.geometry.area

    keep = (
        (gdf["confidence"] >= conf_min)
        & (gdf["area_m2"] >= area_min)
        & (gdf["area_m2"] <= area_max)
    )
    return gdf[keep].reset_index(drop=True)


# ── step 2: IoU merge (NMS) ─────────────────────────────────────────────────────
def nms_merge(gdf: gpd.GeoDataFrame, iou_threshold: float) -> gpd.GeoDataFrame:
    """
    Merge polygon pairs whose IoU exceeds the threshold (duplicates of one tree).
    Processed highest-confidence first; the anchor keeps its confidence/class.
    """
    polys   = list(gdf.geometry)
    confs   = list(gdf["confidence"])
    classes = list(gdf["class_id"])

    # Spatial index for fast overlap candidate lookup.
    idx = rtree_index.Index()
    for i, p in enumerate(polys):
        idx.insert(i, p.bounds)

    order = sorted(range(len(polys)), key=lambda i: confs[i], reverse=True)
    used = set()
    out_geom, out_conf, out_cls = [], [], []

    for i in order:
        if i in used:
            continue
        used.add(i)
        p1 = polys[i]
        group = [p1]
        for j in idx.intersection(p1.bounds):
            if j == i or j in used:
                continue
            p2 = polys[j]
            if not p1.intersects(p2):
                continue
            inter = p1.intersection(p2).area
            union = p1.union(p2).area
            if union > 0 and inter / union > iou_threshold:
                group.append(p2)
                used.add(j)
        out_geom.append(unary_union(group))
        out_conf.append(confs[i])      # anchor (highest-confidence) attributes
        out_cls.append(classes[i])

    return gpd.GeoDataFrame(
        {"confidence": out_conf, "class_id": out_cls, "geometry": out_geom},
        geometry="geometry", crs=gdf.crs,
    )


# ── step 3: convex hull ─────────────────────────────────────────────────────────
def apply_convex_hull(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.convex_hull
    return gdf


# ── orchestration ───────────────────────────────────────────────────────────────
def postprocess(input_csv: str, output_csv: str, utm_epsg: int,
                conf_min: float, area_min: float, area_max: float,
                iou_threshold: float):
    print(f"Loading {input_csv}…")
    gdf = load_polygons(input_csv, utm_epsg)
    print(f"  Raw polygons: {len(gdf)} (EPSG:{utm_epsg})")

    gdf = filter_confidence_area(gdf, conf_min, area_min, area_max)
    print(f"  After confidence+area filter: {len(gdf)} "
          f"(conf≥{conf_min}, {area_min}–{area_max} m²)")

    gdf = nms_merge(gdf, iou_threshold)
    print(f"  After IoU merge (>{iou_threshold}): {len(gdf)}")

    gdf = apply_convex_hull(gdf)
    print(f"  After convex hull: {len(gdf)}")

    gdf_out = gdf.to_crs("EPSG:4326")
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({
        "geo_polygon": gdf_out.geometry.apply(lambda g: g.wkt),
        "confidence":  gdf_out["confidence"],
        "class_id":    gdf_out["class_id"],
    })
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} polygons → {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process YOLO segmentation polygons (filter + clean)")
    parser.add_argument("--input",         required=True)
    parser.add_argument("--output",        required=True)
    parser.add_argument("--utm-epsg",      type=int,   default=32643,
                        help="UTM EPSG for metric ops (default 32643 = WGS84 UTM 43N, Bengaluru)")
    parser.add_argument("--conf-min",      type=float, default=0.3,
                        help="Drop polygons below this confidence (thesis: 0.3)")
    parser.add_argument("--area-min",      type=float, default=18.0,
                        help="Drop polygons smaller than this in m² (thesis: 18)")
    parser.add_argument("--area-max",      type=float, default=1200.0,
                        help="Drop polygons larger than this in m² (single-crown ceiling)")
    parser.add_argument("--iou-threshold", type=float, default=0.8,
                        help="IoU above which polygons are merged as duplicates (thesis: 0.8)")
    args = parser.parse_args()

    postprocess(args.input, args.output, args.utm_epsg,
                args.conf_min, args.area_min, args.area_max, args.iou_threshold)