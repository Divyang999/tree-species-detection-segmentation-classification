"""
Merge ground-truth tree point locations with segmented tree crown polygons.

Logic:
  1. Spatial join — assign tree points that fall inside a polygon (1:1, closest
     to centroid wins when multiple trees map to the same polygon)
  2. Iterative buffer expansion — remaining unmatched trees are assigned to the
     nearest unassigned polygon within a 3-metre buffer
  3. Produce a merged GeoDataFrame (polygon geometry + tree attributes)

Input CSVs must have a WKT column:
  polygon_csv  — polygon geometries (WKT column for MultiPolygon/Polygon)
  tree_csv     — tree point locations (WKT column for Point) with species labels

Usage:
    python merge_ground_segments.py \\
        --polygons processed_segments.csv \\
        --trees    ground_truth_trees.csv \\
        --output   merged_output.gpkg
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.wkt import loads

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _safe_load_wkt(wkt, expected_type):
    try:
        geom = loads(wkt)
        if isinstance(geom, expected_type):
            return geom
    except Exception:
        pass
    return None


def load_data(polygon_csv: str, tree_csv: str):
    poly_df  = pd.read_csv(polygon_csv)
    trees_df = pd.read_csv(tree_csv)

    poly_df["geometry"]  = poly_df["WKT"].apply(
        lambda w: _safe_load_wkt(w, (Polygon, MultiPolygon)))
    trees_df["geometry"] = trees_df["WKT"].apply(
        lambda w: _safe_load_wkt(w, Point))

    poly_gdf  = gpd.GeoDataFrame(poly_df.dropna(subset=["geometry"]),
                                  geometry="geometry", crs="EPSG:4326")
    tree_gdf  = gpd.GeoDataFrame(trees_df.dropna(subset=["geometry"]),
                                  geometry="geometry", crs="EPSG:4326")

    logging.info("Loaded %d polygons, %d tree points", len(poly_gdf), len(tree_gdf))
    return poly_gdf, tree_gdf


def metres_to_degrees(metres: float, latitude: float) -> float:
    return metres / (111320 * np.cos(np.radians(latitude)))


def assign_trees(poly_gdf: gpd.GeoDataFrame, tree_gdf: gpd.GeoDataFrame,
                 buffer_metres: float = 3.0) -> gpd.GeoDataFrame:
    poly_to_tree = {}
    tree_used    = set()
    poly_sindex  = poly_gdf.sindex

    # ── Step 1: spatial join (points inside polygons) ─────────────────────────
    joined = gpd.sjoin(tree_gdf, poly_gdf[["geometry"]], how="left", predicate="within")
    for poly_idx, group in joined.groupby("index_right"):
        poly_idx = int(poly_idx)
        if len(group) == 1:
            t_idx = group.index[0]
            if t_idx not in tree_used:
                poly_to_tree[poly_idx] = t_idx
                tree_used.add(t_idx)
        else:
            centroid = poly_gdf.loc[poly_idx, "geometry"].centroid
            dist     = group["geometry"].apply(lambda g: g.distance(centroid))
            t_idx    = dist.idxmin()
            if t_idx not in tree_used:
                poly_to_tree[poly_idx] = t_idx
                tree_used.add(t_idx)

    assigned_polys   = set(poly_to_tree.keys())
    remaining_trees  = tree_gdf[~tree_gdf.index.isin(tree_used)].copy()
    logging.info("After spatial join: %d / %d trees assigned",
                 len(tree_used), len(tree_gdf))

    # ── Step 2: buffer-based expansion ───────────────────────────────────────
    iteration = 1
    while len(remaining_trees) > 0:
        avg_lat     = remaining_trees.geometry.y.mean()
        buf_deg     = metres_to_degrees(buffer_metres, avg_lat)
        buffered    = remaining_trees.geometry.buffer(buf_deg)
        new_count   = 0

        for t_idx, buf in zip(remaining_trees.index, buffered):
            candidates = list(poly_sindex.intersection(buf.bounds))
            for p_idx in candidates:
                if p_idx in assigned_polys:
                    continue
                if not buf.intersects(poly_gdf.loc[p_idx, "geometry"]):
                    continue
                poly_to_tree[p_idx] = t_idx
                tree_used.add(t_idx)
                assigned_polys.add(p_idx)
                new_count += 1
                break

        remaining_trees = tree_gdf[~tree_gdf.index.isin(tree_used)].copy()
        logging.info("Iteration %d: +%d assigned, %d remaining",
                     iteration, new_count, len(remaining_trees))
        if new_count == 0:
            break
        iteration += 1

    # ── Step 3: build merged GeoDataFrame ────────────────────────────────────
    records = []
    for p_idx, p_row in poly_gdf.iterrows():
        rec = {"geometry": p_row["geometry"]}
        t_idx = poly_to_tree.get(p_idx)
        if t_idx is not None:
            t_row = tree_gdf.loc[t_idx].drop(["geometry", "WKT"], errors="ignore")
            rec.update(t_row.to_dict())
        records.append(rec)

    merged = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    logging.info("Merged GDF: %d rows, %d with tree labels",
                 len(merged), len(poly_to_tree))
    return merged


def main(polygon_csv: str, tree_csv: str, output_gpkg: str):
    poly_gdf, tree_gdf = load_data(polygon_csv, tree_csv)
    merged = assign_trees(poly_gdf, tree_gdf)
    Path(output_gpkg).parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(output_gpkg, driver="GPKG")
    logging.info("Saved → %s", output_gpkg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge ground truth tree points with segmented polygons")
    parser.add_argument("--polygons", required=True, help="Post-processed segment CSV")
    parser.add_argument("--trees",    required=True, help="Ground truth tree point CSV")
    parser.add_argument("--output",   required=True, help="Output GeoPackage path")
    args = parser.parse_args()

    main(args.polygons, args.trees, args.output)
