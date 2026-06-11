"""
Merge ground-truth tree points with segmented tree crown polygons (Part B).

Follows the thesis logic Zonal Statistics:
    "Each ground truth tree point was assigned to the nearest unassigned polygon
     within a 2 m buffer, assuming GPS accuracy during ground survey is ~1-5 m."

Implementation:
  - All matching is done in UTM (EPSG:32643), so distances and the 2 m buffer
    are true metres.
  - One-to-one, nearest-first: every (tree, polygon) pair within 2 m is ranked by
    point-to-polygon distance (0 if the point is inside the polygon), with ties
    broken by distance to the polygon centroid. Tree are assigned from
    closest to farthest, so each tree claims its nearest free polygon and each
    polygon receives at most one tree.
  - Polygons that receive no tree stay in the output unlabeled.

Inputs:
  --polygons : Part A output CSV. Geometry in column `geo_polygon` (WKT),
               also carries `confidence`, `class_id`.
  --trees    : ground-truth CSV. Point geometry as WKT, or lat/long columns.

Usage:
    python merge_ground_segments.py \\
        --polygons processed_segments.csv \\
        --trees    ground_truth_trees.csv \\
        --output   merged_output.gpkg \\
        --buffer 2.0 --utm-epsg 32643
"""

import argparse
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.wkt import loads

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# column-name candidates used for auto-detection
WKT_CANDIDATES = ["geo_polygon", "WKT", "wkt", "geometry", "geom"]
LAT_CANDIDATES = ["latitude", "Latitude", "lat", "Lat", "Y", "y"]
LON_CANDIDATES = ["longitude", "Longitude", "lon", "Lon", "lng", "X", "x"]


def _first_present(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def load_polygons(path: str, geom_col: str, utm_epsg: int) -> gpd.GeoDataFrame:
    df = pd.read_csv(path)
    col = geom_col if geom_col in df.columns else _first_present(df.columns, WKT_CANDIDATES)
    if col is None:
        raise ValueError(f"No polygon WKT column found in {path} (looked for {WKT_CANDIDATES})")
    df["geometry"] = df[col].apply(loads)
    df = df.drop(columns=[col])
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326").to_crs(epsg=utm_epsg)
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid].reset_index(drop=True)
    return gdf


def load_trees(path: str, wkt_col: str, lat_col: str, lon_col: str,
               utm_epsg: int) -> gpd.GeoDataFrame:
    df = pd.read_csv(path)

    wcol = wkt_col if (wkt_col and wkt_col in df.columns) else _first_present(df.columns, WKT_CANDIDATES)
    if wcol is not None:
        df["geometry"] = df[wcol].apply(loads)
        df = df.drop(columns=[wcol])
    else:
        la = lat_col if (lat_col and lat_col in df.columns) else _first_present(df.columns, LAT_CANDIDATES)
        lo = lon_col if (lon_col and lon_col in df.columns) else _first_present(df.columns, LON_CANDIDATES)
        if la is None or lo is None:
            raise ValueError(
                f"No point geometry found in {path}. Provide a WKT column or lat/long columns."
            )
        df["geometry"] = [Point(xy) for xy in zip(df[lo], df[la])]

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326").to_crs(epsg=utm_epsg)
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.geom_type.eq("Point")].reset_index(drop=True)
    return gdf


def assign_trees(poly_gdf: gpd.GeoDataFrame, tree_gdf: gpd.GeoDataFrame,
                 buffer_m: float) -> gpd.GeoDataFrame:
    """Nearest-unassigned-polygon, one-to-one, within `buffer_m` metres."""
    poly_geoms     = poly_gdf.geometry.values
    poly_centroids = poly_gdf.geometry.centroid.values
    sindex         = poly_gdf.sindex

    # Build (tree, polygon) pairs within the buffer.
    candidates = []  # (boundary_dist, centroid_dist, tree_pos, poly_pos)
    for t_pos, tgeom in enumerate(tree_gdf.geometry.values):
        bbox = (tgeom.x - buffer_m, tgeom.y - buffer_m,
                tgeom.x + buffer_m, tgeom.y + buffer_m)
        for p_pos in sindex.intersection(bbox):
            d = tgeom.distance(poly_geoms[p_pos])
            if d <= buffer_m:
                cd = tgeom.distance(poly_centroids[p_pos])
                candidates.append((d, cd, t_pos, p_pos))

    # Assign closest first, closer to centroid wins
    candidates.sort(key=lambda r: (r[0], r[1]))
    tree_used, poly_used, poly_to_tree = set(), set(), {}
    for d, cd, t_pos, p_pos in candidates:
        if t_pos in tree_used or p_pos in poly_used:
            continue
        poly_to_tree[p_pos] = t_pos
        tree_used.add(t_pos)
        poly_used.add(p_pos)

    logging.info("Matched %d tree-polygon pairs (of %d trees, %d polygons)",
                 len(poly_to_tree), len(tree_gdf), len(poly_gdf))
    logging.info("Unlabeled polygons (reserved for inference): %d",
                 len(poly_gdf) - len(poly_to_tree))
    logging.info("Unmatched trees (no polygon within %.1f m): %d",
                 buffer_m, len(tree_gdf) - len(tree_used))

    # Attach tree attributes onto their matched polygons.
    merged = poly_gdf.copy()
    tree_attr_cols = [c for c in tree_gdf.columns if c != "geometry"]

    if poly_to_tree:
        match = pd.Series(poly_to_tree, name="tree_pos")          # index = poly_pos
        attrs = tree_gdf.iloc[match.values][tree_attr_cols].copy()
        attrs.index = match.index                                  # re-key to poly_pos
        merged = merged.join(attrs)                                # aligns on poly index
    else:
        for c in tree_attr_cols:
            merged[c] = np.nan

    return merged


def main(polygon_csv, tree_csv, output_gpkg, poly_geom_col, tree_wkt_col,
         tree_lat_col, tree_lon_col, buffer_m, utm_epsg):
    poly_gdf = load_polygons(polygon_csv, poly_geom_col, utm_epsg)
    tree_gdf = load_trees(tree_csv, tree_wkt_col, tree_lat_col, tree_lon_col, utm_epsg)
    logging.info("Loaded %d polygons, %d tree points (EPSG:%d)",
                 len(poly_gdf), len(tree_gdf), utm_epsg)

    merged = assign_trees(poly_gdf, tree_gdf, buffer_m)

    from pathlib import Path
    Path(output_gpkg).parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(output_gpkg, driver="GPKG")
    logging.info("Saved %d polygons (%d labeled) -> %s (EPSG:%d)",
                 len(merged), merged.iloc[:, -1].notna().sum() if len(merged) else 0,
                 output_gpkg, utm_epsg)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Merge ground-truth tree points with segmented polygons")
    p.add_argument("--polygons", required=True, help="Part A output CSV (geo_polygon WKT)")
    p.add_argument("--trees",    required=True, help="Ground-truth tree CSV (WKT or lat/long)")
    p.add_argument("--output",   required=True, help="Output GeoPackage path")
    p.add_argument("--poly-geom-col", default="geo_polygon", help="Polygon WKT column name")
    p.add_argument("--tree-wkt-col",  default="WKT", help="Tree point WKT column name (if used)")
    p.add_argument("--tree-lat-col",  default="", help="Tree latitude column (if no WKT)")
    p.add_argument("--tree-lon-col",  default="", help="Tree longitude column (if no WKT)")
    p.add_argument("--buffer",   type=float, default=2.0, help="Match buffer in metres (thesis: 2.0)")
    p.add_argument("--utm-epsg", type=int,   default=32643, help="UTM EPSG (default 32643 = UTM 43N)")
    args = p.parse_args()

    main(args.polygons, args.trees, args.output, args.poly_geom_col,
         args.tree_wkt_col, args.tree_lat_col, args.tree_lon_col,
         args.buffer, args.utm_epsg)