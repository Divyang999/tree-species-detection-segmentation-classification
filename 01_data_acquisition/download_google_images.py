"""
Build a satellite-image grid over an Area of Interest (AOI) and download every
tile from the Google Maps Static API.

This is hardest code of this notebook for me (everything is to be manually edited) so be careful:
    Step 1 (calculate_grid_size)  : confirm the ground size of one tile  -> ~186 m
    Step 2 (create_grids)         : split the AOI into grid cells, save their
                                    centers to grid_centers.csv
    Step 3 (bulk_download)        : download one image per center, named
                                    {grid_number}_{lat}_{lng}.png

----------------------------------------------------------------------------
HOW TO RUN
----------------------------------------------------------------------------
1. Install libraries

2. Put your Google key in an environment variable

3. Edit the CONFIG section below and run:
       python download_aoi_grid_images.py

   Optional: override paths on the command line instead of editing the file:
       python download_aoi_grid_images.py --aoi "South AOI.csv" --out ./images/

----------------------------------------------------------------------------
NOTES
----------------------------------------------------------------------------
* The math: ground_resolution = base * cos(lat) / 2**zoom / scale,
  and the grid is built in EPSG:3857 (metre-based) then converted back to lat/long.
* IMAGE_SIZE is (1280, 1280) as in the notebook run. The Google free tier caps
  requests at 640; if you find a downloaded PNG is 2560x2560 instead of 1280x1280,
  set IMAGE_SIZE = (640, 640) so that 640 * scale(2) = 1280.
"""

import os
import math
import argparse

import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point


# ============================================================================
# CONFIG  --  edit these, or override with command-line flags
# ============================================================================
AOI_PATH = "South AOI.csv"          # AOI vector file geopandas can read
                                    # (GeoJSON / shapefile / geopackage, or a
                                    #  CSV containing a WKT geometry column)
OUTPUT_FOLDER = "./Banglore_images" # where the .png tiles are saved
GRID_CSV = "./grid_centers.csv"     # where the grid centers are written/read

ZOOM = 19                           # Google Maps zoom level
SCALE = 2                           # 2 = high-resolution
IMAGE_SIZE = (1280, 1280)           # requested image size
GRID_SIZE_METERS = 185              # grid spacing in metres (gives ~1 m overlap)

REGENERATE_GRID = True              # False = reuse existing GRID_CSV, skip Step 2
TEST_LIMIT = None                   # None = download all; or e.g. 10 to test first
# ============================================================================


# ---- Step 1: ground size of a single tile ----------------------------------
def calculate_grid_size(lat, zoom=ZOOM, scale=SCALE, img_size=IMAGE_SIZE[0]):
    """Ground size in metres of one tile at the given latitude."""
    base_resolution = 156543.03392                      # m/px at zoom 0, equator
    resolution = (math.cos(math.radians(lat)) * base_resolution) / (2 ** zoom) / scale
    return img_size * resolution


# ---- Step 2: split the AOI into grid cells ----------------------------------
def create_grids(aoi_path, grid_size_meters, output_csv):
    """
    Split the AOI polygon into `grid_size_meters` square cells and save the
    center of every cell that touches the AOI.

    Returns a DataFrame with columns: Grid_number, Center_lat, Center_long
    """
    aoi = gpd.read_file(aoi_path)
    if aoi.empty:
        raise ValueError("The AOI file is empty.")

    # Ensure WGS84, then reproject to metre-based Web Mercator for square grids.
    aoi = aoi.set_crs(epsg=4326) if aoi.crs is None else aoi.to_crs(epsg=4326)
    aoi_m = aoi.to_crs(epsg=3857)

    minx, miny, maxx, maxy = aoi_m.total_bounds
    num_x = math.ceil((maxx - minx) / grid_size_meters)
    num_y = math.ceil((maxy - miny) / grid_size_meters)

    rows = []
    grid_number = 1
    for i in range(num_x):
        for j in range(num_y):
            x1 = minx + i * grid_size_meters
            y1 = miny + j * grid_size_meters
            x2, y2 = x1 + grid_size_meters, y1 + grid_size_meters
            cell = box(x1, y1, x2, y2)

            # Keep only cells that overlap the AOI.
            if aoi_m.intersects(cell).any():
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                # Convert the cell center back to lat/long.
                center = gpd.GeoSeries([Point(cx, cy)], crs=3857).to_crs(epsg=4326).iloc[0]
                rows.append([grid_number, center.y, center.x])   # lat, long
                grid_number += 1

    if not rows:
        raise ValueError("No grid cells intersected the AOI.")

    grid_df = pd.DataFrame(rows, columns=["Grid_number", "Center_lat", "Center_long"])
    grid_df.to_csv(output_csv, index=False)
    print(f"Step 2: {len(grid_df)} grid centers saved -> {output_csv}")
    return grid_df


# ---- Step 3a: download one tile ---------------------------------------------
def download_satellite_image(lat, lng, image_name, output_folder, api_key,
                             zoom=ZOOM, scale=SCALE, image_size=IMAGE_SIZE):
    """Download a single satellite tile centered on (lat, lng)."""
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lng}",
        "zoom": zoom,
        "size": f"{image_size[0]}x{image_size[1]}",
        "scale": scale,
        "maptype": "satellite",
        "key": api_key,
    }
    response = requests.get(base_url, params=params, stream=True, timeout=30)
    if response.status_code == 200:
        os.makedirs(output_folder, exist_ok=True)
        path = os.path.join(output_folder, image_name)
        with open(path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        return True, path
    return False, f"HTTP {response.status_code}: {response.text[:200]}"


# ---- Step 3b: download every tile in the grid CSV ---------------------------
def bulk_download_from_csv(csv_file, output_folder, api_key,
                           zoom=ZOOM, scale=SCALE, image_size=IMAGE_SIZE,
                           limit=None):
    """Read grid centers and download one image per row."""
    data = pd.read_csv(csv_file)
    required = ["Grid_number", "Center_lat", "Center_long"]
    if not all(c in data.columns for c in required):
        raise ValueError(f"CSV must contain columns: {required}")

    if limit:
        data = data.head(limit)

    total = len(data)
    downloaded = failed = 0
    for _, row in data.iterrows():
        grid_number = row["Grid_number"]
        lat, lng = row["Center_lat"], row["Center_long"]
        # Filename format matches the thesis: 1.0_<lat>_<lng>.png
        image_name = f"{float(grid_number)}_{lat}_{lng}.png"

        ok, msg = download_satellite_image(
            lat, lng, image_name, output_folder, api_key, zoom, scale, image_size
        )
        if ok:
            downloaded += 1
        else:
            failed += 1
            print(f"  FAILED {image_name}: {msg}")

        done = downloaded + failed
        if done % 10 == 0 or done == total:
            print(f"\rStep 3: {downloaded}/{total} downloaded, {failed} failed",
                  end="", flush=True)

    print(f"\nStep 3: complete -- downloaded {downloaded}, failed {failed}")


# ---- Orchestration ----------------------------------------------------------
def main(aoi_path, output_folder, grid_csv, api_key,
         zoom, scale, image_size, grid_size_meters,
         regenerate_grid, test_limit):

    # Step 1 -- sanity check the tile size (In my case Bengaluru latitude ~12.97)
    print(f"Step 1: tile ground size ~= {calculate_grid_size(12.97, zoom, scale, image_size[0]):.1f} m")

    # Step 2 -- build the grid (or reuse an existing one)
    if regenerate_grid or not os.path.exists(grid_csv):
        create_grids(aoi_path, grid_size_meters, grid_csv)
    else:
        print(f"Step 2: reusing existing grid -> {grid_csv}")

    # Step 3 -- download the images
    bulk_download_from_csv(grid_csv, output_folder, api_key,
                           zoom, scale, image_size, limit=test_limit)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build an AOI grid and download Google satellite tiles.")
    p.add_argument("--aoi", default=AOI_PATH, help="Path to the AOI vector file")
    p.add_argument("--out", default=OUTPUT_FOLDER, help="Output folder for images")
    p.add_argument("--grid-csv", default=GRID_CSV, help="Path for the grid_centers CSV")
    p.add_argument("--zoom", type=int, default=ZOOM)
    p.add_argument("--scale", type=int, default=SCALE)
    p.add_argument("--grid-size", type=float, default=GRID_SIZE_METERS, help="Grid spacing in metres")
    p.add_argument("--test", type=int, default=TEST_LIMIT, help="Only download the first N images")
    p.add_argument("--reuse-grid", action="store_true", help="Reuse existing grid CSV instead of rebuilding")
    args = p.parse_args()

    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise EnvironmentError("Set the GOOGLE_API_KEY environment variable before running.")

    main(
        aoi_path=args.aoi,
        output_folder=args.out,
        grid_csv=args.grid_csv,
        api_key=key,
        zoom=args.zoom,
        scale=args.scale,
        image_size=IMAGE_SIZE,
        grid_size_meters=args.grid_size,
        regenerate_grid=not args.reuse_grid if args.reuse_grid else REGENERATE_GRID,
        test_limit=args.test,
    )