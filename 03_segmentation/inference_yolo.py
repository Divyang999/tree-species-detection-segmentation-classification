"""
Run YOLO11l-seg inference on satellite images and export tree crown polygons to CSV.

Output CSV columns: image_id, geo_polygon (WKT), confidence, class_id
Georeferencing: each image filename encodes  {id}_{lat}_{lon}.png — the script
uses the Google Maps Static API zoom/scale to derive pixel → coordinate transform.

Usage:
    python inference_yolo.py --model best.pt --images ./images/ --output segments.csv
"""

import argparse
import math
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from ultralytics import YOLO

# Google Maps Static API tile parameters used during image download
ZOOM   = 19
SCALE  = 2
IMG_PX = 1280       # Confirm manually against actual pixel size of downloaded images


def meters_per_pixel(lat: float, zoom: int) -> float:
    """Ground resolution at a given latitude and zoom level (metres/pixel)."""
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)


def pixel_to_latlon(px_x: float, px_y: float, centre_lat: float,
                    centre_lon: float, mpp: float, img_size: int) -> tuple:
    half = img_size / 2
    dx_m = (px_x - half) * mpp
    dy_m = (half - px_y) * mpp
    lat = centre_lat + (dy_m / 111320)
    lon = centre_lon + (dx_m / (111320 * math.cos(math.radians(centre_lat))))
    return lat, lon


def parse_filename(fname: str):
    """Extract (image_id, lat, lon) from  '{id}_{lat}_{lon}.png'."""
    m = re.match(r"^([\d.]+)_([\d.\-]+)_([\d.\-]+)\.png$", fname)
    if not m:
        return None, None, None
    return m.group(1), float(m.group(2)), float(m.group(3))


def run_inference(model_path: str, images_dir: str, output_csv: str,
                  conf_threshold: float = 0.25):
    model = YOLO(model_path)
    print(f"Loaded model: {model_path}")
    image_dir = Path(images_dir)
    image_files = sorted(image_dir.glob("*.png"))
    print(f"Found {len(image_files)} images in {images_dir}")

    records = []
    for img_path in image_files:
        img_id, lat, lon = parse_filename(img_path.name)
        if lat is None:
            print(f"  Skipping {img_path.name} — cannot parse coordinates")
            continue

        # Inference call with retina_masks
        results = model.predict(
            str(img_path),
            conf=conf_threshold,
            imgsz=IMG_PX,
            retina_masks=True,
            verbose=False,
        )
        mpp = meters_per_pixel(lat, ZOOM) / SCALE

        for result in results:
            if result.masks is None:
                continue
            for mask_data, box in zip(result.masks.xy, result.boxes):
                pts = mask_data  # shape (N, 2) in pixel coords
                if len(pts) < 3:
                    continue

                # Pixel polygon to geo-coordinates
                geo_pts = [
                    pixel_to_latlon(x, y, lat, lon, mpp, IMG_PX)
                    for x, y in pts
                ]
                poly = Polygon([(lo, la) for la, lo in geo_pts])
                if not poly.is_valid or poly.area == 0:
                    continue

                records.append({
                    "image_id":   img_id,
                    "geo_polygon": poly.wkt,
                    "confidence":  float(box.conf),
                    "class_id":    int(box.cls),
                })

    df = pd.DataFrame(records)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} segments → {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO tree crown inference")
    parser.add_argument("--model",  required=True, help="Path to trained YOLO weights (.pt)") # From transfer_learning best.pt
    parser.add_argument("--images", required=True, help="Directory containing PNG images")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--conf",   type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    run_inference(args.model, args.images, args.output, args.conf)