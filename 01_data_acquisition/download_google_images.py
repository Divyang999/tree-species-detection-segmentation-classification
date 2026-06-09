"""
Download satellite images from Google Maps Static API for a grid of tree locations.

Usage:
    Set GOOGLE_API_KEY environment variable, then:
    python download_google_images.py --csv grid_centers.csv --output ./images/
"""

import os
import argparse
import requests
import pandas as pd


def download_satellite_image(lat, lng, image_name, output_folder,
                              zoom=19, scale=2, image_size=(1280, 1280), api_key=""):
    """Download a single satellite image centred on (lat, lng) via Google Maps Static API."""
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lng}",
        "zoom": zoom,
        "size": f"{image_size[0]}x{image_size[1]}",
        "scale": scale,
        "maptype": "satellite",
        "key": api_key,
    }

    response = requests.get(base_url, params=params, stream=True)
    if response.status_code == 200:
        os.makedirs(output_folder, exist_ok=True)
        image_path = os.path.join(output_folder, image_name)
        with open(image_path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        return True, f"Saved: {image_path}"
    return False, f"HTTP {response.status_code}: {response.text}"


def bulk_download_from_csv(csv_file, output_folder, api_key, zoom=19, scale=2,
                            image_size=(1280, 1280)):
    """
    Read grid centres from CSV and download satellite images for each point.

    Expected CSV columns: Grid_number, Center_lat, Center_long
    Output filename format: {Grid_number}_{lat}_{lng}.png
    """
    try:
        data = pd.read_csv(csv_file)
        required = ["Grid_number", "Center_lat", "Center_long"]
        if not all(c in data.columns for c in required):
            raise ValueError(f"CSV must contain columns: {required}")
    except FileNotFoundError:
        print(f"Error: CSV not found at {csv_file}")
        return

    total = len(data)
    downloaded = failed = 0

    for _, row in data.iterrows():
        grid_num = int(row["Grid_number"])
        lat, lng = row["Center_lat"], row["Center_long"]
        name = f"{grid_num}_{lat}_{lng}.png"

        ok, msg = download_satellite_image(
            lat, lng, name, output_folder, zoom, scale, image_size, api_key
        )
        if ok:
            downloaded += 1
        else:
            failed += 1
            print(f"  FAILED {name}: {msg}")

        if (downloaded + failed) % 10 == 0 or (downloaded + failed) == total:
            print(f"\rProgress: {downloaded}/{total} downloaded, {failed} failed", end="", flush=True)

    print(f"\n\nComplete — downloaded: {downloaded}, failed: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk download Google satellite images")
    parser.add_argument("--csv", required=True, help="Path to grid_centers CSV")
    parser.add_argument("--output", required=True, help="Output folder for images")
    parser.add_argument("--zoom", type=int, default=19, help="Zoom level (default 19)")
    parser.add_argument("--scale", type=int, default=2, help="Scale factor (default 2)")
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise EnvironmentError("Set the GOOGLE_API_KEY environment variable before running.")

    bulk_download_from_csv(args.csv, args.output, api_key, zoom=args.zoom, scale=args.scale)
