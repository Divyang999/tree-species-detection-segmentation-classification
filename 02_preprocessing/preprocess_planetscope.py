"""
Preprocess PlanetScope 8-band PSB.SD SuperDove imagery.

Steps:
  1. Convert DN to surface reflectance (divide by 10000)
  2. Apply UDM2 cloud/shadow mask (exclude pixels with confidence < threshold)
  3. Save masked float32 GeoTIFF

Usage:
    python preprocess_planetscope.py --input scene.tif --udm2 scene_udm2.tif --output masked.tif
"""

import argparse
import numpy as np
import rasterio
from pathlib import Path

# UDM2 band indices (0-based)
UDM2_CLEAR      = 0
UDM2_SNOW       = 1
UDM2_SHADOW     = 2
UDM2_LT_HAZE    = 3
UDM2_HT_HAZE    = 4
UDM2_CLOUD      = 5
UDM2_CONFIDENCE = 6

BAND_NAMES = [
    "CoastalBlue", "Blue", "GreenI", "Green",
    "Yellow", "Red", "RedEdge", "NIR"
]


def load_reflectance(image_path: Path):
    """Read image and convert DN → surface reflectance [0, 1]."""
    with rasterio.open(image_path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile.copy()
    data /= 10000.0
    if np.nanmin(data) < 0 or np.nanmax(data) > 1:
        print(f"  Warning: values outside [0,1] after scaling in {image_path.name}")
    return data, profile


def apply_udm2_mask(image_data: np.ndarray, udm2_path: Path,
                    confidence_threshold: int = 75) -> np.ndarray:
    """
    Mask pixels flagged as cloud / shadow / haze in the UDM2 file.
    Unusable pixels are set to NaN.
    """
    with rasterio.open(udm2_path) as src:
        udm2 = src.read()

    if udm2.shape[1:] != image_data.shape[1:]:
        raise ValueError("Image and UDM2 spatial dimensions do not match.")

    clear_mask      = udm2[UDM2_CLEAR]      == 1
    confidence_mask = udm2[UDM2_CONFIDENCE] >= confidence_threshold
    snow_mask       = udm2[UDM2_SNOW]       == 0
    shadow_mask     = udm2[UDM2_SHADOW]     == 0
    cloud_mask      = udm2[UDM2_CLOUD]      == 0
    haze_mask       = (udm2[UDM2_LT_HAZE] == 0) & (udm2[UDM2_HT_HAZE] == 0)

    usable = clear_mask & confidence_mask & snow_mask & shadow_mask & cloud_mask & haze_mask

    usable_pct = usable.mean() * 100
    print(f"  Usable pixels: {usable_pct:.1f}%")

    masked = image_data.copy()
    for i in range(image_data.shape[0]):
        masked[i] = np.where(usable, image_data[i], np.nan)
    return masked


def save_image(data: np.ndarray, profile: dict, output_path: Path):
    """Write float32 GeoTIFF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype=rasterio.float32, nodata=np.nan)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data.astype(np.float32))
    print(f"  Saved → {output_path}")


def preprocess(image_path: Path, udm2_path: Path, output_path: Path,
               confidence_threshold: int = 75):
    print(f"Processing {image_path.name}...")
    data, profile = load_reflectance(image_path)
    data = apply_udm2_mask(data, udm2_path, confidence_threshold)
    save_image(data, profile, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PlanetScope imagery")
    parser.add_argument("--input",  required=True, help="Input 8-band TIF")
    parser.add_argument("--udm2",   required=True, help="UDM2 mask TIF")
    parser.add_argument("--output", required=True, help="Output masked TIF")
    parser.add_argument("--confidence", type=int, default=75,
                        help="UDM2 confidence threshold (default 75)")
    args = parser.parse_args()

    preprocess(
        Path(args.input), Path(args.udm2), Path(args.output),
        confidence_threshold=args.confidence,
    )