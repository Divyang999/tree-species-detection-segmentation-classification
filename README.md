# Individual Tree Classification using ML and High-Resolution Satellite Imagery

**IIT Roorkee — Thesis (May 2025)**  
Author: Divyang Raj Verma  
Supervisor: Emeritus Fellow Dr. P.K. Garg, Geospatial Engineering Group, Dept. of Civil Engineering

---

## Overview

Full pipeline for classifying individual urban trees by species family using:

- **Segmentation**: YOLOv11l-seg trained on Google Maps Static imagery (Bangalore study area)
- **Imagery**: PlanetScope 8-band SuperDove multispectral imagery (3 m resolution, Bangalore AOI)
- **Ground truth**: Bengaluru Municipal Tree Census (~50,000 raw points → 14,224 retained after filtering out flowering plants and small canopies; 6,322 usable after polygon matching)
- **Classifier**: Random Forest with SMOTE oversampling and PCA dimensionality reduction

**Best result**: 4-class SMOTE + PCA  
Train / Val / Test accuracy: **70.1 % / 65.1 % / 62.5 %** — Cohen's Kappa: **0.80**

### Tree Families Classified

Sample counts below are the final labeled set used for classification (6,322 total)
after all the filtering and pre-processing steps.

| Family | Common Examples | Samples |
|---|---|---|
| Fabaceae | Indian Rosewood, Acacia, Tamarind, Albizia | 2,604 |
| Meliaceae | Neem, Mahogany | 964 |
| Arecaceae | Coconut Palm, Areca Palm | 933 |
| Bignoniaceae | Trumpet Tree, Jacaranda | 753 |
| Moraceae | Jackfruit, Peepal, Banyan | 385 |
| Combretaceae | Indian Almond, Arjuna | 354 |
| Anacardiaceae | Mango, Indian Hog Plum | 199 |

The **4-class** subset uses the top four families (Fabaceae, Meliaceae,
Arecaceae, Bignoniaceae; 5,254 samples). The **7-class** subset adds
Combretaceae, Moraceae, and Anacardiaceae.

---

## Pipeline

```
01_data_acquisition/          Google Maps Static API → Maxar satellite PNGs
02_preprocessing/             PlanetScope UDM2 masking + vegetation indices
03_segmentation/              YOLO11l-seg training (Optuna) + inference
04_postprocessing/            IoU merge, confidence/area filters, convex-hull smoothing, ground truth merge
05_feature_extraction/        Zonal stats (mean/std/median) + GLCM texture
06_classification/            Dataset preprocessing, RF training, inference
07_analysis/                  Confusion matrices, F1 charts, feature importance
```

### Feature Set (384 features total)

| Group | Details | Count |
|---|---|---|
| Spectral zonal stats | 8 bands × 5 months × 3 stats (mean, std, median) | 120 |
| Index zonal stats | 6 indices × 5 months × 3 stats | 90 |
| GLCM texture | 2 bands (Red, NIR) × 3 properties (Dec 2024) | 6 |
| Temporal differences | 4 intervals × 14 features × 3 stats | 168 |
| **Total** | | **384** |

**Vegetation indices**: NDVI, NDVI2, NDVI-RE (red-edge NDVI), EVI, SAVI, GNDVI  
**GLCM properties**: Contrast, Homogeneity, Correlation  
**Date suffixes**: May2024, Sept2024, Dec2024, Jan2025, Feb2025

---

## Installation

```bash
git clone https://github.com/<your-username>/individual-tree-classification.git
cd individual-tree-classification
pip install -r requirements.txt
```

> **GPU note**: YOLO training requires CUDA. The classification pipeline runs on CPU.

---

## Usage

### 1 — Download satellite images (Bangalore)

```bash
export GOOGLE_API_KEY="your_key_here"
python 01_data_acquisition/download_google_images.py \
    --csv   data/grid_centers.csv \
    --output data/satellite_images/
```

### 2 — Preprocess PlanetScope imagery

```bash
# For each date (May2024, Sept2024, Dec2024, Jan2025, Feb2025):
python 02_preprocessing/preprocess_planetscope.py \
    --input  data/rasters/May2024_raw.tif \
    --udm2   data/rasters/May2024_udm2.tif \
    --output data/rasters/May2024_8b.tif

python 02_preprocessing/compute_vegetation_indices.py \
    --input  data/rasters/May2024_8b.tif \
    --outdir data/indices/May2024/
```

### 3 — Train YOLO segmentation model

```bash
# Hyperparameter search (GPU server, runs ~100 trials)
python 03_segmentation/optuna_yolo.py

# Run inference to get tree crown polygons
python 03_segmentation/inference_yolo.py \
    --model  runs/segment/custom/best.pt \
    --images data/satellite_images/ \
    --output data/raw_segments.csv
```

### 4 — Post-process segments and merge ground truth

```bash
python 04_postprocessing/polygon_postprocess.py \
    --input  data/raw_segments.csv \
    --output data/processed_segments.csv

python 04_postprocessing/merge_ground_segments.py \
    --polygons data/processed_segments.csv \
    --trees    data/ground_truth_trees.csv \
    --output   data/merged_output.gpkg
```

### 5 — Extract features

```bash
# Edit 05_feature_extraction/feature_config.yaml with your raster paths
python 05_feature_extraction/zonal_statistics_glcm.py \
    --config 05_feature_extraction/feature_config.yaml
```

### 6 — Train classifier

```bash
python 06_classification/preprocess_dataset.py \
    --input    data/zonal_stats.gpkg \
    --output   data/preprocessed.gpkg \
    --label-map data/family_label_mapping.csv

python 06_classification/train_random_forest.py \
    --input  data/preprocessed.gpkg \
    --outdir model_outputs/
```

### 7 — Run full inference

```bash
python 06_classification/predict.py \
    --input       data/preprocessed.gpkg \
    --model       model_outputs/4_class_smote/pca/model_4_class_smote_pca.joblib \
    --scaler      model_outputs/4_class_smote/pca/scaler_4_class_smote_pca.joblib \
    --pca         model_outputs/4_class_smote/pca/pca_4_class_smote_pca.joblib \
    --output      inference/final_predictions.gpkg \
    --combination pca
```

### 8 — Visualise results

```bash
python 07_analysis/visualize_results.py \
    --metrics model_outputs/all_metrics.csv \
    --gpkg    model_outputs/4_class_smote/pca/test_predictions_4_class_smote_pca.gpkg \
    --outdir  results/plots/
```

---

## Key Results

| Dataset | Combination | Test Acc | Test F1 | Kappa |
|---|---|---|---|---|
| 4-class SMOTE | **PCA** | **0.625** | **0.687** | **0.801** |
| 4-class SMOTE | Spectral+Temporal+GLCM | 0.609 | 0.611 | 0.790 |
| 4-class SMOTE | Spectral+Temporal | 0.618 | 0.619 | 0.783 |
| 7-class SMOTE | PCA | 0.515 | 0.526 | 0.751 |

---

## Repository Structure

```
individual-tree-classification/
├── README.md
├── requirements.txt
├── 01_data_acquisition/
│   └── download_google_images.py
├── 02_preprocessing/
│   ├── preprocess_planetscope.py
│   └── compute_vegetation_indices.py
├── 03_segmentation/
│   ├── optuna_yolo.py
│   └── inference_yolo.py
├── 04_postprocessing/
│   ├── polygon_postprocess.py
│   └── merge_ground_segments.py
├── 05_feature_extraction/
│   ├── zonal_statistics_glcm.py
│   └── feature_config.yaml
├── 06_classification/
│   ├── preprocess_dataset.py
│   ├── train_random_forest.py
│   └── predict.py
└── 07_analysis/
    └── visualize_results.py
```

---

## Data Sources

- **Satellite imagery**: [PlanetScope](https://www.planet.com/) SuperDove PSB.SD (8-band, 3 m)
- **Google basemap**: Google Maps Static API (zoom 19, scale 2 → 0.30 m/pixel nominal, ~0.15 m/pixel effective)
- **Ground truth**: Bangalore Municipal Tree Census (OpenCity Portal), KGIS shapefiles
- **Segmentation training data**: Manually annotated via [Roboflow](https://roboflow.com/) — 10,287 trees across 350 images (300 initial + 50 added for transfer learning)

---

## Citation

If you use this code, please cite:

```
Divyang Raj Verma, "Individual Tree Classification using ML and
High-Resolution Satellite Imagery," M.Tech Thesis,
IIT Roorkee, May 2025.
```
