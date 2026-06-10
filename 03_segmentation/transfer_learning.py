"""
Transfer-learning (refinement) stage for YOLO11l-seg tree crown segmentation.

This is the SECOND training step, run AFTER the Optuna hyperparameter search
(optuna_yolo.py). It takes the best checkpoint from that search and fine-tunes
it on the expanded 350-image dataset to improve masks in dense canopy areas.

Settings follow the thesis (Table 7 "Retraining" / Table 11 "Transfer Learning"):
    - start from the previous best checkpoint (NOT a COCO-pretrained file)
    - freeze the first 10 layers
    - slower learning rate:  lr0 = 0.0001,  lrf = 0.00005
    - fewer epochs (~16) with low patience (5)  -> avoids overfitting
    - augmentation DISABLED
    - input resolution 1280 px

Note on "transfer learning" vs "resume":
    We LOAD the checkpoint weights and start a fresh, differently-configured
    training run (new LR, freezing, no augmentation). That is transfer learning.
    We do NOT use resume=True -- that would continue the old run with the OLD
    hyperparameters, which is not what we want here.

Paths:
    CHECKPOINT_PATH  - best.pt produced by the Optuna search
    DATA_PATH        - the 350-image dataset (Roboflow YOLOv11 export; auto-unzipped)
    PROJECT_DIR      - where the run is saved
"""

import gc
import logging
import os
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

# ── configuration ──────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "runs/segment/custom/best.pt"            # best weights from Optuna
DATA_PATH       = "dataset/google_masking_350.v1.yolov11.zip"  # 350-image dataset
PROJECT_DIR     = "runs/segment/transfer"
RUN_NAME        = "transfer_learning"

# Transfer-learning hyperparameters (thesis Table 7 "Retraining")
EPOCHS   = 16
PATIENCE = 5
FREEZE   = 10          # freeze first 10 layers
LR0      = 0.0001
LRF      = 0.00005
IMGSZ    = 1280

# OPTIONAL: carry over the loss/optimizer params from the best Optuna trial for
# continuity. Leave empty {} to use Ultralytics defaults. Example:
#   BEST_PARAMS = {"box": 1.7, "cls": 0.9, "dfl": 1.4, "momentum": 0.9, "weight_decay": 7e-4}
BEST_PARAMS = {}
# ───────────────────────────────────────────────────────────────────────────────

Path(PROJECT_DIR).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_DIR, "transfer_learning.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("yolo_transfer")


def train_transfer():
    torch.cuda.empty_cache()
    gc.collect()

    if not Path(CHECKPOINT_PATH).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. "
            "Point CHECKPOINT_PATH at the best.pt from the Optuna stage."
        )

    logger.info("Loading checkpoint: %s", CHECKPOINT_PATH)
    model = YOLO(CHECKPOINT_PATH)          # transfer learning: load weights

    params = {
        "data":     DATA_PATH,
        "epochs":   EPOCHS,
        "patience": PATIENCE,
        "freeze":   FREEZE,
        "lr0":      LR0,
        "lrf":      LRF,
        "imgsz":    IMGSZ,
        # augmentation disabled
        "hsv_h":      0.0,
        "hsv_s":      0.0,
        "hsv_v":      0.0,
        "degrees":    0.0,
        "translate":  0.0,
        "scale":      0.0,
        "shear":      0.0,
        "perspective":0.0,
        "flipud":     0.0,
        "fliplr":     0.0,
        "mosaic":     0.0,
        "mixup":      0.0,
        "copy_paste": 0.0,
        "erasing":    0.0,
        "optimizer":    "AdamW",
        "cos_lr":       True,
        "retina_masks": True,
        "val":          True,
        "split":        "val",
        "device":       0,
        "amp":          True,
        "project":      PROJECT_DIR,
        "name":         RUN_NAME,
        "exist_ok":     True,
        "verbose":      True,
    }
    params.update(BEST_PARAMS)

    logger.info("Starting transfer learning: epochs=%d freeze=%d lr0=%g lrf=%g imgsz=%d",
                EPOCHS, FREEZE, LR0, LRF, IMGSZ)

    results = model.train(**params)

    # final validation metrics
    try:
        p   = float(results.seg.mp)        # mean precision (mask)
        r   = float(results.seg.mr)        # mean recall (mask)
        m50 = float(results.seg.map50)     # mAP@0.5
        m95 = float(results.seg.map)       # mAP@0.5:0.95
        logger.info("Final  P=%.4f  R=%.4f  mAP50=%.4f  mAP50-95=%.4f", p, r, m50, m95)
    except Exception as exc:
        logger.warning("Could not read final metrics: %s", exc)

    # Define best weights path
    best_src = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
    best_dst = Path(PROJECT_DIR) / "transfer_best.pt"
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        logger.info("Best model saved -> %s", best_dst)
    else:
        logger.warning("Expected best weights not found at %s", best_src)

    del model, results
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Transfer-learning fine-tuning for YOLO11l-seg.")
    p.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="Best .pt from the Optuna stage")
    p.add_argument("--data",       default=DATA_PATH,       help="350-image dataset (Roboflow export)")
    p.add_argument("--project",    default=PROJECT_DIR,     help="Output project folder")
    p.add_argument("--epochs",     type=int,   default=EPOCHS)
    p.add_argument("--freeze",     type=int,   default=FREEZE)
    p.add_argument("--imgsz",      type=int,   default=IMGSZ)
    args = p.parse_args()

    # Apply any CLI overrides
    CHECKPOINT_PATH = args.checkpoint
    DATA_PATH       = args.data
    PROJECT_DIR     = args.project
    EPOCHS          = args.epochs
    FREEZE          = args.freeze
    IMGSZ           = args.imgsz

    logger.info("Transfer-learning stage starting…")
    train_transfer()
    logger.info("Done.")
