"""
Hyperparameter optimisation for YOLO11l-seg tree crown segmentation using Optuna.

Optimises: lr0, lrf, momentum, weight_decay, box/cls/dfl losses,
           augmentation params (mosaic, mixup, degrees, …), batch size.

Objective: maximise (precision + recall) on the validation segmentation mask.
Early stopping: if both precision > 0.75 and recall > 0.75 after ≥10 trials.

Usage (on a GPU server):
    nohup python optuna_yolo.py > output.log 2>&1 &

Paths to configure:
    MODEL_PATH  — pretrained YOLO11l-seg weights
    DATA_PATH   — dataset YAML or zip
    PROJECT_DIR — where training runs are saved
    STUDY_DB    — SQLite file for Optuna persistence
"""

import gc
import logging
import os
from pathlib import Path

import optuna
import torch
from ultralytics import YOLO

# ── configuration ──────────────────────────────────────────────────────────────
MODEL_PATH  = "yolo11l-seg.pt"                          # pretrained weights
DATA_PATH   = "dataset/google_masking.v9.yolov11.zip"   # dataset
PROJECT_DIR = "runs/segment/custom"
STUDY_DB    = "sqlite:///optuna_yolo.db"
N_TRIALS    = 100
EPOCHS      = 150
# ───────────────────────────────────────────────────────────────────────────────

Path(PROJECT_DIR).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_DIR, "optuna_study.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("yolo_optuna")

study = optuna.create_study(
    direction="maximize",
    study_name="yolo_optuna",
    storage=STUDY_DB,
    load_if_exists=True,
)


def objective(trial: optuna.Trial) -> float:
    params = {
        "lr0":          trial.suggest_float("lr0",          1e-4, 1e-3),
        "lrf":          trial.suggest_float("lrf",          1e-4, 1e-3),
        "momentum":     trial.suggest_float("momentum",     0.80, 0.95),
        "weight_decay": trial.suggest_float("weight_decay", 5e-4, 1e-3),
        "box":          trial.suggest_float("box",          0.5,  3.0),
        "cls":          trial.suggest_float("cls",          0.5,  3.0),
        "dfl":          trial.suggest_float("dfl",          1.0,  2.0),
        "degrees":      trial.suggest_float("degrees",      0.0, 20.0),
        "translate":    trial.suggest_float("translate",    0.0,  0.2),
        "scale":        trial.suggest_float("scale",        0.3,  0.7),
        "shear":        trial.suggest_float("shear",        0.0, 10.0),
        "mosaic":       trial.suggest_float("mosaic",       0.0,  1.0),
        "mixup":        trial.suggest_float("mixup",        0.0,  0.3),
        "copy_paste":   trial.suggest_float("copy_paste",   0.0,  0.2),
        "batch":        trial.suggest_int("batch",           8,   32),
        "patience":     trial.suggest_int("patience",       20,   40),
        "iou":          trial.suggest_float("iou",          0.5,  0.8),
        "workers":      trial.suggest_int("workers",         2,    4),
        # fixed
        "epochs":       EPOCHS,
        "imgsz":        640,
        "cos_lr":       True,
        "val":          True,
        "split":        "val",
        "retina_masks": True,
        "optimizer":    "AdamW",
        "device":       0,
        "project":      PROJECT_DIR,
        "name":         f"trial_{trial.number}",
        "exist_ok":     True,
        "pretrained":   True,
        "verbose":      False,
        "amp":          True,
    }

    logger.info("Trial %d — batch=%d", trial.number, params["batch"])
    torch.cuda.empty_cache()
    gc.collect()

    try:
        model = YOLO(MODEL_PATH)

        # Reduce batch size automatically on OOM
        while True:
            try:
                results = model.train(data=DATA_PATH, **params)
                break
            except torch.cuda.OutOfMemoryError:
                if params["batch"] <= 8:
                    logger.error("Trial %d: OOM at minimum batch size", trial.number)
                    return float("-inf")
                params["batch"] = max(8, params["batch"] // 2)
                logger.warning("OOM — reduced batch to %d", params["batch"])
                torch.cuda.empty_cache()
                gc.collect()

        precision = float(results.seg.p)
        recall    = float(results.seg.r)
        value     = precision + recall

        trial.report(value, step=EPOCHS)
        trial.set_user_attr("precision", precision)
        trial.set_user_attr("recall",    recall)

        if trial.should_prune():
            raise optuna.TrialPruned()

        logger.info("Trial %d — P=%.4f  R=%.4f", trial.number, precision, recall)
        del model, results
        torch.cuda.empty_cache()
        gc.collect()
        return value

    except optuna.TrialPruned:
        logger.info("Trial %d pruned", trial.number)
        raise
    except Exception as exc:
        logger.error("Trial %d failed: %s", trial.number, exc)
        torch.cuda.empty_cache()
        gc.collect()
        return float("-inf")


def _early_stop_callback(study: optuna.Study, trial: optuna.FrozenTrial):
    p = trial.user_attrs.get("precision", 0.0)
    r = trial.user_attrs.get("recall",    0.0)
    n = len(study.trials)
    if p > 0.75 and r > 0.75 and n >= 10:
        logger.info("Early stop — P=%.4f R=%.4f after %d trials", p, r, n)
        study.stop()


if __name__ == "__main__":
    logger.info("Starting Optuna study…")
    torch.cuda.empty_cache()
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        callbacks=[_early_stop_callback],
        gc_after_trial=True,
    )
    best = study.best_trial
    logger.info("Best trial %d — P=%.4f  R=%.4f", best.number,
                best.user_attrs.get("precision", float("nan")),
                best.user_attrs.get("recall",    float("nan")))
    logger.info("Best params: %s", best.params)
