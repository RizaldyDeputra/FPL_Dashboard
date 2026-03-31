"""
FPL AI Optimizer - Model Training Script
=========================================
Trains the ML models on the latest player data and saves them to disk.
Called by update_data.py after each data refresh.

Usage
-----
  python models/train.py            # train on available data
  python models/train.py --force    # force retrain even if model is fresh
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make project root importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.loader import load_and_prepare, get_feature_columns
from models.predictor import run_ml_pipeline
from models.model_store import save_predictor, model_is_fresh, read_model_meta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")

PREDICTIONS_DIR = ROOT / "data" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_PATH = PREDICTIONS_DIR / "latest.csv"


def train(force: bool = False, min_minutes: int = 0) -> dict:
    """
    Run full training pipeline.

    Parameters
    ----------
    force       : retrain even if the cached model is still fresh
    min_minutes : filter players below this minutes threshold before training

    Returns
    -------
    dict with training results / metadata
    """
    start = time.perf_counter()

    # ── Check if retraining is needed ─────────────────────────────────────────
    if not force and model_is_fresh():
        meta = read_model_meta()
        logger.info(
            "Skipping retraining — cached model is fresh (trained at %s, model=%s).",
            meta.get("trained_at", "?"), meta.get("best_model", "?"),
        )
        return {"status": "skipped", "reason": "model_fresh", **meta}

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading player data…")
    df = load_and_prepare(prefer_live=True, min_minutes=min_minutes)
    logger.info("Loaded %d players.", len(df))

    if len(df) < 50:
        logger.error("Too few players (%d) to train reliably.", len(df))
        return {"status": "error", "reason": "insufficient_data", "n_players": len(df)}

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Training ML models…")
    feat_cols = get_feature_columns()
    df_with_preds, predictor, metrics = run_ml_pipeline(df, feat_cols)
    logger.info("Training complete. Metrics: %s", metrics)
    logger.info("Best model: %s", predictor.best_name)

    # ── Save model ────────────────────────────────────────────────────────────
    save_predictor(predictor, df)
    logger.info("Model saved.")

    # ── Save predictions CSV ──────────────────────────────────────────────────
    pred_cols = [
        "player_name", "position", "team", "cost", "total_points",
        "minutes", "goals_scored", "assists", "ict_index",
        "selected_by_percent", "predicted_points", "form_score",
        "value_score", "pts_per_90", "status",
    ]
    available_cols = [c for c in pred_cols if c in df_with_preds.columns]
    df_with_preds[available_cols].to_csv(PREDICTIONS_PATH, index=False)
    logger.info("Predictions saved → %s", PREDICTIONS_PATH)

    elapsed = time.perf_counter() - start
    result = {
        "status":      "trained",
        "best_model":  predictor.best_name,
        "metrics":     metrics,
        "n_players":   len(df),
        "elapsed_sec": round(elapsed, 2),
        "trained_at":  datetime.now(timezone.utc).isoformat(),
    }

    # Write training log entry
    _append_training_log(result)

    logger.info("Training pipeline done in %.1fs.", elapsed)
    return result


def _append_training_log(result: dict) -> None:
    """Append one line to the training log JSON-Lines file."""
    log_path = ROOT / "logs" / "training.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FPL AI models")
    parser.add_argument("--force", action="store_true", help="Force retraining")
    parser.add_argument("--min-minutes", type=int, default=0, help="Min minutes filter")
    args = parser.parse_args()

    result = train(force=args.force, min_minutes=args.min_minutes)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get("status") in ("trained", "skipped") else 1)
