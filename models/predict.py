"""
FPL AI Optimizer - Fast Predictor
===================================
Generates predictions using the cached trained model.
Falls back to retraining if no valid cache is found.

Usage
-----
  python models/predict.py             # predict on latest data
  python models/predict.py --output predictions/custom.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.loader import load_and_prepare, get_feature_columns
from models.predictor import run_ml_pipeline, FPLPredictor
from models.model_store import load_predictor, save_predictor

logger = logging.getLogger("predict")

PREDICTIONS_PATH = ROOT / "data" / "predictions" / "latest.csv"


def predict(
    df=None,
    min_minutes: int = 0,
    force_retrain: bool = False,
) -> tuple:
    """
    Generate predictions for all players.

    Parameters
    ----------
    df            : pre-loaded DataFrame (optional; loads from disk if None)
    min_minutes   : filter threshold
    force_retrain : skip cache and retrain from scratch

    Returns
    -------
    (df_with_predictions, predictor, metrics_dict)
    """
    if df is None:
        df = load_and_prepare(prefer_live=True, min_minutes=min_minutes)

    feat_cols = get_feature_columns()

    # Try to load cached model
    cached = None if force_retrain else load_predictor(df)

    if cached is not None:
        logger.info("Using cached model: %s", cached.best_name)
        df = df.copy()
        df["predicted_points"] = cached.predict(df).clip(0)
        return df, cached, cached.metrics

    # No valid cache — train fresh
    logger.info("Training fresh model…")
    df_out, predictor, metrics = run_ml_pipeline(df, feat_cols)
    save_predictor(predictor, df)
    return df_out, predictor, metrics


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Generate FPL predictions")
    parser.add_argument("--min-minutes", type=int, default=0)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--output", type=str, default=str(PREDICTIONS_PATH))
    args = parser.parse_args()

    df_pred, pred, met = predict(min_minutes=args.min_minutes, force_retrain=args.force_retrain)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(out_path, index=False)
    print(f"Predictions saved → {out_path}")
    print(f"Best model: {pred.best_name}")
    print(json.dumps(met, indent=2))
