"""
Saves and loads trained FPLPredictor instances using pickle,
avoiding expensive retraining on every dashboard refresh.

Cache invalidation strategy
---------------------------
  A model is considered fresh if:
    - The model file exists
    - It was trained within MAX_MODEL_AGE_HOURS
    - The data it was trained on has not been replaced since
      (detected by comparing CSV modification timestamps)
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("model_store")

_HERE       = Path(__file__).parent
SAVED_DIR   = _HERE / "saved"
SAVED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH    = SAVED_DIR / "predictor.pkl"
META_PATH     = SAVED_DIR / "model_meta.json"
MAX_MODEL_AGE_HOURS = 24.0   # retrain if older than this


# ─────────────────────────────────────────────────────────────────────────────
# Save / load
# ─────────────────────────────────────────────────────────────────────────────

def save_predictor(predictor, df: pd.DataFrame) -> None:
    """Persist a trained FPLPredictor to disk along with metadata."""
    import json

    payload = {
        "predictor": predictor,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_players": len(df),
        "best_model": predictor.best_name,
        "data_hash": _df_hash(df),
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {k: v for k, v in payload.items() if k != "predictor"}
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Saved predictor → %s (model=%s)", MODEL_PATH, predictor.best_name)


def load_predictor(df: pd.DataFrame):
    """
    Load the cached predictor if it is still fresh for the given df.
    Returns the predictor object or None if cache is stale/missing.
    """
    if not MODEL_PATH.exists():
        logger.info("No cached model found.")
        return None

    try:
        with open(MODEL_PATH, "rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        logger.warning("Failed to load model cache: %s", exc)
        return None

    trained_at_str = payload.get("trained_at")
    if not trained_at_str:
        return None

    trained_at = datetime.fromisoformat(trained_at_str)
    if trained_at.tzinfo is None:
        trained_at = trained_at.replace(tzinfo=timezone.utc)

    age_hours = (datetime.now(timezone.utc) - trained_at).total_seconds() / 3600

    if age_hours > MAX_MODEL_AGE_HOURS:
        logger.info("Cached model is %.1f hours old (max %s) → retraining.", age_hours, MAX_MODEL_AGE_HOURS)
        return None

    if payload.get("data_hash") != _df_hash(df):
        logger.info("Data changed since last training → retraining.")
        return None

    logger.info(
        "Using cached model '%s' (%.1fh old, %d players).",
        payload.get("best_model"), age_hours, payload.get("n_players", 0),
    )
    return payload["predictor"]


def model_is_fresh() -> bool:
    """Quick check without loading the full pickle."""
    import json
    if not META_PATH.exists():
        return False
    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        trained_at = datetime.fromisoformat(meta["trained_at"])
        if trained_at.tzinfo is None:
            trained_at = trained_at.replace(tzinfo=timezone.utc)
        age_h = (datetime.now(timezone.utc) - trained_at).total_seconds() / 3600
        return age_h < MAX_MODEL_AGE_HOURS
    except Exception:
        return False


def read_model_meta() -> dict:
    """Return model metadata dict (no pickle loading)."""
    import json
    if META_PATH.exists():
        try:
            return json.loads(META_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Hash helper
# ─────────────────────────────────────────────────────────────────────────────

def _df_hash(df: pd.DataFrame) -> str:
    """Compute a quick hash of the DataFrame to detect data changes."""
    try:
        key = f"{len(df)}-{list(df.columns)}-{df['total_points'].sum():.0f}"
        return hashlib.md5(key.encode()).hexdigest()[:12]
    except Exception:
        return "unknown"
