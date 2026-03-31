"""
FPL AI Optimizer - Data Update Pipeline
========================================
Master automation script that:

  1. Checks if a new gameweek is available (smart staleness detection)
  2. Fetches latest data from the FPL API
  3. Re-runs ML training (with model cache to avoid unnecessary retraining)
  4. Saves predictions

Run this script:
  python update_data.py                  # update only if data is stale
  python update_data.py --force          # force full refresh
  python update_data.py --check-only     # just check, don't update
  python update_data.py --schedule       # run on repeat (APScheduler)

Scheduling Options
------------------
Windows Task Scheduler:
  schtasks /create /tn "FPL Update" /tr "python C:\\path\\update_data.py" /sc DAILY /st 06:00

Linux/macOS cron (6 AM daily):
  0 6 * * * cd /path/to/fpl_optimizer && python update_data.py >> logs/cron.log 2>&1

Python scheduler (APScheduler):
  python update_data.py --schedule
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR  = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("update_data")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline steps
# ─────────────────────────────────────────────────────────────────────────────

def step_fetch(force: bool = False) -> dict:
    """
    Step 1 — Fetch latest data from the FPL API.
    Skips if data is fresh and force=False.
    """
    from data.fpl_api import FPLAPIClient

    client = FPLAPIClient(offline_fallback=True)

    # Staleness check
    if not force:
        if not client.is_data_stale(max_age_hours=6.0):
            meta = client.read_metadata()
            logger.info(
                "Data is fresh (fetched at %s, GW%s). Skipping fetch.",
                meta.get("fetched_at", "?"), meta.get("current_gw", "?"),
            )
            return {"status": "skipped", **meta}

    logger.info("Fetching latest FPL data…")
    try:
        df = client.fetch_and_save()
        meta = client.get_gameweek_metadata()
        logger.info(
            "Fetched %d players. Current GW: %s. Next GW deadline: %s",
            len(df), meta.get("current_gw"), meta.get("gw_deadline"),
        )
        return {"status": "fetched", "n_players": len(df), **meta}
    except Exception as exc:
        logger.error("Fetch failed: %s", exc, exc_info=True)
        return {"status": "error", "error": str(exc)}


def step_train(force: bool = False) -> dict:
    """
    Step 2 — Train ML models on latest data.
    Uses cached model if still fresh (< 24 h old) and data hasn't changed.
    """
    logger.info("Running ML training pipeline…")
    try:
        from models.train import train
        result = train(force=force)
        logger.info("Training result: %s", result.get("status"))
        return result
    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        return {"status": "error", "error": str(exc)}


def step_log(fetch_result: dict, train_result: dict) -> None:
    """Write a structured pipeline run log."""
    entry = {
        "run_at":       datetime.now(timezone.utc).isoformat(),
        "fetch":        fetch_result,
        "train":        train_result,
    }
    log_path = LOG_DIR / "pipeline_runs.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info("Pipeline run logged → %s", log_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(force: bool = False) -> dict:
    """
    Execute the full data → ML pipeline.

    Returns
    -------
    dict with 'fetch' and 'train' results
    """
    logger.info("=" * 60)
    logger.info("FPL AI Optimizer — Data Update Pipeline")
    logger.info("=" * 60)

    t_start = time.perf_counter()

    # Step 1: Fetch
    fetch_result = step_fetch(force=force)

    # Step 2: Train (always attempt unless fetch errored)
    if fetch_result.get("status") == "error":
        logger.warning("Skipping training due to fetch failure.")
        train_result = {"status": "skipped", "reason": "fetch_failed"}
    else:
        train_result = step_train(force=force)

    # Step 3: Log
    step_log(fetch_result, train_result)

    elapsed = time.perf_counter() - t_start
    logger.info("Pipeline complete in %.1fs.", elapsed)
    logger.info("=" * 60)

    return {"fetch": fetch_result, "train": train_result, "elapsed_sec": round(elapsed, 2)}


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler (APScheduler)
# ─────────────────────────────────────────────────────────────────────────────

def run_scheduler(interval_hours: float = 6.0) -> None:
    """
    Run the pipeline on a repeating schedule using APScheduler.
    Ideal for a long-running server process.

    Falls back to a simple time.sleep loop if APScheduler is not installed.
    """
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.interval import IntervalTrigger

        scheduler = BlockingScheduler(timezone="UTC")
        scheduler.add_job(
            run_pipeline,
            trigger=IntervalTrigger(hours=interval_hours),
            id="fpl_update",
            replace_existing=True,
            misfire_grace_time=300,
        )
        logger.info("APScheduler started — running every %.1f hours.", interval_hours)
        logger.info("Press Ctrl+C to stop.")

        # Run immediately on first start
        run_pipeline(force=False)

        scheduler.start()

    except ImportError:
        logger.warning(
            "APScheduler not installed. Falling back to simple loop. "
            "Install with: pip install apscheduler"
        )
        _simple_loop(interval_hours)


def _simple_loop(interval_hours: float) -> None:
    """Minimal scheduler: sleep between runs."""
    logger.info("Simple scheduler: running every %.1f hours.", interval_hours)
    while True:
        try:
            run_pipeline(force=False)
        except Exception as exc:
            logger.error("Pipeline error: %s", exc, exc_info=True)
        sleep_sec = interval_hours * 3600
        logger.info("Sleeping %.0f seconds until next run…", sleep_sec)
        time.sleep(sleep_sec)


# ─────────────────────────────────────────────────────────────────────────────
# Status check
# ─────────────────────────────────────────────────────────────────────────────

def check_status() -> None:
    """Print current pipeline status without running anything."""
    from data.fpl_api import FPLAPIClient
    from models.model_store import read_model_meta

    api_meta   = FPLAPIClient.read_metadata()
    model_meta = read_model_meta()
    stale      = FPLAPIClient.is_data_stale(max_age_hours=6.0)

    print("\n── FPL AI Optimizer — Pipeline Status ──────────────────────")
    print(f"  Data last fetched : {api_meta.get('fetched_at', 'Never')}")
    print(f"  Current gameweek  : GW{api_meta.get('current_gw', '?')}")
    print(f"  Next GW deadline  : {api_meta.get('gw_deadline', 'Unknown')}")
    print(f"  Total players     : {api_meta.get('n_players', 'Unknown')}")
    print(f"  Data stale?       : {'YES — update needed' if stale else 'No'}")
    print()
    print(f"  Model trained at  : {model_meta.get('trained_at', 'Never')}")
    print(f"  Best model        : {model_meta.get('best_model', 'Unknown')}")
    print(f"  Trained on        : {model_meta.get('n_players', '?')} players")
    print("────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FPL AI Optimizer — Data & ML Update Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force full refresh regardless of staleness",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Print status and exit without running pipeline",
    )
    parser.add_argument(
        "--schedule", action="store_true",
        help="Run on repeat using APScheduler (or simple loop)",
    )
    parser.add_argument(
        "--interval", type=float, default=6.0,
        help="Scheduler interval in hours (default: 6)",
    )
    args = parser.parse_args()

    if args.check_only:
        check_status()
        sys.exit(0)

    if args.schedule:
        run_scheduler(interval_hours=args.interval)
    else:
        result = run_pipeline(force=args.force)
        success = all(
            r.get("status") in ("fetched", "trained", "skipped")
            for r in [result.get("fetch", {}), result.get("train", {})]
        )
        sys.exit(0 if success else 1)
