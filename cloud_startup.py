"""
cloud_startup.py
================
Called once at app startup (imported by app.py) to ensure data is available
on Streamlit Community Cloud where the repo starts fresh with no live data.

Flow
----
1. Check if processed CSV or static fallback exists
2. If neither exists → attempt live FPL API fetch
3. If API unreachable → raise a clear user-facing error
4. Log result to Streamlit (visible in cloud logs)

This is NOT needed locally if you run update_data.py manually first.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("cloud_startup")


def ensure_data_available() -> dict:
    """
    Guarantee that at least one player CSV is readable before the app loads.

    Returns
    -------
    dict  with keys: source ("live" | "static" | "already_present"), n_players
    """
    from data.loader import PROCESSED_PATH, STATIC_PATH

    # Case 1 — processed live data already on disk (normal case after first run)
    if PROCESSED_PATH.exists() and PROCESSED_PATH.stat().st_size > 5_000:
        logger.info("Processed CSV found (%s). No startup fetch needed.", PROCESSED_PATH)
        return {"source": "already_present", "path": str(PROCESSED_PATH)}

    # Case 2 — static fallback exists (committed CSV in repo)
    if STATIC_PATH.exists() and STATIC_PATH.stat().st_size > 5_000:
        logger.info("Static fallback CSV found (%s). No startup fetch needed.", STATIC_PATH)
        return {"source": "static", "path": str(STATIC_PATH)}

    # Case 3 — nothing on disk → try live API
    logger.info("No player data found. Attempting live FPL API fetch…")
    try:
        from data.fpl_api import FPLAPIClient
        client = FPLAPIClient(offline_fallback=False)
        df = client.fetch_and_save()
        logger.info("Startup fetch successful: %d players.", len(df))
        return {"source": "live", "n_players": len(df)}
    except Exception as exc:
        # All options exhausted — raise with a helpful message
        msg = (
            f"FPL data unavailable: {exc}\n\n"
            "To fix this:\n"
            "  1. Ensure data/players.csv is committed to your GitHub repo, OR\n"
            "  2. Ensure the FPL API is reachable (check https://fantasy.premierleague.com), OR\n"
            "  3. Re-deploy the app to trigger a fresh startup fetch."
        )
        logger.error(msg)
        raise RuntimeError(msg) from exc
