"""
FPL AI Optimizer - Live API Data Fetcher
========================================
Pulls the latest Fantasy Premier League data from the official FPL API
and converts it into a clean pandas DataFrame ready for the ML pipeline.

API Endpoints Used
------------------
  bootstrap-static  : players, teams, gameweek metadata
  event/{gw}/live   : live gameweek scores (bonus, stats)

Usage
-----
  from data.fpl_api import FPLAPIClient
  client = FPLAPIClient()
  df = client.fetch_and_save()          # fetch + persist
  meta = client.get_gameweek_metadata() # current GW info
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ─────────────────────────────────────────────────────────────────────────────
# Paths — cloud-safe
# ─────────────────────────────────────────────────────────────────────────────
# On Streamlit Community Cloud the repo root is read-only after startup.
# Set FPL_DATA_DIR env var to redirect writes to a writable location.
# Default: paths relative to this file (works locally and with GitHub Actions).

import os as _os

_HERE = Path(__file__).parent
_DATA_ROOT = Path(_os.environ.get("FPL_DATA_DIR", str(_HERE)))

RAW_DIR       = _DATA_ROOT / "raw"
PROCESSED_DIR = _DATA_ROOT / "processed"
CACHE_DIR     = _DATA_ROOT / "cache"

for _d in (RAW_DIR, PROCESSED_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_BOOTSTRAP_PATH  = RAW_DIR / "bootstrap_static.json"
RAW_GW_PATH         = RAW_DIR / "gw_live_{gw}.json"
PROCESSED_CSV_PATH  = PROCESSED_DIR / "players.csv"
METADATA_PATH       = CACHE_DIR / "metadata.json"

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("fpl_api")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
BASE_URL       = "https://fantasy.premierleague.com/api"
BOOTSTRAP_URL  = f"{BASE_URL}/bootstrap-static/"
GW_LIVE_URL    = f"{BASE_URL}/event/{{gw}}/live/"
FIXTURES_URL   = f"{BASE_URL}/fixtures/"

POSITION_MAP   = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

REQUEST_TIMEOUT   = 15   # seconds
REQUEST_HEADERS   = {
    "User-Agent": "FPL-AI-Optimizer/3.0 (portfolio project)",
    "Accept": "application/json",
}

# ─────────────────────────────────────────────────────────────────────────────
# HTTP session with retry logic
# ─────────────────────────────────────────────────────────────────────────────

def _build_session() -> requests.Session:
    """Create a requests session with exponential-backoff retry."""
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=1.5,          # 0s, 1.5s, 3s, 6s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    session.headers.update(REQUEST_HEADERS)
    return session


# ─────────────────────────────────────────────────────────────────────────────
# API Client
# ─────────────────────────────────────────────────────────────────────────────

class FPLAPIClient:
    """
    Fetches, validates, and persists FPL data.

    Parameters
    ----------
    offline_fallback : bool
        If True, use cached raw JSON when the API is unreachable.
    """

    def __init__(self, offline_fallback: bool = True):
        self.offline_fallback = offline_fallback
        self._session = _build_session()
        self._bootstrap: dict[str, Any] = {}

    # ── Low-level fetch ───────────────────────────────────────────────────────

    def _get(self, url: str, cache_path: Path | None = None) -> dict:
        """
        GET a URL and return parsed JSON.
        Falls back to cached file if the request fails and offline_fallback=True.
        """
        try:
            resp = self._session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            if cache_path:
                cache_path.write_text(json.dumps(data), encoding="utf-8")
            logger.debug("Fetched %s → %d bytes", url, len(resp.content))
            return data

        except Exception as exc:
            logger.warning("Request failed for %s: %s", url, exc)
            if self.offline_fallback and cache_path and cache_path.exists():
                logger.info("Using cached file: %s", cache_path)
                return json.loads(cache_path.read_text(encoding="utf-8"))
            raise RuntimeError(
                f"FPL API unavailable and no cache found for {url}. "
                "Check your internet connection."
            ) from exc

    # ── Bootstrap data ────────────────────────────────────────────────────────

    def fetch_bootstrap(self) -> dict:
        """Fetch and cache the FPL bootstrap-static endpoint."""
        logger.info("Fetching bootstrap-static…")
        self._bootstrap = self._get(BOOTSTRAP_URL, cache_path=RAW_BOOTSTRAP_PATH)
        return self._bootstrap

    # ── Gameweek metadata ─────────────────────────────────────────────────────

    def get_gameweek_metadata(self) -> dict:
        """
        Return current and next gameweek info.

        Returns
        -------
        dict with keys:
          current_gw, next_gw, gw_deadline, gw_finished,
          season_start, fetched_at
        """
        if not self._bootstrap:
            self.fetch_bootstrap()

        events = self._bootstrap.get("events", [])
        current = next((e for e in events if e.get("is_current")), None)
        nxt     = next((e for e in events if e.get("is_next")),    None)
        prev    = next((e for e in events if e.get("is_previous")), None)

        return {
            "current_gw":   current["id"]          if current else None,
            "next_gw":      nxt["id"]               if nxt     else None,
            "prev_gw":      prev["id"]               if prev    else None,
            "gw_finished":  current.get("finished", False) if current else False,
            "gw_deadline":  nxt["deadline_time"]    if nxt     else None,
            "total_players": self._bootstrap.get("total_players", 0),
            "fetched_at":   datetime.now(timezone.utc).isoformat(),
        }

    # ── Live GW data ──────────────────────────────────────────────────────────

    def fetch_gw_live(self, gw: int) -> dict:
        """Fetch live scores for a specific gameweek."""
        url   = GW_LIVE_URL.format(gw=gw)
        cache = Path(str(RAW_GW_PATH).format(gw=gw))
        logger.info("Fetching GW%d live data…", gw)
        return self._get(url, cache_path=cache)

    # ── Build player DataFrame ────────────────────────────────────────────────

    def build_player_df(self) -> pd.DataFrame:
        """
        Convert bootstrap-static elements → clean player DataFrame.

        Columns produced
        ----------------
        id, player_name, team, team_name, position,
        first_name, second_name,
        total_points, minutes, goals_scored, assists,
        clean_sheets, goals_conceded, yellow_cards, red_cards,
        bonus, bps,
        ict_index, influence, creativity, threat,
        now_cost, selected_by_percent,
        element_type, value_per_m,
        cost  (= now_cost / 10)
        """
        if not self._bootstrap:
            self.fetch_bootstrap()

        elements = self._bootstrap.get("elements", [])
        teams    = {t["id"]: t["name"] for t in self._bootstrap.get("teams", [])}

        if not elements:
            raise ValueError("No player data returned from FPL API.")

        records = []
        for el in elements:
            pos_id  = el.get("element_type", 4)
            cost    = el.get("now_cost", 0) / 10.0
            sel_pct = _safe_float(el.get("selected_by_percent", 0))
            records.append({
                # Identity
                "id":                    el.get("id"),
                "first_name":            el.get("first_name", ""),
                "second_name":           el.get("second_name", ""),
                "player_name":           f"{el.get('first_name','')} {el.get('second_name','')}".strip(),
                "web_name":              el.get("web_name", ""),
                "team":                  teams.get(el.get("team"), "Unknown"),
                "team_id":               el.get("team"),
                "element_type":          POSITION_MAP.get(pos_id, "MID"),
                "position":              POSITION_MAP.get(pos_id, "MID"),
                # Performance
                "total_points":          _safe_int(el.get("total_points", 0)),
                "minutes":               _safe_int(el.get("minutes", 0)),
                "goals_scored":          _safe_int(el.get("goals_scored", 0)),
                "assists":               _safe_int(el.get("assists", 0)),
                "clean_sheets":          _safe_int(el.get("clean_sheets", 0)),
                "goals_conceded":        _safe_int(el.get("goals_conceded", 0)),
                "yellow_cards":          _safe_int(el.get("yellow_cards", 0)),
                "red_cards":             _safe_int(el.get("red_cards", 0)),
                "bonus":                 _safe_int(el.get("bonus", 0)),
                "bps":                   _safe_int(el.get("bps", 0)),
                # ICT
                "ict_index":             _safe_float(el.get("ict_index", 0)),
                "influence":             _safe_float(el.get("influence", 0)),
                "creativity":            _safe_float(el.get("creativity", 0)),
                "threat":                _safe_float(el.get("threat", 0)),
                # Cost / ownership
                "now_cost":              el.get("now_cost", 0),
                "cost":                  cost,
                "selected_by_percent":   sel_pct,
                "value_per_m":           round(_safe_int(el.get("total_points", 0)) / max(cost, 1), 2),
                # Status
                "status":                el.get("status", "u"),  # a=available, i=injured, s=suspended
                "chance_of_playing":     el.get("chance_of_playing_next_round"),
                "news":                  el.get("news", ""),
                "form":                  _safe_float(el.get("form", 0)),
                "points_per_game":       _safe_float(el.get("points_per_game", 0)),
                "transfers_in_event":    _safe_int(el.get("transfers_in_event", 0)),
                "transfers_out_event":   _safe_int(el.get("transfers_out_event", 0)),
            })

        df = pd.DataFrame(records)

        # Validate schema
        _validate_schema(df)

        logger.info("Built player DataFrame: %d players", len(df))
        return df

    # ── Enrich with live GW data ───────────────────────────────────────────────

    def enrich_with_live_gw(self, df: pd.DataFrame, gw: int) -> pd.DataFrame:
        """
        Add live gameweek bonus points to the player DataFrame.
        Only applied if the GW is currently in progress or just finished.
        """
        try:
            live = self.fetch_gw_live(gw)
        except Exception as exc:
            logger.warning("Could not fetch GW%d live data: %s", gw, exc)
            return df

        live_pts = {
            el["id"]: el.get("stats", {}).get("total_points", 0)
            for el in live.get("elements", [])
        }

        df = df.copy()
        df["live_gw_points"] = df["id"].map(live_pts).fillna(0).astype(int)
        logger.info("Enriched with GW%d live points", gw)
        return df

    # ── Fixtures / next opponent ───────────────────────────────────────────────

    def fetch_next_fixture_difficulty(self) -> dict[int, int]:
        """
        Return {team_id: fixture_difficulty_rating} for the next gameweek.
        FDR: 1 (very easy) → 5 (very hard).
        """
        try:
            fixtures = self._get(FIXTURES_URL)
            # Find soonest unfinished fixture per team
            upcoming = [f for f in fixtures if not f.get("finished", True)]
            if not upcoming:
                return {}
            next_gw = min(f["event"] for f in upcoming if f.get("event"))
            gw_fixtures = [f for f in upcoming if f.get("event") == next_gw]
            fdr = {}
            for f in gw_fixtures:
                fdr[f["team_h"]] = f.get("team_h_difficulty", 3)
                fdr[f["team_a"]] = f.get("team_a_difficulty", 3)
            return fdr
        except Exception as exc:
            logger.warning("Could not fetch fixture difficulty: %s", exc)
            return {}

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def fetch_and_save(self) -> pd.DataFrame:
        """
        Full fetch → build → validate → save pipeline.

        1. Pull bootstrap-static (with retry + cache fallback)
        2. Build player DataFrame
        3. Optionally enrich with live GW data
        4. Save raw JSON + processed CSV
        5. Write metadata file with timestamp + GW info

        Returns
        -------
        df : processed player DataFrame
        """
        # Step 1: fetch
        self.fetch_bootstrap()
        meta = self.get_gameweek_metadata()

        # Step 2: build
        df = self.build_player_df()

        # Step 3: live GW enrichment (if current GW in progress)
        current_gw = meta.get("current_gw")
        if current_gw and not meta.get("gw_finished", True):
            df = self.enrich_with_live_gw(df, current_gw)

        # Step 4: save
        PROCESSED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_CSV_PATH, index=False)
        logger.info("Saved processed CSV → %s", PROCESSED_CSV_PATH)

        # Also overwrite the legacy path used by loader.py
        legacy_path = _HERE / "players.csv"
        df.to_csv(legacy_path, index=False)
        logger.info("Updated legacy players.csv → %s", legacy_path)

        # Step 5: metadata
        meta["csv_path"]      = str(PROCESSED_CSV_PATH)
        meta["n_players"]     = len(df)
        meta["current_gw"]    = current_gw
        METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        METADATA_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Saved metadata → %s", METADATA_PATH)

        return df

    # ── Metadata reader (no network needed) ──────────────────────────────────

    @staticmethod
    def read_metadata() -> dict:
        """Read cached metadata without making any API calls."""
        if METADATA_PATH.exists():
            try:
                return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"fetched_at": None, "current_gw": None, "next_gw": None}

    # ── Staleness check ───────────────────────────────────────────────────────

    @staticmethod
    def is_data_stale(max_age_hours: float = 6.0) -> bool:
        """
        Return True if the cached data is older than max_age_hours,
        or if no metadata file exists.
        """
        meta = FPLAPIClient.read_metadata()
        fetched_at = meta.get("fetched_at")
        if not fetched_at:
            return True
        try:
            fetched_dt = datetime.fromisoformat(fetched_at)
            # Make timezone-aware if naive
            if fetched_dt.tzinfo is None:
                fetched_dt = fetched_dt.replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - fetched_dt).total_seconds() / 3600
            return age_hours > max_age_hours
        except Exception:
            return True


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED_COLUMNS = {
    "id", "player_name", "position", "team",
    "total_points", "minutes", "goals_scored", "assists",
    "ict_index", "now_cost", "selected_by_percent",
}


def _validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing."""
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"FPL API response missing expected columns: {missing}. "
            "The API schema may have changed."
        )
    if len(df) < 100:
        raise ValueError(
            f"Too few players returned ({len(df)}). API may have returned partial data."
        )


def _safe_int(val) -> int:
    try:
        return int(val) if val is not None else 0
    except (TypeError, ValueError):
        return 0


def _safe_float(val) -> float:
    try:
        return float(val) if val is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function for quick fetches
# ─────────────────────────────────────────────────────────────────────────────

def quick_fetch(offline_fallback: bool = True) -> pd.DataFrame:
    """
    One-liner to fetch latest data and return a processed DataFrame.
    Equivalent to FPLAPIClient().fetch_and_save().
    """
    client = FPLAPIClient(offline_fallback=offline_fallback)
    return client.fetch_and_save()
