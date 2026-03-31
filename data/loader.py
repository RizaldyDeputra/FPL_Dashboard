"""
FPL AI Optimizer - Data Loader (v3 - Live + Static)
====================================================
Smart loader that prefers live API data, falls back to static CSV.
Backward-compatible with v1/v2.
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger("fpl_loader")

_HERE          = Path(__file__).parent
PROCESSED_PATH = _HERE / "processed" / "players.csv"
STATIC_PATH    = _HERE / "players.csv"


def load_raw(prefer_live: bool = True) -> pd.DataFrame:
    """Load raw player data; prefer live processed CSV, fallback to static."""
    path = _resolve_path(prefer_live)
    logger.info("Loading data from: %s", path)
    df = pd.read_csv(path, low_memory=False)
    return _normalise(df)


def _resolve_path(prefer_live: bool) -> Path:
    if prefer_live and PROCESSED_PATH.exists() and PROCESSED_PATH.stat().st_size > 1000:
        return PROCESSED_PATH
    if STATIC_PATH.exists():
        return STATIC_PATH
    raise FileNotFoundError(
        "No player data found. Run update_data.py or ensure data/players.csv exists."
    )


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    if "player_name" not in df.columns:
        if "first_name" in df.columns and "second_name" in df.columns:
            df["player_name"] = df["first_name"] + " " + df["second_name"]
        elif "web_name" in df.columns:
            df["player_name"] = df["web_name"]
    if "position" not in df.columns:
        pos_map = {1:"GK",2:"DEF",3:"MID",4:"FWD","GK":"GK","DEF":"DEF","MID":"MID","FWD":"FWD"}
        df["position"] = df.get("element_type", pd.Series("MID", index=df.index)).map(pos_map).fillna("MID")
    if "cost" not in df.columns:
        df["cost"] = df["now_cost"] / 10.0 if "now_cost" in df.columns else 5.0
    defaults = {
        "goals_scored":0,"assists":0,"clean_sheets":0,"goals_conceded":0,
        "yellow_cards":0,"red_cards":0,"bonus":0,"bps":0,
        "ict_index":0.0,"influence":0.0,"creativity":0.0,"threat":0.0,
        "selected_by_percent":0.0,"form":0.0,"points_per_game":0.0,
        "transfers_in_event":0,"transfers_out_event":0,"status":"a",
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    safe_min  = df["minutes"].clip(lower=1)
    safe_cost = df["cost"].clip(lower=1)
    df["minutes_90s"]     = df["minutes"] / 90
    df["pts_per_90"]      = df["total_points"] / (safe_min / 90)
    df["goals_per_90"]    = df["goals_scored"] / (safe_min / 90)
    df["assists_per_90"]  = df["assists"] / (safe_min / 90)
    df["ga_per_90"]       = df["goals_per_90"] + df["assists_per_90"]
    df["cs_rate"]         = df["clean_sheets"] / df["minutes_90s"].clip(lower=1)
    df["gc_per_90"]       = df["goals_conceded"] / (safe_min / 90)
    df["ict_ratio"]       = df["ict_index"] / safe_cost
    df["threat_norm"]     = df["threat"] / (safe_min / 90)
    df["creativity_norm"] = df["creativity"] / (safe_min / 90)
    df["form_score"] = (
        0.40 * df["pts_per_90"].clip(0, 20) +
        0.20 * df["ict_ratio"].clip(0, 5) +
        0.20 * df["ga_per_90"].clip(0, 2) +
        0.10 * df["cs_rate"].clip(0, 1) +
        0.10 * (1 - df["gc_per_90"].clip(0, 5) / 5)
    )
    df["value_score"]      = df["total_points"] / safe_cost
    df["is_differential"]  = (df["selected_by_percent"] < 5.0).astype(int)
    net = df["transfers_in_event"] - df["transfers_out_event"]
    abs_max = max(float(net.abs().max()), 1.0)
    df["transfer_momentum"] = net / abs_max
    df.fillna(0, inplace=True)
    return df


def get_feature_columns() -> list[str]:
    return [
        "minutes_90s","pts_per_90","goals_per_90","assists_per_90",
        "ga_per_90","cs_rate","gc_per_90","ict_ratio",
        "threat_norm","creativity_norm","form_score",
        "bonus","bps","selected_by_percent","cost","transfer_momentum",
    ]


def load_and_prepare(prefer_live: bool = True, min_minutes: int = 0) -> pd.DataFrame:
    df = load_raw(prefer_live=prefer_live)
    df = engineer_features(df)
    if min_minutes > 0:
        df = df[df["minutes"] >= min_minutes]
    return df.reset_index(drop=True)


def load_available_only(prefer_live: bool = True, min_minutes: int = 0) -> pd.DataFrame:
    df = load_and_prepare(prefer_live=prefer_live, min_minutes=min_minutes)
    if "status" in df.columns:
        df = df[df["status"].isin(["a", "d"])]
    return df.reset_index(drop=True)
