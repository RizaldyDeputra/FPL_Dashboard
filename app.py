"""
FPL AI Optimizer Dashboard  ·  v3 – Live Pipeline Edition
===========================================================
"""

import json, logging, sys, threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

from data.fpl_api import FPLAPIClient
from data.loader import get_feature_columns, load_and_prepare
from models.model_store import load_predictor, read_model_meta, save_predictor
from models.predictor import run_ml_pipeline
from optimizer.team_selector import (
    best_value_picks, captain_candidates, find_differentials,
    generate_key_insights, optimise_squad, risky_picks,
)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("app")

# ─────────────────────────────────────────────────────────────────────────────
# Cloud startup 
# Runs once per container lifecycle; silently passes if data already exists.
# ─────────────────────────────────────────────────────────────────────────────
try:
    from cloud_startup import ensure_data_available
    _startup = ensure_data_available()
    logger.info("Startup check: %s", _startup)
except RuntimeError as _startup_err:
    st.set_page_config(page_title="FPL AI Optimizer", page_icon="⚽", layout="wide")
    st.error(f"### Data unavailable\n\n{_startup_err}")
    st.stop()
except Exception as _startup_exc:
    logger.warning("Startup check failed (non-fatal): %s", _startup_exc)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FPL AI Optimizer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_STALE_HOURS = 6.0
POS_COLORS = {"GK": "#f8d100", "DEF": "#00e87a", "MID": "#01faf9", "FWD": "#ff4d6d"}
POS_BORDER = {"GK": "#f8d100", "DEF": "#00e87a", "MID": "#01faf9", "FWD": "#ff4d6d"}

# Known FPL player IDs for image lookup (extended map for common players)
PLAYER_ID_MAP = {
    "Raya": 169187, "Flekken": 241978, "Roefs": 493237, "Trippier": 80680, "Pedro Porro": 241568,
    "Gabriel": 148225, "Timber": 493260, "Saliba": 223340, "White": 204480,
    "Van Dijk": 97032, "Robertson": 122798, "Tarkowski": 85971, "Mykolenko": 463916,
    "Guéhi": 223094, "Pedro": 241568, "Senesi": 464035, "Van Hecke": 464034,
    "Haaland": 447,  "Watkins": 216238, "Isak": 224005, "Bowen": 219847,
    "Thiago": 464100, "Wood": 60826,
    "Salah": 118748, "Saka": 223085, "Palmer": 244723, "Mbeumo": 195473,
    "Fernandes": 6021, "Rice": 184341, "Semenyo": 464041, "Rogers": 464043,
    "Wilson": 57892, "Anderson": 238718, "Garner": 444145,
    "Guimarães": 219847, "Gibbs-White": 215966,
}

def get_player_image_url(player_name: str) -> tuple[str, str]:
    """Return (img_url, fallback_initials) for a player."""
    surname = player_name.split()[-1]
    pid = PLAYER_ID_MAP.get(surname)
    if pid:
        return f"https://resources.premierleague.com/premierleague/photos/players/250x250/p{pid}.png", ""
    # Generic placeholder via UI Avatars (free, no API key)
    initials = "".join(w[0] for w in player_name.split()[:2]).upper()
    return f"https://ui-avatars.com/api/?name={initials}&background=1a1a2e&color=fff&size=128&bold=true&rounded=true", initials

def fmt_formation(raw: str) -> str:
    """Convert '1-4-4-2' → '4-4-2' (remove GK from display)."""
    parts = raw.split("-")
    if len(parts) == 4 and parts[0] == "1":
        return "-".join(parts[1:])
    return raw

# ─────────────────────────────────────────────────────────────────────────────
# CSS — complete, self-contained
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root{--g:#00e87a;--gold:#ffd700;--rd:#ff4d6d}

html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important;background:#1c0021!important;color:#e8e0f0!important}
.stApp{background:linear-gradient(160deg,#1c0021 0%,#2d003a 100%)!important}

section[data-testid="stSidebar"]{background:rgba(28,0,33,.98)!important;border-right:1px solid rgba(0,232,122,.15)!important}
section[data-testid="stSidebar"] *{color:#e8e0f0!important}
section[data-testid="stSidebar"] select{background:rgba(255,255,255,.08)!important}

h1,h2,h3{font-family:'Bebas Neue',sans-serif!important;letter-spacing:2px;color:#e8e0f0!important}

/* KPI cards */
.kpi-card{background:rgba(255,255,255,.06);border:1px solid rgba(0,232,122,.22);border-radius:12px;padding:1rem 1.2rem;text-align:center}
.kpi-label{font-size:.65rem;color:rgba(255,255,255,.42);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}
.kpi-value{font-family:'Bebas Neue',sans-serif;font-size:1.7rem;color:#00e87a;line-height:1}
.kpi-value-sm{font-family:'Bebas Neue',sans-serif;font-size:1.05rem;color:#00e87a;line-height:1.2}
.kpi-sub{font-size:.6rem;color:rgba(255,255,255,.3);margin-top:3px}

/* Live status badge */
.status-badge{display:inline-flex;align-items:center;gap:6px;border-radius:20px;padding:4px 12px;font-size:.7rem;font-weight:500;letter-spacing:.3px;margin-bottom:.4rem}
.status-live{background:rgba(0,232,122,.1);border:1px solid rgba(0,232,122,.3);color:#00e87a}
.status-stale{background:rgba(255,200,0,.08);border:1px solid rgba(255,200,0,.25);color:#ffc800}
.status-error{background:rgba(255,77,109,.08);border:1px solid rgba(255,77,109,.25);color:#ff4d6d}
.dot-pulse{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.dot-live{background:#00e87a;animation:pulse-g 2s infinite}
.dot-stale{background:#ffc800}
.dot-error{background:#ff4d6d}
@keyframes pulse-g{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(0,232,122,.4)}50%{opacity:.7;box-shadow:0 0 0 4px rgba(0,232,122,0)}}

/* Pipeline status box */
.pipeline-box{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);border-radius:8px;padding:.7rem .9rem;font-size:.73rem;color:rgba(255,255,255,.5);line-height:1.8}
.pipeline-box strong{color:rgba(255,255,255,.7)}

/* Section headers */
.sh{font-family:'Bebas Neue',sans-serif;font-size:.95rem;letter-spacing:2px;color:rgba(255,255,255,.65);margin-bottom:.6rem}

/* Player rows */
.player-row{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.09);border-radius:8px;padding:.45rem .8rem;margin-bottom:.28rem;display:flex;align-items:center;gap:.45rem;font-size:.82rem}
.player-row:hover{border-color:rgba(0,232,122,.28);background:rgba(0,232,122,.04);cursor:default}
.bench-row{background:rgba(255,255,255,.025);border:1px dashed rgba(255,255,255,.1);border-radius:8px;padding:.38rem .75rem;margin-bottom:.22rem;display:flex;align-items:center;gap:.45rem;font-size:.79rem;color:rgba(255,255,255,.62)}

/* Position tags */
.tag{font-size:.58rem;font-weight:700;padding:2px 5px;border-radius:3px;min-width:28px;text-align:center;flex-shrink:0}
.tag-GK{background:#f8d100;color:#000}
.tag-DEF{background:#00e87a;color:#000}
.tag-MID{background:#01faf9;color:#000}
.tag-FWD{background:#ff4d6d;color:#fff}

/* Captain / VC */
.badge-cap{background:#ffd700;color:#000;font-size:.56rem;font-weight:700;border-radius:3px;padding:1px 4px;flex-shrink:0}
.badge-vc{background:rgba(255,255,255,.22);color:#fff;font-size:.56rem;font-weight:700;border-radius:3px;padding:1px 4px;flex-shrink:0}

/* Injury / availability */
.inj-tag{background:rgba(255,77,109,.2);color:#ff4d6d;border:1px solid rgba(255,77,109,.3);font-size:.56rem;font-weight:700;border-radius:3px;padding:1px 4px;flex-shrink:0}

/* Transfer trend arrows */
.trending-up{color:#00e87a;font-size:.75rem;font-weight:700}
.trending-down{color:#ff4d6d;font-size:.75rem;font-weight:700}

/* Insight & other cards */
.insight-card{background:rgba(0,232,122,.07);border-left:3px solid #00e87a;border-radius:0 8px 8px 0;padding:.65rem .95rem;margin-bottom:.45rem;font-size:.82rem;line-height:1.55}
.insight-card strong{color:#00e87a}
.vc-card{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.1);border-radius:8px;padding:.62rem .9rem;margin-bottom:.38rem}
.diff-card{background:rgba(255,215,0,.06);border:1px solid rgba(255,215,0,.2);border-radius:7px;padding:.55rem .85rem;margin-bottom:.35rem}

/* Chat bubbles */
.chat-user{background:rgba(0,232,122,.1);border:1px solid rgba(0,232,122,.22);border-radius:10px 10px 2px 10px;padding:.55rem .85rem;margin-bottom:.35rem;font-size:.8rem;text-align:right;max-width:82%;margin-left:auto}
.chat-ai{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.09);border-radius:10px 10px 10px 2px;padding:.6rem .9rem;margin-bottom:.45rem;font-size:.8rem;max-width:90%;line-height:1.5}
.cl{font-size:.62rem;color:rgba(0,232,122,.55);margin-bottom:2px}
.ul{font-size:.62rem;color:rgba(255,255,255,.3);text-align:right;margin-bottom:2px}

/* Budget bar */
.budget-bar{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.09);border-radius:8px;padding:.65rem .9rem;font-size:.78rem;margin-top:.5rem}

/* Last-updated pill */
.updated-pill{display:inline-flex;align-items:center;gap:5px;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);border-radius:20px;padding:2px 10px;font-size:.65rem;color:rgba(255,255,255,.4)}

/* Buttons */
div.stButton>button{background:linear-gradient(135deg,#00e87a,#00b85e)!important;color:#1c0021!important;font-weight:700!important;border:none!important;border-radius:8px!important;font-family:'Bebas Neue',sans-serif!important;letter-spacing:1px!important;font-size:1rem!important;padding:.5rem 1.4rem!important}
div.stButton>button:hover{filter:brightness(1.08)}

.stSelectbox>div>div{background:rgba(255,255,255,.06)!important;border-color:rgba(255,255,255,.15)!important;color:#e8e0f0!important}
.stTextInput>div>div>input{background:rgba(255,255,255,.06)!important;border-color:rgba(255,255,255,.15)!important;color:#e8e0f0!important;border-radius:8px!important}
.stSlider [data-testid="stSlider"]{color:#00e87a!important}

div[data-testid="stMetricValue"]{color:#00e87a!important}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data / pipeline helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_data_status() -> dict:
    """Read metadata from disk — zero network calls."""
    api_meta   = FPLAPIClient.read_metadata()
    model_meta = read_model_meta()
    is_stale   = FPLAPIClient.is_data_stale(max_age_hours=DATA_STALE_HOURS)

    fetched_at = api_meta.get("fetched_at")
    age_str = "never"
    if fetched_at:
        try:
            dt = datetime.fromisoformat(fetched_at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
            if age_h < 1:
                age_str = f"{int(age_h*60)}m ago"
            elif age_h < 48:
                age_str = f"{age_h:.1f}h ago"
            else:
                age_str = f"{age_h/24:.1f}d ago"
        except Exception:
            age_str = "unknown"

    return {
        "fetched_at":        fetched_at,
        "age_str":           age_str,
        "is_stale":          is_stale,
        "current_gw":        api_meta.get("current_gw"),
        "next_gw":           api_meta.get("next_gw"),
        "gw_deadline":       api_meta.get("gw_deadline"),
        "n_players":         api_meta.get("n_players", "?"),
        "total_managers":    api_meta.get("total_players"),
        "model_name":        model_meta.get("best_model", "—"),
        "model_trained_at":  model_meta.get("trained_at"),
        "model_n_players":   model_meta.get("n_players"),
        "data_source":       "🟢 Live API" if (fetched_at and not is_stale) else ("🟡 Cached" if fetched_at else "⚪ Static CSV"),
    }


def _fmt_deadline(iso_str: str | None) -> str:
    if not iso_str:
        return "TBC"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%a %d %b, %H:%M UTC")
    except Exception:
        return iso_str[:16]


def _fmt_trained_at(iso_str: str | None) -> str:
    if not iso_str:
        return "Never"
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
        return f"{age_h:.1f}h ago"
    except Exception:
        return iso_str[:16]


def do_live_refresh() -> tuple[bool, str]:
    """Fetch latest data from FPL API and trigger re-save."""
    try:
        client = FPLAPIClient(offline_fallback=True)
        df_raw = client.fetch_and_save()
        return True, f"✓ Fetched {len(df_raw)} players from FPL API (GW{client.get_gameweek_metadata().get('current_gw','?')})"
    except Exception as exc:
        logger.error("Live refresh failed: %s", exc)
        return False, f"API unavailable: {exc}. Using cached data."


def get_cache_key() -> str:
    """Cache-bust key derived from data file mtime."""
    from data.loader import PROCESSED_PATH, STATIC_PATH
    for p in [PROCESSED_PATH, STATIC_PATH]:
        if p.exists():
            return f"{p.stem}_{int(p.stat().st_mtime)}"
    return "fallback_static"


@st.cache_data(show_spinner=False, ttl=3600)
def run_pipeline_cached(cache_key: str, _min_minutes: int = 0):
    """
    ML pipeline — auto-invalidated when cache_key changes (= new data fetched).
    Loads cached model from disk first; only retrains if data changed.
    """
    df = load_and_prepare(prefer_live=True, min_minutes=0)
    feat_cols = get_feature_columns()
    cached_pred = load_predictor(df)
    if cached_pred is not None:
        df = df.copy()
        df["predicted_points"] = cached_pred.predict(df).clip(0)
        return df, cached_pred, cached_pred.metrics
    df, predictor, metrics = run_ml_pipeline(df, feat_cols)
    save_predictor(predictor, df)
    return df, predictor, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color:#00e87a;margin-bottom:0'>⚽ Fantasy Premier League</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:.7rem;color:rgba(255,255,255,.35);margin-top:0;letter-spacing:.5px'>LIVE SQUAD OPTIMIZER</p>", unsafe_allow_html=True)

    status = get_data_status()
    gw_label = f"GW {status['current_gw']}" if status["current_gw"] else "Pre-Season"

    # Status badge
    if status["fetched_at"] is None:
        badge_cls, dot_cls, badge_text = "status-error",  "dot-error",  "No live data yet"
    elif status["is_stale"]:
        badge_cls, dot_cls, badge_text = "status-stale",  "dot-stale",  f"Stale · {status['age_str']}"
    else:
        badge_cls, dot_cls, badge_text = "status-live",   "dot-live",   f"Live · {status['age_str']}"

    st.markdown(
        f"<div class='status-badge {badge_cls}'>"
        f"<span class='dot-pulse {dot_cls}'></span> {badge_text}</div>",
        unsafe_allow_html=True,
    )

    # Refresh button
    if st.button("🔄  Refresh Live Data", use_container_width=True):
        with st.spinner("Fetching from FPL API…"):
            ok, msg = do_live_refresh()
        if ok:
            st.success(msg)
            st.cache_data.clear()
            st.rerun()
        else:
            st.warning(msg)

    st.markdown("---")

    # GW info
    deadline_str = _fmt_deadline(status["gw_deadline"])
    st.markdown(
        f"<div style='font-size:.73rem;color:rgba(255,255,255,.5);line-height:2'>"
        f"<b style='color:#00e87a'>{gw_label}</b><br>"
        f"Players tracked: <b>{status['n_players']}</b><br>"
        f"Deadline: <b style='color:rgba(255,255,255,.7)'>{deadline_str}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Squad controls
    max_per_club = int(st.selectbox("Max per club", ["3","2","1"], index=0))
    min_minutes  = st.slider("Min. minutes", 0, 2000, 500, 100,
                             help="Exclude players with fewer than this many minutes played")

    st.markdown("---")
    optimise_btn = st.button("🚀  Optimise Squad", use_container_width=True)
    st.markdown("---")

    # Pipeline status expander
    with st.expander("🔧 Pipeline Status", expanded=False):
        trained_ago = _fmt_trained_at(status["model_trained_at"])
        st.markdown(
            f"<div class='pipeline-box'>"
            f"<strong>Data</strong><br>"
            f"Source: {status['data_source']}<br>"
            f"Last fetch: {status['age_str']}<br>"
            f"Players: {status['n_players']}<br>"
            f"Stale threshold: {DATA_STALE_HOURS}h<br><br>"
            f"<strong>ML Model</strong><br>"
            f"Algorithm: {status['model_name']}<br>"
            f"Trained: {trained_ago}<br>"
            f"On: {status.get('model_n_players','?')} players<br><br>"
            f"<strong>Automation</strong><br>"
            f"<code style='font-size:.65rem;color:#00e87a'>python update_data.py</code><br>"
            f"<code style='font-size:.65rem;color:#00e87a'>python update_data.py --schedule</code><br>"
            f"<span style='font-size:.65rem;color:rgba(255,255,255,.3)'>Cron: 0 6 * * * python update_data.py</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<small style='color:rgba(255,255,255,.2)'>Budget £100M · LP formation<br>"
        "Squad: 2GK · 5DEF · 5MID · 3FWD</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Load data + ML pipeline
# ─────────────────────────────────────────────────────────────────────────────
cache_key = get_cache_key()
with st.spinner("Loading player data & predictions…"):
    df_all, predictor, ml_metrics = run_pipeline_cached(cache_key, min_minutes)

df_filtered = df_all[df_all["minutes"] >= min_minutes].copy()

# ─────────────────────────────────────────────────────────────────────────────
# Run optimizer (invalidated by cache key or user pressing Optimise)
# ─────────────────────────────────────────────────────────────────────────────
if (
    "squad_result" not in st.session_state
    or optimise_btn
    or st.session_state.get("_cache_key") != cache_key
    or st.session_state.get("_max_club") != max_per_club
    or st.session_state.get("_min_min") != min_minutes
):
    with st.spinner("Running LP optimizer…"):
        res = optimise_squad(df_filtered, budget=100.0, max_per_club=max_per_club)
        st.session_state["squad_result"] = res
        st.session_state["_cache_key"]   = cache_key
        st.session_state["_max_club"]    = max_per_club
        st.session_state["_min_min"]     = min_minutes

result  = st.session_state["squad_result"]
xi      = result["xi"]
bench   = result["bench"]
summary = result["summary"]

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
src_icon = "🟢" if not status["is_stale"] and status["fetched_at"] else "🟡"
st.markdown(
    f"<h1 style='text-align:center;color:#00e87a;margin-bottom:3px'>Fantasy Premier League</h1>"
    f"<p style='text-align:center;color:rgba(255,255,255,.35);font-size:.78rem;margin-top:0'>"
    f"{gw_label} &nbsp;·&nbsp; {src_icon} {status['data_source']} &nbsp;·&nbsp; "
    f"Updated {status['age_str']} &nbsp;·&nbsp; {len(df_filtered)} players</p>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# KPI strip
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
kpis = [
    (k1, "Squad Cost",   f"£{summary['squad_cost']}M",              f"£{summary['budget_remaining']}M free"),
    (k2, "XI Predicted", f"{summary['xi_pts']} pts",                "Starting eleven"),
    (k3, "Formation",    fmt_formation(summary["formation"]),        "Auto-optimised"),
    (k4, "Captain",      summary["captain"].split()[-1],            "2× points pick"),
    (k5, "Vice Captain", summary["vice_captain"].split()[-1],       "Backup captain"),
    (k6, "Bench Value",  f"{summary['bench_pts']} pts",             "Coverage score"),
]
for col, label, val, sub in kpis:
    with col:
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value-sm'>{val}</div>"
            f"<div class='kpi-sub'>{sub}</div></div>",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_squad, tab_insights, tab_top, tab_diff, tab_ai, tab_about = st.tabs([
    "🏟 My Squad", "💡 Insights", "📈 Top Players",
    "🎯 Differentials", "🤖 AI Advisor", "ℹ️ About",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MY SQUAD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_squad:
    col_pitch, col_list = st.columns([1.1, 1], gap="large")

    # ── Pitch ─────────────────────────────────────────────────────────────────
    with col_pitch:
        st.markdown("<div class='sh'>🏟 Starting XI</div>", unsafe_allow_html=True)

        def render_html_pitch(xi_df: pd.DataFrame, bench_df: pd.DataFrame) -> str:
            """
            Flexbox row-based pitch layout.
            RULES:
              - No HTML comments (<!-- -->) — Streamlit strips content after them
              - No triple-quoted multiline f-strings — use string concatenation
              - Player rows use display:flex, not position:absolute
              - CSS classes defined in <style> block above handle all presentation
            """

            # ── Build player cards per position row ───────────────────────────
            def player_card(p) -> str:
                img_url, _ = get_player_image_url(p["player_name"])
                surname = p["player_name"].split()[-1][:13]
                bc = POS_BORDER[p["position"]]
                pts_val = f"{float(p['predicted_points']):.1f}"
                # Fallback initials for onerror
                initials = "".join(w[0] for w in p["player_name"].split()[:2]).upper()
                fb = f"https://ui-avatars.com/api/?name={initials}&background=0d1117&color=fff&size=128&bold=true&rounded=true"

                badge = ""
                if p.get("is_captain"):
                    badge = "<span class='pc-badge pc-cap'>C</span>"
                elif p.get("is_vice_captain"):
                    badge = "<span class='pc-badge pc-vc'>V</span>"

                dot = ""
                st_p = str(p.get("status", "a"))
                if st_p == "i":
                    dot = "<span class='pc-dot pc-inj'></span>"
                elif st_p == "d":
                    dot = "<span class='pc-dot pc-dtd'></span>"

                return (
                    "<div class='pc'>"
                    f"<div class='pc-av' style='border-color:{bc}'>"
                    f"<img src='{img_url}' "
                    f"onerror=\"this.onerror=null;this.src='{fb}'\" "
                    f"style='width:100%;height:100%;border-radius:50%;object-fit:cover;display:block;background:#0d1117'/>"
                    f"{badge}{dot}"
                    "</div>"
                    "<div class='pc-info'>"
                    f"<span class='pc-name'>{surname}</span>"
                    f"<span class='pc-pts' style='color:{bc}'>{pts_val}</span>"
                    "</div>"
                    "</div>"
                )

            # ── Build bench cards ─────────────────────────────────────────────
            def bench_card(p, order: int) -> str:
                img_url, _ = get_player_image_url(p["player_name"])
                surname = p["player_name"].split()[-1][:11]
                bc = POS_BORDER[p["position"]]
                pts_val = f"{float(p['predicted_points']):.1f}"
                initials = "".join(w[0] for w in p["player_name"].split()[:2]).upper()
                fb = f"https://ui-avatars.com/api/?name={initials}&background=0d1117&color=888&size=128&bold=true&rounded=true"

                return (
                    "<div class='bc'>"
                    f"<div class='bc-av' style='border-color:{bc}'>"
                    f"<img src='{img_url}' "
                    f"onerror=\"this.onerror=null;this.src='{fb}'\" "
                    f"style='width:100%;height:100%;border-radius:50%;object-fit:cover;display:block;background:#0d1117'/>"
                    f"<span class='bc-ord'>{order}</span>"
                    "</div>"
                    f"<span class='bc-name'>{surname}</span>"
                    f"<span class='bc-pts' style='color:{bc}'>{pts_val}</span>"
                    "</div>"
                )

            # ── Assemble formation rows ───────────────────────────────────────
            pos_groups: dict[str, list] = {"GK": [], "DEF": [], "MID": [], "FWD": []}
            for _, p in xi_df.iterrows():
                pos_groups[p["position"]].append(p)

            rows = ""
            for pos in ["FWD", "MID", "DEF", "GK"]:
                players = pos_groups[pos]
                if not players:
                    continue
                cards = "".join(player_card(p) for p in players)
                rows += f"<div class='prow'>{cards}</div>"

            bench_cards = "".join(
                bench_card(p, i + 1)
                for i, (_, p) in enumerate(bench_df.iterrows())
            )

            # ── Final HTML — NO HTML comments, string concatenation only ──────
            return (
                "<div class='fpl-pitch-wrap'>"
                "<div class='fpl-pitch'>"
                "<div class='fpl-pitch-lines' aria-hidden='true'>"
                "<div class='pl-border'></div>"
                "<div class='pl-halfway'></div>"
                "<div class='pl-circle'></div>"
                "<div class='pl-dot'></div>"
                "<div class='pl-box-top'></div>"
                "<div class='pl-box-bot'></div>"
                "</div>"
                + rows
                + "</div>"
                "<div class='fpl-bench'>"
                "<div class='fpl-bench-label'>BENCH</div>"
                "<div class='fpl-bench-row'>"
                + bench_cards
                + "</div>"
                "</div>"
                "</div>"
            )

        # ── Pitch CSS (scoped, injected once alongside the HTML) ──────────────
        pitch_css = """
<style>
.fpl-pitch-wrap{font-family:'DM Sans',sans-serif;width:100%}
.fpl-pitch{
  width:100%;display:flex;flex-direction:column;justify-content:space-evenly;
  gap:4px;padding:14px 8px;border-radius:14px;position:relative;overflow:hidden;
  background:linear-gradient(180deg,#0b5c24 0%,#177032 48%,#177032 52%,#0b5c24 100%);
  border:1.5px solid rgba(255,255,255,.18);min-height:0;
}
.fpl-pitch-lines{position:absolute;inset:0;pointer-events:none;z-index:0}
.pl-border{position:absolute;inset:10px;border:1.5px solid rgba(255,255,255,.22);border-radius:8px}
.pl-halfway{position:absolute;left:10px;right:10px;top:50%;height:1.5px;background:rgba(255,255,255,.22)}
.pl-circle{position:absolute;left:50%;top:50%;width:88px;height:88px;transform:translate(-50%,-50%);border:1.5px solid rgba(255,255,255,.22);border-radius:50%}
.pl-dot{position:absolute;left:50%;top:50%;width:7px;height:7px;transform:translate(-50%,-50%);background:rgba(255,255,255,.35);border-radius:50%}
.pl-box-top{position:absolute;left:50%;top:10px;width:180px;height:60px;transform:translateX(-50%);border:1.5px solid rgba(255,255,255,.2);border-top:none;border-radius:0 0 8px 8px}
.pl-box-bot{position:absolute;left:50%;bottom:10px;width:180px;height:60px;transform:translateX(-50%);border:1.5px solid rgba(255,255,255,.2);border-bottom:none;border-radius:8px 8px 0 0}

.prow{display:flex;flex-direction:row;justify-content:space-evenly;align-items:center;width:100%;padding:6px 0;position:relative;z-index:2}

.pc{display:flex;flex-direction:column;align-items:center;gap:5px;width:72px;flex-shrink:0}
.pc-av{width:52px;height:52px;border-radius:50%;border:3px solid #fff;position:relative;flex-shrink:0;box-shadow:0 4px 14px rgba(0,0,0,.5)}
.pc-badge{position:absolute;top:-4px;right:-4px;width:18px;height:18px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:800;color:#000;border:1.5px solid #fff;z-index:3}
.pc-cap{background:#ffd700}
.pc-vc{background:#c8d0dc}
.pc-dot{position:absolute;top:0;left:0;width:12px;height:12px;border-radius:50%;border:1.5px solid #fff;z-index:3}
.pc-inj{background:#ff4d6d}
.pc-dtd{background:#ffc800}
.pc-info{background:rgba(0,0,0,.72);border-radius:6px;padding:3px 6px;width:100%;text-align:center;border:1px solid rgba(255,255,255,.08)}
.pc-name{display:block;color:#fff;font-size:10px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:68px;line-height:1.3}
.pc-pts{display:block;font-size:10px;font-weight:800;line-height:1.3}

.fpl-bench{margin-top:10px;background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.1);border-radius:12px;padding:10px 8px}
.fpl-bench-label{text-align:center;font-size:10px;letter-spacing:.18em;color:rgba(255,255,255,.4);margin-bottom:8px}
.fpl-bench-row{display:flex;flex-direction:row;justify-content:space-evenly;align-items:flex-start}

.bc{display:flex;flex-direction:column;align-items:center;gap:4px;width:70px;flex-shrink:0}
.bc-av{width:44px;height:44px;border-radius:50%;border:2.5px solid #fff;position:relative;opacity:.82;flex-shrink:0;box-shadow:0 3px 10px rgba(0,0,0,.4)}
.bc-ord{position:absolute;top:-4px;left:-4px;width:15px;height:15px;border-radius:50%;background:rgba(8,10,16,.88);border:1px solid rgba(255,255,255,.2);display:flex;align-items:center;justify-content:center;font-size:8px;font-weight:800;color:rgba(255,255,255,.7)}
.bc-name{font-size:10px;color:rgba(232,224,240,.7);font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:68px;text-align:center}
.bc-pts{font-size:10px;font-weight:800;text-align:center}
</style>"""

        pitch_html = pitch_css + render_html_pitch(xi, bench)
        st.markdown(pitch_html, unsafe_allow_html=True)

    # ── Player list ────────────────────────────────────────────────────────────
    with col_list:
        st.markdown("<div class='sh'>Starting XI</div>", unsafe_allow_html=True)
        po = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
        for _, p in xi.sort_values("position", key=lambda s: s.map(po)).iterrows():
            cap_h = "<span class='badge-cap'>C</span>"  if p.get("is_captain")      else ""
            vc_h  = "<span class='badge-vc'>Vc</span>"  if p.get("is_vice_captain") else ""
            st_p  = str(p.get("status", "a"))
            inj_h = ""
            if st_p == "i":
                inj_h = "<span class='inj-tag'>INJ</span>"
            elif st_p == "s":
                inj_h = "<span class='inj-tag'>SUSP</span>"
            elif st_p == "d":
                inj_h = "<span class='inj-tag' style='background:rgba(255,200,0,.18);color:#ffc800;border-color:rgba(255,200,0,.3)'>DTD</span>"
            mom = p.get("transfer_momentum", 0)
            trend_h = ""
            if mom > 0.1:
                trend_h = "<span class='trending-up'>▲</span>"
            elif mom < -0.1:
                trend_h = "<span class='trending-down'>▼</span>"

            st.markdown(
                f"<div class='player-row'>"
                f"<span class='tag tag-{p['position']}'>{p['position']}</span>"
                f"<span style='flex:1;font-weight:500'>{p['player_name']}</span>"
                f"{inj_h}{cap_h}{vc_h}{trend_h}"
                f"<span style='color:#00e87a;font-weight:700;margin-left:.25rem'>{p['predicted_points']:.1f} pts</span>"
                f"<span style='color:rgba(255,255,255,.3);font-size:.7rem;margin-left:.35rem'>£{p['cost']:.1f}M</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br><div class='sh' style='color:rgba(255,255,255,.38)'>Bench</div>", unsafe_allow_html=True)
        for i, (_, p) in enumerate(bench.iterrows()):
            st.markdown(
                f"<div class='bench-row'>"
                f"<span style='font-size:.6rem;color:rgba(255,255,255,.22);min-width:12px'>{i}</span>"
                f"<span class='tag tag-{p['position']}'>{p['position']}</span>"
                f"<span style='flex:1'>{p['player_name']}</span>"
                f"<span style='color:rgba(0,232,122,.6);font-weight:600'>{p['predicted_points']:.1f} pts</span>"
                f"<span style='color:rgba(255,255,255,.22);font-size:.7rem;margin-left:.35rem'>£{p['cost']:.1f}M</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Budget bar
        prog = int(min(summary["squad_cost"] / 100.0, 1.0) * 100)
        xi_c_val   = summary["xi_cost"]
        bnch_c_val = round(summary["squad_cost"] - xi_c_val, 1)
        st.markdown(
            f"<div class='budget-bar'>"
            f"<div style='display:flex;justify-content:space-between;margin-bottom:.4rem'>"
            f"<span>Squad Budget</span>"
            f"<span style='color:#00e87a;font-weight:700'>£{summary['squad_cost']}M / £100M</span></div>"
            f"<div style='height:6px;background:rgba(255,255,255,.08);border-radius:3px'>"
            f"<div style='width:{prog}%;height:100%;background:linear-gradient(90deg,#00e87a,#00b85e);border-radius:3px'></div></div>"
            f"<div style='display:flex;justify-content:space-between;margin-top:.35rem;font-size:.68rem;color:rgba(255,255,255,.3)'>"
            f"<span>£{summary['budget_remaining']}M remaining</span>"
            f"<span>XI: £{xi_c_val}M · Bench: £{bnch_c_val}M</span></div></div>",
            unsafe_allow_html=True,
        )

        # Last updated line
        st.markdown(
            f"<p style='font-size:.63rem;color:rgba(255,255,255,.22);text-align:right;margin-top:.35rem'>"
            f"<span class='updated-pill'>🕐 {status['data_source']} · {status['age_str']}</span></p>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_insights:
    ins_a, ins_b = st.columns([1, 1], gap="large")

    with ins_a:
        st.markdown("<div class='sh'>💡 Key Insights</div>", unsafe_allow_html=True)
        for ins in generate_key_insights(df_filtered, xi):
            st.markdown(f"<div class='insight-card'>{ins}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='sh'>© Captain Picks</div>", unsafe_allow_html=True)
        cap_picks = captain_candidates(xi, top_n=3)
        medals = ["🥇", "🥈", "🥉"]
        for rank, (_, p) in enumerate(cap_picks.iterrows(), 1):
            ppm = p.get("pts_per_90", 0)
            stars = "⭐⭐⭐" if ppm > 4 else ("⭐⭐" if ppm > 2.5 else "⭐")
            pos = p["position"]
            st.markdown(
                f"<div class='vc-card'>"
                f"<div style='display:flex;align-items:center;gap:.5rem'>"
                f"<span style='font-size:1.05rem'>{medals[rank-1]}</span>"
                f"<span class='tag tag-{pos}'>{pos}</span>"
                f"<strong>{p['player_name']}</strong>"
                f"<span style='margin-left:auto;color:#00e87a;font-weight:700'>{p['predicted_points']:.1f} pts</span>"
                f"</div>"
                f"<div style='display:flex;gap:1rem;font-size:.7rem;color:rgba(255,255,255,.42);margin-top:.38rem'>"
                f"<span>Form {stars}</span>"
                f"<span>pts/90: {ppm:.1f}</span>"
                f"<span>ICT: {p.get('ict_index',0):.0f}</span>"
                f"<span>Owned: {p.get('selected_by_percent',0):.1f}%</span>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    with ins_b:
        st.markdown("<div class='sh'>💎 Best Value Picks</div>", unsafe_allow_html=True)
        vp = best_value_picks(df_filtered, top_n=8)
        max_vpp = (vp["predicted_points"] / vp["cost"].clip(lower=1)).max() if len(vp) else 1
        for _, p in vp.iterrows():
            vpp = p["predicted_points"] / max(p["cost"], 1)
            bar = min(int(vpp / max_vpp * 100), 100)
            trend = "<span class='trending-up'>▲</span>" if p.get("transfer_momentum", 0) > 0.1 else ""
            pos = p["position"]
            st.markdown(
                f"<div style='background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);"
                f"border-radius:8px;padding:.48rem .78rem;margin-bottom:.3rem'>"
                f"<div style='display:flex;align-items:center;gap:.4rem'>"
                f"<span class='tag tag-{pos}'>{pos}</span>"
                f"<span style='flex:1;font-size:.8rem;font-weight:500'>{p['player_name']}{' ' + trend if trend else ''}</span>"
                f"<span style='color:#ffd700;font-weight:700;font-size:.74rem'>{vpp:.2f} pts/£M</span>"
                f"</div>"
                f"<div style='display:flex;align-items:center;gap:.5rem;margin-top:.3rem'>"
                f"<div style='flex:1;height:4px;background:rgba(255,255,255,.08);border-radius:2px'>"
                f"<div style='width:{bar}%;height:100%;background:linear-gradient(90deg,#ffd700,#ffa500);border-radius:2px'></div></div>"
                f"<span style='font-size:.68rem;color:rgba(255,255,255,.32)'>£{p['cost']:.1f}M · {p['predicted_points']:.1f} pts</span>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sh'>📉 Risky High-Ceiling</div>"
            "<p style='font-size:.72rem;color:rgba(255,255,255,.38);margin-bottom:.5rem'>"
            "High predicted pts but inconsistent per-game output — gamble carefully.</p>",
            unsafe_allow_html=True,
        )
        for _, p in risky_picks(df_filtered, top_n=5).iterrows():
            risk = "🔴 High risk" if p.get("pts_per_90", 0) < 3 else "🟡 Medium risk"
            pos = p["position"]
            st.markdown(
                f"<div class='diff-card'>"
                f"<div style='display:flex;align-items:center;gap:.4rem'>"
                f"<span class='tag tag-{pos}'>{pos}</span>"
                f"<span style='flex:1;font-size:.8rem;font-weight:500'>{p['player_name']}</span>"
                f"<span style='font-size:.7rem'>{risk}</span>"
                f"</div>"
                f"<div style='font-size:.68rem;color:rgba(255,255,255,.38);margin-top:.3rem'>"
                f"Predicted: {p['predicted_points']:.1f} pts · pts/90: {p.get('pts_per_90',0):.1f} · "
                f"Owned: {p['selected_by_percent']:.1f}%"
                f"</div></div>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TOP PLAYERS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_top:
    tf1, tf2 = st.columns([3, 1])
    with tf2:
        pos_f  = st.selectbox("Position", ["All","GK","DEF","MID","FWD"], key="top_pos")
        n_show = st.slider("Show N", 5, 30, 15, key="top_n")
        avail_only = st.checkbox("Available only", value=False, key="avail_only")
    with tf1:
        st.markdown("<div class='sh'>📈 Top Predicted Players</div>", unsafe_allow_html=True)

    pool = df_filtered if pos_f == "All" else df_filtered[df_filtered["position"] == pos_f]
    if avail_only and "status" in pool.columns:
        pool = pool[pool["status"].isin(["a","d"])]
    top = pool.nlargest(n_show, "predicted_points")
    max_pts = top["predicted_points"].max() if len(top) else 1

    for _, p in top.iterrows():
        bw      = int(p["predicted_points"] / max_pts * 100)
        in_xi   = p["player_name"] in xi["player_name"].values
        in_bnch = p["player_name"] in bench["player_name"].values
        sq_tag  = (
            "<span style='font-size:.58rem;background:#00e87a;color:#000;border-radius:3px;padding:1px 4px'>IN XI</span>"
            if in_xi else (
            "<span style='font-size:.58rem;background:rgba(255,255,255,.18);color:#fff;border-radius:3px;padding:1px 4px'>BENCH</span>"
            if in_bnch else "")
        )
        st_p = str(p.get("status", "a"))
        inj_tag = ""
        if st_p == "i":
            inj_tag = "<span class='inj-tag'>INJ</span>"
        elif st_p == "s":
            inj_tag = "<span class='inj-tag'>SUSP</span>"
        elif st_p == "d":
            inj_tag = "<span class='inj-tag' style='background:rgba(255,200,0,.15);color:#ffc800;border-color:rgba(255,200,0,.3)'>DTD</span>"
        mom_arr = ""
        if p.get("transfer_momentum", 0) > 0.1:
            mom_arr = "<span class='trending-up'>▲</span>"
        elif p.get("transfer_momentum", 0) < -0.1:
            mom_arr = "<span class='trending-down'>▼</span>"

        st.markdown(
            f"<div class='player-row' style='flex-direction:column;align-items:stretch;gap:0'>"
            f"<div style='display:flex;align-items:center;gap:.4rem'>"
            f"<span class='tag tag-{p['position']}'>{p['position']}</span>"
            f"<span style='flex:1;font-weight:500;font-size:.82rem'>{p['player_name']}</span>"
            f"{inj_tag}{sq_tag}{mom_arr}"
            f"<span style='color:#00e87a;font-weight:700;font-size:.82rem;margin-left:.2rem'>{p['predicted_points']:.1f} pts</span>"
            f"<span style='color:rgba(255,255,255,.28);font-size:.7rem;margin-left:.3rem'>£{p['cost']:.1f}M</span>"
            f"</div>"
            f"<div style='display:flex;align-items:center;gap:.5rem;margin-top:.3rem'>"
            f"<div style='flex:1;height:4px;background:rgba(255,255,255,.08);border-radius:2px'>"
            f"<div style='width:{bw}%;height:100%;background:{POS_COLORS.get(p['position'], '#aaa')};border-radius:2px;opacity:.75'></div></div>"
            f"<span style='font-size:.67rem;color:rgba(255,255,255,.28)'>"
            f"G:{int(p.get('goals_scored',0))} A:{int(p.get('assists',0))} · "
            f"{p.get('selected_by_percent',0):.1f}% owned · "
            f"ICT:{p.get('ict_index',0):.0f}</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DIFFERENTIALS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_diff:
    dc1, dc2 = st.columns([3, 1])
    with dc2:
        own_thresh = st.slider("Max ownership %", 1.0, 20.0, 5.0, 0.5, key="diff_t")
        diff_pos   = st.selectbox("Position", ["All","GK","DEF","MID","FWD"], key="diff_pos")
    with dc1:
        st.markdown("<div class='sh'>🎯 Differential Picks</div>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:.75rem;color:rgba(255,255,255,.38);margin-bottom:.6rem'>"
            "Low-ownership picks with strong predicted returns — your edge over the template.</p>",
            unsafe_allow_html=True,
        )

    d_pool = df_filtered if diff_pos == "All" else df_filtered[df_filtered["position"] == diff_pos]
    diffs  = find_differentials(d_pool, threshold=own_thresh, top_n=14)

    for _, p in diffs.iterrows():
        vpp    = p["predicted_points"] / max(p["cost"], 1)
        in_sq  = p["player_name"] in result["squad"]["player_name"].values
        sq_tag = (
            "<span style='font-size:.58rem;background:#00e87a;color:#000;border-radius:3px;padding:1px 4px'>IN SQUAD</span>"
            if in_sq else ""
        )
        mom_arr = "<span class='trending-up'>▲</span>" if p.get("transfer_momentum", 0) > 0.1 else ""

        st.markdown(
            f"<div class='diff-card'>"
            f"<div style='display:flex;align-items:center;gap:.5rem'>"
            f"<span class='tag tag-{p['position']}'>{p['position']}</span>"
            f"<span style='flex:1;font-size:.82rem;font-weight:500'>{p['player_name']}</span>"
            f"{sq_tag}{mom_arr}"
            f"<span style='color:#00e87a;font-weight:700'>{p['predicted_points']:.1f} pts</span>"
            f"</div>"
            f"<div style='font-size:.7rem;color:rgba(255,255,255,.4);margin-top:.3rem;display:flex;gap:1.2rem'>"
            f"<span>£{p['cost']:.1f}M</span>"
            f"<span style='color:#ffd700'>👁 {p['selected_by_percent']:.1f}% owned</span>"
            f"<span>{vpp:.2f} pts/£M</span>"
            f"<span>ICT: {p.get('ict_index',0):.0f}</span>"
            f"<span>G:{int(p.get('goals_scored',0))} A:{int(p.get('assists',0))}</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — AI ADVISOR  (free: rule-based engine + optional Ollama LLM)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ai:

    def build_rag_context(xi_df, bench_df, df_all_f) -> str:
        lines = [f"=== FPL SQUAD CONTEXT ({gw_label}) ===", "STARTING XI:"]
        for _, p in xi_df.iterrows():
            badge = " [CAPTAIN]" if p.get("is_captain") else (" [VC]" if p.get("is_vice_captain") else "")
            lines.append(
                f"  {p['position']} | {p['player_name']}{badge} | "
                f"£{p['cost']:.1f}M | {p['predicted_points']:.1f} pred pts | "
                f"G:{int(p.get('goals_scored',0))} A:{int(p.get('assists',0))} | "
                f"ICT:{p.get('ict_index',0):.0f} | Owned:{p.get('selected_by_percent',0):.1f}%"
            )
        lines += ["", "BENCH:"]
        for _, p in bench_df.iterrows():
            lines.append(f"  {p['position']} | {p['player_name']} | £{p['cost']:.1f}M | {p['predicted_points']:.1f} pts")
        lines += ["", "TOP 10 PLAYERS:"]
        for _, p in df_all_f.nlargest(10, "predicted_points").iterrows():
            lines.append(f"  {p['position']} | {p['player_name']} | £{p['cost']:.1f}M | {p['predicted_points']:.1f} pts | {p.get('selected_by_percent',0):.1f}% owned")
        val = df_all_f.assign(vpp=df_all_f["predicted_points"] / df_all_f["cost"].clip(lower=1))
        lines += ["", "BEST VALUE:"]
        for _, p in val.nlargest(5, "vpp").iterrows():
            lines.append(f"  {p['player_name']} ({p['position']}) — {p['vpp']:.2f} pts/£M")
        diffs_ctx = find_differentials(df_all_f, threshold=5.0, top_n=5)
        lines += ["", "TOP DIFFERENTIALS (<5% owned):"]
        for _, p in diffs_ctx.iterrows():
            lines.append(f"  {p['player_name']} ({p['position']}) — {p['predicted_points']:.1f} pts, {p['selected_by_percent']:.1f}% owned")
        return "\n".join(lines)

    def rule_based_ai(question: str, xi_df, bench_df, df_all_f) -> str:
        q = question.lower()
        cap_row = xi_df.sort_values("predicted_points", ascending=False).iloc[0]
        vc_row  = xi_df.sort_values("predicted_points", ascending=False).iloc[1] if len(xi_df) > 1 else cap_row

        if any(w in q for w in ["captain", "cap", "armband"]):
            top3 = xi_df.nlargest(3, "predicted_points")
            lines = [f"**© Captain recommendation: {cap_row['player_name']}**\n"]
            for rank, (_, p) in enumerate(top3.iterrows(), 1):
                icon = ["🥇","🥈","🥉"][rank-1]
                lines.append(f"{icon} **{p['player_name']}** ({p['position']}) — {p['predicted_points']:.1f} pts predicted · £{p['cost']:.1f}M · {p.get('selected_by_percent',0):.1f}% owned")
            lines.append(f"\nWith captain double: **{cap_row['predicted_points']*2:.1f} pts** projected.")
            return "\n".join(lines)

        if any(w in q for w in ["transfer", "sell", "buy", "replace", "upgrade", "swap"]):
            xi_names = set(xi_df["player_name"])
            weakest = xi_df.nsmallest(3, "predicted_points")
            targets = df_all_f[~df_all_f["player_name"].isin(xi_names)].nlargest(6, "predicted_points")
            lines = [f"**🔄 Transfer suggestions:**\n"]
            for _, target in targets.iterrows():
                for _, out in weakest.iterrows():
                    if target["position"] == out["position"]:
                        gain = round(target["predicted_points"] - out["predicted_points"], 1)
                        lines.append(f"**OUT:** {out['player_name']} ({out['predicted_points']:.1f} pts) → **IN:** {target['player_name']} ({target['predicted_points']:.1f} pts, £{target['cost']:.1f}M) | {chr(43) if gain>0 else ''}{gain:.1f} pts gain")
                        break
                if len(lines) >= 4: break
            return "\n".join(lines)

        if any(w in q for w in ["differential", "low ownership", "unique", "diff"]):
            diffs = find_differentials(df_all_f, threshold=5.0, top_n=5)
            lines = ["**🎯 Best differentials (under 5% owned):**\n"]
            for _, p in diffs.iterrows():
                lines.append(f"• **{p['player_name']}** ({p['position']}) — {p['predicted_points']:.1f} pts · £{p['cost']:.1f}M · **{p['selected_by_percent']:.1f}%** owned")
            lines.append("\nLow ownership = big rank gain potential if they deliver.")
            return "\n".join(lines)

        if any(w in q for w in ["weakness", "weak", "worst", "problem"]):
            weakest = xi_df.nsmallest(3, "predicted_points")
            lines = ["**⚠️ Squad weaknesses:**\n"]
            for _, p in weakest.iterrows():
                lines.append(f"• **{p['player_name']}** ({p['position']}) — only {p['predicted_points']:.1f} predicted pts")
            lines.append(f"\nConsider upgrading these positions first.")
            return "\n".join(lines)

        if any(w in q for w in ["value", "cheap", "bargain", "hidden gem"]):
            val = df_all_f.assign(vpp=df_all_f["predicted_points"] / df_all_f["cost"].clip(lower=1))
            top_val = val[df_all_f["minutes"] >= 500].nlargest(5, "vpp")
            lines = ["**💎 Best value picks (pts per £M):**\n"]
            for _, p in top_val.iterrows():
                lines.append(f"• **{p['player_name']}** ({p['position']}) — {p['predicted_points']:.1f} pts · £{p['cost']:.1f}M · **{p['vpp']:.2f} pts/£M**")
            return "\n".join(lines)

        if any(w in q for w in ["analyse", "analyze", "analysis", "overview", "full"]):
            pos_pts = xi_df.groupby("position")["predicted_points"].sum()
            lines = [f"**📊 Squad analysis — {gw_label}**\n",
                     f"**Formation:** {fmt_formation(summary['formation'])} | **Cost:** £{summary['squad_cost']}M | **XI pts:** {summary['xi_pts']}",
                     "\n**By position:**"]
            for pos in ["GK","DEF","MID","FWD"]:
                if pos in pos_pts: lines.append(f"• {pos}: {pos_pts[pos]:.1f} pts")
            lines.append(f"\n**Captain:** {summary['captain']} → {cap_row['predicted_points']*2:.1f} pts with double")
            return "\n".join(lines)

        if any(w in q for w in ["trending", "popular", "transfer in", "hot"]):
            if "transfer_momentum" in df_all_f.columns:
                trending = df_all_f[df_all_f["transfer_momentum"] > 0].nlargest(5, "predicted_points")
                lines = ["**📈 Trending players:**\n"]
                for _, p in trending.iterrows():
                    lines.append(f"• **{p['player_name']}** ({p['position']}) — {p['predicted_points']:.1f} pts · £{p['cost']:.1f}M ▲")
                return "\n".join(lines)

        xi_top = ", ".join(xi_df.nlargest(3,"predicted_points")["player_name"].tolist())
        return (
            f"**Your squad — {gw_label}:**\n\n"
            f"• **Captain:** {summary['captain']} ({cap_row['predicted_points']:.1f} pts → {cap_row['predicted_points']*2:.1f} with double)\n"
            f"• **XI predicted:** {summary['xi_pts']} pts | **Cost:** £{summary['squad_cost']}M\n"
            f"• **Top players:** {xi_top}\n\n"
            f"Try: *'Who should I captain?'* · *'Suggest transfers'* · *'Best differentials?'* · *'Analyse my squad'*"
        )

    def try_ollama(question: str, context: str, model: str = "llama3") -> str | None:
        try:
            payload = {
                "model": model,
                "prompt": (
                    "You are an FPL analyst. Use this squad data to answer concisely (under 150 words, bullet points).\n\n"
                    f"{context}\n\nQuestion: {question}"
                ),
                "stream": False,
            }
            resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=15)
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
        except Exception:
            pass
        return None

    def call_ai_free(user_message: str, context: str, use_ollama: bool, ollama_model: str) -> str:
        if use_ollama:
            ollama_resp = try_ollama(user_message, context, ollama_model)
            if ollama_resp:
                return ollama_resp
            return rule_based_ai(user_message, xi, bench, df_filtered) + "\n\n*⚠️ Ollama not reachable — using built-in AI.*"
        return rule_based_ai(user_message, xi, bench, df_filtered)

    rag_context = build_rag_context(xi, bench, df_filtered)
    ai_left, ai_right = st.columns([1.25, 1], gap="large")

    with ai_left:
        st.markdown("<div class='sh'>🤖 AI Advisor</div>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='font-size:.75rem;color:rgba(255,255,255,.38);margin-bottom:.7rem'>"
            f"Free AI · {gw_label} · {len(df_filtered)} players analysed</p>",
            unsafe_allow_html=True,
        )
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(f"<div class='ul'>You</div><div class='chat-user'>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                content = msg["content"].replace("\n", "<br>").replace("**", "<b>", 1)
                import re
                content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', msg["content"].replace("\n","<br>"))
                st.markdown(f"<div class='cl'>⚽ FPL AI</div><div class='chat-ai'>{content}</div>", unsafe_allow_html=True)
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("", placeholder="e.g. 'Who should I captain?' · 'Suggest 2 transfers' · 'Best differentials?'", label_visibility="collapsed")
            s_col, c_col = st.columns([3, 1])
            with s_col: send  = st.form_submit_button("Send  →", use_container_width=True)
            with c_col: clear = st.form_submit_button("Clear",   use_container_width=True)
        if clear:
            st.session_state["chat_history"] = []; st.rerun()
        if send and user_input.strip():
            use_ol = st.session_state.get("use_ollama", False)
            ol_mdl = st.session_state.get("ollama_model", "llama3")
            st.session_state["chat_history"].append({"role":"user","content":user_input.strip()})
            with st.spinner("Thinking…"):
                reply = call_ai_free(user_input.strip(), rag_context, use_ol, ol_mdl)
            st.session_state["chat_history"].append({"role":"assistant","content":reply})
            st.rerun()

    with ai_right:
        st.markdown("<div class='sh'>💬 Quick Prompts</div>", unsafe_allow_html=True)
        for prompt_text, icon in [
            ("Who should I captain?", "©"), ("Suggest 2 transfers", "🔄"),
            ("Best differentials?", "🎯"), ("Analyse my squad", "📊"),
            ("Squad weaknesses?", "⚠️"), ("Best value picks?", "💎"),
            ("Which players are trending?", "📈"),
        ]:
            if st.button(f"{icon}  {prompt_text}", key=f"qp_{hash(prompt_text)}", use_container_width=True):
                use_ol = st.session_state.get("use_ollama", False)
                ol_mdl = st.session_state.get("ollama_model", "llama3")
                st.session_state["chat_history"].append({"role":"user","content":prompt_text})
                with st.spinner("Thinking…"):
                    reply = call_ai_free(prompt_text, rag_context, use_ol, ol_mdl)
                st.session_state["chat_history"].append({"role":"assistant","content":reply})
                st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🦙 Local LLM (Ollama) — optional", expanded=False):
            st.markdown("<div style='font-size:.72rem;color:rgba(255,255,255,.5);line-height:1.7'>Upgrade to a local LLM.<br>1. Install <a href='https://ollama.ai' style='color:#00e87a'>ollama.ai</a><br>2. Run: <code>ollama pull llama3</code><br>3. Enable below</div>", unsafe_allow_html=True)
            use_ollama = st.toggle("Use Ollama LLM", value=False, key="use_ollama")
            ollama_model = st.selectbox("Model", ["llama3","mistral","phi3","llama3.2"], key="ollama_model")
            if use_ollama:
                if try_ollama("ping","test",ollama_model): st.success(f"✓ {ollama_model} connected")
                else: st.warning("⚠️ Ollama not running — using built-in AI")
        st.markdown(
            f"<div style='background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:8px;padding:.7rem .9rem;font-size:.73rem;color:rgba(255,255,255,.45);line-height:1.8;margin-top:.5rem'>"
            f"<strong style='color:rgba(255,255,255,.68)'>Context loaded:</strong><br>"
            f"✓ {len(xi)} starters · {len(bench)} bench<br>✓ {len(df_filtered)} players<br>"
            f"✓ {gw_label} · {fmt_formation(summary['formation'])}<br>✓ Captain: {summary['captain']}</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_about:
    ab1, ab2 = st.columns([1, 1], gap="large")
    with ab1:
        st.markdown("<div class='sh'>⚽ About This App</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.09);border-radius:10px;padding:1.2rem 1.4rem;font-size:.84rem;line-height:1.8'>
        <p><strong style='color:#00e87a'>FPL AI Optimizer</strong> is a free, open-source Fantasy Premier League assistant that uses machine learning and mathematical optimization to help you build the best possible squad.</p>
        <br>
        <strong style='color:rgba(255,255,255,.8)'>What it does</strong><br>
        • Fetches live player data from the official FPL API<br>
        • Predicts each player's expected points using ML models<br>
        • Selects the optimal 15-player squad using linear programming<br>
        • Provides AI-powered recommendations — completely free<br>
        <br>
        <strong style='color:rgba(255,255,255,.8)'>Data source</strong><br>
        All data comes from the official FPL API at <code>fantasy.premierleague.com</code>. Refreshed daily.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br><div class='sh'>🧠 How Predictions Work</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.09);border-radius:10px;padding:1.2rem 1.4rem;font-size:.84rem;line-height:1.8'>
        <strong style='color:rgba(255,255,255,.8)'>1. Feature engineering</strong><br>
        Raw FPL stats (goals, assists, ICT index, minutes) are transformed into meaningful signals like pts/90, form score, and value ratio.<br><br>
        <strong style='color:rgba(255,255,255,.8)'>2. Machine learning</strong><br>
        Two models are trained — Random Forest and Gradient Boosting. The better-performing model (measured by error rate) is automatically selected.<br><br>
        <strong style='color:rgba(255,255,255,.8)'>3. Optimization</strong><br>
        Integer Linear Programming selects the best 15-player squad that maximises predicted points within the £100M budget and formation constraints.<br><br>
        <strong style='color:rgba(255,255,255,.8)'>4. AI Advisor</strong><br>
        A rule-based engine analyses your squad and answers questions about captaincy, transfers, and differentials using the model's predictions.
        </div>
        """, unsafe_allow_html=True)

    with ab2:
        st.markdown("<div class='sh'>📖 How to Use</div>", unsafe_allow_html=True)
        steps = [
            ("🏟 My Squad", "View your AI-optimised 15-player squad on the pitch. Player photos, predicted points, and captain badge shown."),
            ("💡 Insights", "Auto-generated gameweek insights: best value picks, captain recommendations, and risky high-ceiling players."),
            ("📈 Top Players", "Browse all players ranked by predicted points. Filter by position or search for specific players."),
            ("🎯 Differentials", "Find low-ownership players with strong predicted returns — the picks that separate you from the crowd."),
            ("🤖 AI Advisor", "Chat with the free AI advisor. Ask about transfers, captaincy, weaknesses, or differentials."),
        ]
        for tab_name, desc in steps:
            st.markdown(
                f"<div style='background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.09);"
                f"border-radius:8px;padding:.65rem .9rem;margin-bottom:.5rem;font-size:.83rem'>"
                f"<strong style='color:#00e87a'>{tab_name}</strong><br>"
                f"<span style='color:rgba(255,255,255,.6)'>{desc}</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br><div class='sh'>🔄 Keeping Data Fresh</div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='background:rgba(0,232,122,.06);border:1px solid rgba(0,232,122,.2);border-radius:8px;padding:.8rem 1rem;font-size:.82rem;line-height:1.7'>"
            "Click <strong style='color:#00e87a'>🔄 Refresh Live Data</strong> in the sidebar to fetch the latest player stats from the FPL API. "
            "Data is automatically refreshed daily — your squad predictions update whenever new data is available."
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div style='background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);border-radius:8px;padding:.8rem 1rem;font-size:.78rem;color:rgba(255,255,255,.45);line-height:1.7'>"
            "<strong style='color:rgba(255,255,255,.6)'>Open source · Free forever</strong><br>"
            "No paid APIs required. All predictions run locally. AI advisor works without any API keys.<br><br>"
            "Built with: Python · Streamlit · Scikit-learn · SciPy · Official FPL API"
            "</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"<br><p style='text-align:center;color:rgba(255,255,255,.15);font-size:.68rem'>"
    f"FPL AI Optimizer · Free & Open Source · ML + Linear Programming · {gw_label}</p>",
    unsafe_allow_html=True,
)
