"""
FPL AI Optimizer Dashboard  ·  v3 – Live Pipeline Edition
===========================================================
streamlit run app.py

Architecture
------------
  data/fpl_api.py     → Live FPL API client (bootstrap-static + fixtures)
  data/loader.py      → Smart loader (live CSV first, static fallback)
  models/predictor.py → Gradient Boosting + Random Forest models
  models/model_store.py → Pickle cache (retrain only when data changes)
  optimizer/team_selector.py → Two-phase MILP squad builder
  update_data.py      → CLI automation script (cron / APScheduler)

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
# Cloud startup — ensure data is available before the app renders
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
    st.markdown("<h2 style='color:#00e87a;margin-bottom:0'>⚽ FPL AI</h2>", unsafe_allow_html=True)
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
    f"<h1 style='text-align:center;color:#00e87a;margin-bottom:3px'>FPL Helper Dashboard</h1>"
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
    (k3, "Formation",    summary["formation"],                       "Auto-optimised LP"),
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
tab_squad, tab_insights, tab_top, tab_diff, tab_ai, tab_pipeline = st.tabs([
    "🏟 My Squad", "💡 Insights", "📈 Top Players",
    "🎯 Differentials", "🤖 AI Advisor", "⚙️ Pipeline",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MY SQUAD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_squad:
    col_pitch, col_list = st.columns([1.1, 1], gap="large")

    # ── Pitch ─────────────────────────────────────────────────────────────────
    with col_pitch:
        st.markdown("<div class='sh'>🏟 Starting XI + Bench</div>", unsafe_allow_html=True)

        def draw_pitch(xi_df: pd.DataFrame, bench_df: pd.DataFrame):
            fig, ax = plt.subplots(figsize=(6, 9.5))
            fig.patch.set_facecolor("#0d3b1e")
            ax.set_facecolor("#1a6b2a")
            ax.set_xlim(0, 100); ax.set_ylim(-48, 162); ax.axis("off")

            # Pitch lines
            for r in [
                dict(xy=(5,5),   width=90, height=150, fill=False, ec="white", lw=1.1, alpha=.5),
                dict(xy=(25,5),  width=50, height=22,  fill=False, ec="white", lw=.7,  alpha=.4),
                dict(xy=(25,133),width=50, height=22,  fill=False, ec="white", lw=.7,  alpha=.4),
                dict(xy=(30,5),  width=40, height=10,  fill=False, ec="white", lw=.7,  alpha=.35),
                dict(xy=(30,140),width=40, height=10,  fill=False, ec="white", lw=.7,  alpha=.35),
            ]:
                ax.add_patch(patches.Rectangle(**r))
            ax.add_patch(patches.Circle((50, 80), 17, fill=False, ec="white", lw=.7, alpha=.4))
            ax.plot([5, 95], [80, 80], "white", lw=.7, alpha=.35)

            # Player circles
            groups = {"GK": [], "DEF": [], "MID": [], "FWD": []}
            for _, p in xi_df.iterrows():
                groups[p["position"]].append(p)

            y_pos = {"GK": 19, "DEF": 54, "MID": 95, "FWD": 134}
            R = 20

            for pos, players in groups.items():
                if not players: continue
                n  = len(players)
                xs = np.linspace(14, 86, n)
                y  = y_pos[pos]
                c  = POS_COLORS[pos]

                for x, p in zip(xs, players):
                    if p.get("is_captain"):
                        ax.add_patch(plt.Circle((x, y), R+6, color="#ffd700", alpha=.18, zorder=4))
                    ax.add_patch(plt.Circle((x, y), R, color=c, zorder=5))
                    ax.add_patch(plt.Circle((x, y), R, fill=False, ec="white", lw=1.1, alpha=.65, zorder=6))

                    parts = p["player_name"].split()
                    init = (parts[0][0] + parts[-1][0]).upper() if len(parts) > 1 else parts[0][:2].upper()
                    ax.text(x, y, init, ha="center", va="center",
                            fontsize=7, fontweight="bold", color="#0d1a0d", zorder=7)

                    surname = p["player_name"].split()[-1][:11]
                    ax.text(x, y - R - 7, surname, ha="center", va="top",
                            fontsize=5.5, color="white", fontweight="600", zorder=7)

                    ax.text(x, y + R + 6, f"{p['predicted_points']:.1f}", ha="center", va="bottom",
                            fontsize=5.8, color=c, fontweight="bold", zorder=7)

                    if p.get("is_captain"):
                        ax.text(x + R - 2, y - R + 5, "C", fontsize=6.5, color="#ffd700", fontweight="bold", zorder=8)
                    elif p.get("is_vice_captain"):
                        ax.text(x + R - 2, y - R + 5, "V", fontsize=6.5, color="#ccc", fontweight="bold", zorder=8)

                    # Injury dot
                    status_p = str(p.get("status", "a"))
                    if status_p in ("i", "s"):
                        ax.add_patch(plt.Circle((x - R + 3, y + R - 4), 4, color="#ff4d6d", zorder=8))
                    elif status_p == "d":
                        ax.add_patch(plt.Circle((x - R + 3, y + R - 4), 4, color="#ffc800", zorder=8))

            # Bench section
            ax.add_patch(patches.FancyBboxPatch((2, -42), 96, 36,
                boxstyle="round,pad=1", fc=(0, 0, 0, .38), ec=(1, 1, 1, .18), lw=.8, zorder=3))
            ax.text(50, -9, "BENCH", ha="center", va="top",
                    fontsize=6, color=(1, 1, 1, .38), fontweight="600", zorder=4)

            BR = 13; by = -26
            for x, (_, bp) in zip(np.linspace(14, 86, 4), bench_df.iterrows()):
                c = POS_COLORS[bp["position"]]
                ax.add_patch(plt.Circle((x, by), BR, color=c, alpha=.5, zorder=4))
                ax.add_patch(plt.Circle((x, by), BR, fill=False, ec="white", lw=.8, alpha=.45, zorder=5))
                parts = bp["player_name"].split()
                init = (parts[0][0] + parts[-1][0]).upper() if len(parts) > 1 else parts[0][:2].upper()
                ax.text(x, by, init, ha="center", va="center",
                        fontsize=5.5, fontweight="bold", color="#0d1a0d", zorder=6)
                ax.text(x, by - BR - 4, bp["player_name"].split()[-1][:10],
                        ha="center", va="top", fontsize=4.8, color=(1, 1, 1, .55), zorder=6)
                ax.text(x, by + BR + 4, f"{bp['predicted_points']:.1f}",
                        ha="center", va="bottom", fontsize=4.8, color=c, fontweight="bold", zorder=6)

            return fig

        fig = draw_pitch(xi, bench)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

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
# TAB 5 — AI ADVISOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ai:

    def build_rag_context(xi_df, bench_df, df_all_f) -> str:
        lines = [
            f"=== FPL AI OPTIMIZER CONTEXT ===",
            f"Gameweek: {gw_label} | Data: {status['data_source']} | Updated: {status['age_str']}",
            "",
            "STARTING XI:",
        ]
        for _, p in xi_df.iterrows():
            badge = " [CAPTAIN]" if p.get("is_captain") else (" [VC]" if p.get("is_vice_captain") else "")
            st_flag = "" if str(p.get("status","a")) == "a" else f" [{str(p.get('status','a')).upper()}]"
            lines.append(
                f"  {p['position']} | {p['player_name']}{badge}{st_flag} | "
                f"£{p['cost']:.1f}M | {p['predicted_points']:.1f} pred pts | "
                f"G:{int(p.get('goals_scored',0))} A:{int(p.get('assists',0))} | "
                f"ICT:{p.get('ict_index',0):.0f} | Owned:{p.get('selected_by_percent',0):.1f}%"
            )
        lines += ["", "BENCH:"]
        for _, p in bench_df.iterrows():
            lines.append(f"  {p['position']} | {p['player_name']} | £{p['cost']:.1f}M | {p['predicted_points']:.1f} pts")

        lines += ["", "=== TOP 10 PLAYERS (transfer targets) ==="]
        for _, p in df_all_f.nlargest(10, "predicted_points").iterrows():
            lines.append(
                f"  {p['position']} | {p['player_name']} | £{p['cost']:.1f}M | "
                f"{p['predicted_points']:.1f} pts | {p.get('selected_by_percent',0):.1f}% owned"
            )

        val = df_all_f.assign(vpp=df_all_f["predicted_points"] / df_all_f["cost"].clip(lower=1))
        lines += ["", "=== BEST VALUE (pts/£M) ==="]
        for _, p in val.nlargest(5, "vpp").iterrows():
            lines.append(f"  {p['player_name']} ({p['position']}) — {p['vpp']:.2f} pts/£M")

        diffs_ctx = find_differentials(df_all_f, threshold=5.0, top_n=5)
        lines += ["", "=== TOP DIFFERENTIALS (< 5% owned) ==="]
        for _, p in diffs_ctx.iterrows():
            lines.append(f"  {p['player_name']} ({p['position']}) — {p['predicted_points']:.1f} pts, {p['selected_by_percent']:.1f}% owned")

        return "\n".join(lines)

    SYSTEM_PROMPT = (
        "You are an expert Fantasy Premier League (FPL) analyst and personal squad advisor. "
        "You have been given the user's current 15-player squad (starting XI + bench), "
        "live player statistics from the official FPL API, and AI-predicted points for the next gameweek. "
        "Be concise, direct, and data-driven. Reference specific player names and their stats. "
        "Use bullet points for lists. Respond like a knowledgeable friend, not a formal report. "
        "Keep replies under 220 words unless asked for more. "
        "Always factor in the current gameweek context and player availability status."
    )

    def _get_api_key() -> str | None:
        """
        Resolve Anthropic API key with priority:
          1. Streamlit secrets   (st.secrets — works on Streamlit Cloud)
          2. Environment variable (works locally / GitHub Actions)
        """
        try:
            return st.secrets["ANTHROPIC_API_KEY"]
        except Exception:
            import os
            return os.environ.get("ANTHROPIC_API_KEY")

    def call_ai(user_message: str, context: str) -> str:
        api_key = _get_api_key()
        if not api_key:
            return (
                f"⚠️ **AI Advisor not configured.**\n\n"
                f"To enable the AI advisor:\n"
                f"- **Streamlit Cloud:** go to your app → ⋮ → Settings → Secrets, "
                f"and add `ANTHROPIC_API_KEY = \"sk-ant-...\"` \n"
                f"- **Local:** set the environment variable `ANTHROPIC_API_KEY=sk-ant-...`\n\n"
                f"**Your squad summary — {gw_label}:**\n"
                f"• Captain: **{summary['captain']}** · {summary['xi_pts']} XI pts predicted\n"
                f"• Squad: £{summary['squad_cost']}M · Formation {summary['formation']}\n"
                f"• Data: {status['data_source']} · {status['age_str']}"
            )
        try:
            payload = {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 650,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": f"{context}\n\n---\n\n{user_message}"}],
            }
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                json=payload, timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]
        except Exception as exc:
            return (
                f"⚠️ AI response failed ({type(exc).__name__}). "
                f"Check your API key in Streamlit secrets.\n\n"
                f"**Squad summary:** Captain **{summary['captain']}** · "
                f"£{summary['squad_cost']}M · {summary['formation']}"
            )

    rag_context = build_rag_context(xi, bench, df_filtered)

    ai_left, ai_right = st.columns([1.25, 1], gap="large")

    with ai_left:
        st.markdown("<div class='sh'>🤖 AI Advisor</div>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='font-size:.75rem;color:rgba(255,255,255,.38);margin-bottom:.7rem'>"
            f"Powered by {gw_label} live data · {len(df_filtered)} players · "
            f"Updated {status['age_str']}</p>",
            unsafe_allow_html=True,
        )

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='ul'>You</div><div class='chat-user'>{msg['content']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='cl'>⚽ FPL AI</div><div class='chat-ai'>{msg['content']}</div>",
                    unsafe_allow_html=True,
                )

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "", placeholder="e.g. 'Why is Haaland captain?' · 'Suggest 2 transfers' · 'Best differential?'",
                label_visibility="collapsed",
            )
            s_col, c_col = st.columns([3, 1])
            with s_col:
                send  = st.form_submit_button("Send  →", use_container_width=True)
            with c_col:
                clear = st.form_submit_button("Clear",   use_container_width=True)

        if clear:
            st.session_state["chat_history"] = []
            st.rerun()

        if send and user_input.strip():
            st.session_state["chat_history"].append({"role": "user", "content": user_input.strip()})
            with st.spinner("Thinking…"):
                reply = call_ai(user_input.strip(), rag_context)
            st.session_state["chat_history"].append({"role": "assistant", "content": reply})
            st.rerun()

    with ai_right:
        st.markdown("<div class='sh'>💬 Quick Prompts</div>", unsafe_allow_html=True)
        cap_name = summary.get("captain", "the captain").split()[-1]
        quick = [
            (f"Why is {cap_name} the captain pick?",          "©"),
            ("Suggest 2 transfers to improve my squad",        "🔄"),
            ("Best differential picks this gameweek?",         "🎯"),
            ("What is my squad's biggest weakness?",           "⚠️"),
            ("Full team analysis — strengths and weaknesses",  "📊"),
            ("Who should I sell to free up budget?",           "💰"),
            ("Best captain if my first choice blanks?",        "🔀"),
            ("Which trending players should I buy?",           "📈"),
        ]
        for prompt_text, icon in quick:
            if st.button(f"{icon}  {prompt_text}", key=f"qp_{hash(prompt_text)}", use_container_width=True):
                st.session_state["chat_history"].append({"role": "user", "content": prompt_text})
                with st.spinner("Thinking…"):
                    reply = call_ai(prompt_text, rag_context)
                st.session_state["chat_history"].append({"role": "assistant", "content": reply})
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);"
            f"border-radius:8px;padding:.7rem .9rem;font-size:.73rem;color:rgba(255,255,255,.45);line-height:1.8'>"
            f"<strong style='color:rgba(255,255,255,.68)'>RAG Context loaded:</strong><br>"
            f"✓ {len(xi)} starters · {len(bench)} bench players<br>"
            f"✓ {len(df_filtered)} players with predictions<br>"
            f"✓ {status['data_source']} · {status['age_str']}<br>"
            f"✓ {gw_label} · {status['n_players']} total players<br>"
            f"✓ Model: {status['model_name']}"
            f"</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — PIPELINE STATUS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pipeline:
    st.markdown("<div class='sh'>⚙️ Live Data Pipeline</div>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown("##### 🔌 Data Source")
        st.markdown(
            f"<div class='pipeline-box'>"
            f"<strong>FPL API</strong><br>"
            f"Endpoint: bootstrap-static/<br>"
            f"Status: {status['data_source']}<br>"
            f"Last fetch: {status['age_str']}<br>"
            f"Players fetched: {status['n_players']}<br>"
            f"Gameweek: {gw_label}<br>"
            f"Deadline: {_fmt_deadline(status['gw_deadline'])}<br>"
            f"Stale after: {DATA_STALE_HOURS}h"
            f"</div>",
            unsafe_allow_html=True,
        )
        if st.button("🔄 Fetch Now", key="fetch_now_btn"):
            with st.spinner("Fetching from FPL API…"):
                ok, msg = do_live_refresh()
            (st.success if ok else st.warning)(msg)
            if ok:
                st.cache_data.clear()
                st.rerun()

    with p2:
        st.markdown("##### 🧠 ML Pipeline")
        best_model_name = status["model_name"]
        best_metrics = ml_metrics.get(best_model_name, {})
        trained_ago = _fmt_trained_at(status["model_trained_at"])
        st.markdown(
            f"<div class='pipeline-box'>"
            f"<strong>Model Cache</strong><br>"
            f"Active model: {best_model_name}<br>"
            f"Trained: {trained_ago}<br>"
            f"On: {status.get('model_n_players','?')} players<br>"
            f"Cache TTL: 24h<br>"
            f"MAE: {best_metrics.get('MAE','—')}<br>"
            f"RMSE: {best_metrics.get('RMSE','—')}<br>"
            f"Retrain trigger: data change"
            f"</div>",
            unsafe_allow_html=True,
        )
        if st.button("🔁 Force Retrain", key="retrain_btn"):
            with st.spinner("Retraining models…"):
                df_rt = load_and_prepare(prefer_live=True, min_minutes=0)
                df_rt, pred_rt, met_rt = run_ml_pipeline(df_rt, get_feature_columns())
                save_predictor(pred_rt, df_rt)
                st.cache_data.clear()
            st.success(f"Retrained {pred_rt.best_name}. MAE: {met_rt[pred_rt.best_name]['MAE']:.4f}")
            st.rerun()

    with p3:
        st.markdown("##### 🗂 Storage Paths")
        from data.loader import PROCESSED_PATH, STATIC_PATH
        pred_path = ROOT / "data" / "predictions" / "latest.csv"
        meta_path = ROOT / "data" / "cache" / "metadata.json"
        model_pkl = ROOT / "models" / "saved" / "predictor.pkl"

        def file_info(p: Path) -> str:
            if p.exists():
                size_kb = p.stat().st_size / 1024
                return f"✓ {size_kb:.0f} KB"
            return "✗ Not found"

        st.markdown(
            f"<div class='pipeline-box'>"
            f"<strong>Files</strong><br>"
            f"Processed CSV: {file_info(PROCESSED_PATH)}<br>"
            f"Static CSV: {file_info(STATIC_PATH)}<br>"
            f"Predictions: {file_info(pred_path)}<br>"
            f"API metadata: {file_info(meta_path)}<br>"
            f"Model pickle: {file_info(model_pkl)}<br>"
            f"Log dir: {file_info(ROOT/'logs')}"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("##### 🕐 Automation")
    auto1, auto2 = st.columns(2)
    with auto1:
        st.markdown(
            "**Option A — cron (Linux/macOS)**\n"
            "```\n# Every day at 06:00 UTC\n"
            "0 6 * * * cd /path/to/fpl_optimizer && "
            "python update_data.py >> logs/cron.log 2>&1\n```\n\n"
            "**Option B — Windows Task Scheduler**\n"
            "```\nschtasks /create /tn \"FPL Update\" \\\n"
            "  /tr \"python C:\\path\\update_data.py\" \\\n"
            "  /sc DAILY /st 06:00\n```"
        )
    with auto2:
        st.markdown(
            "**Option C — APScheduler (long-running process)**\n"
            "```bash\npython update_data.py --schedule --interval 6\n```\n\n"
            "**Manual CLI commands**\n"
            "```bash\n# Check status\npython update_data.py --check-only\n\n"
            "# Force full refresh\npython update_data.py --force\n\n"
            "# Retrain model only\npython models/train.py --force\n```"
        )

    # Recent log entries
    st.markdown("---")
    st.markdown("##### 📋 Recent Pipeline Runs")
    log_path = ROOT / "logs" / "pipeline_runs.jsonl"
    if log_path.exists():
        try:
            lines = log_path.read_text().strip().splitlines()
            last_5 = [json.loads(l) for l in lines[-5:]]
            for run in reversed(last_5):
                run_at = run.get("run_at", "?")[:19]
                f_stat = run.get("fetch", {}).get("status", "?")
                t_stat = run.get("train", {}).get("status", "?")
                elapsed = run.get("elapsed_sec", "?")
                icon_f = "✅" if f_stat in ("fetched","skipped") else "❌"
                icon_t = "✅" if t_stat in ("trained","skipped") else "❌"
                st.markdown(
                    f"<div style='font-size:.73rem;color:rgba(255,255,255,.5);padding:.3rem 0;border-bottom:1px solid rgba(255,255,255,.06)'>"
                    f"<b style='color:rgba(255,255,255,.7)'>{run_at}</b> · "
                    f"{icon_f} Fetch: {f_stat} · {icon_t} Train: {t_stat} · {elapsed}s"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.caption(f"Could not parse log: {e}")
    else:
        st.markdown(
            "<p style='font-size:.75rem;color:rgba(255,255,255,.35)'>"
            "No pipeline runs logged yet. Run <code>python update_data.py</code> to start.</p>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"<br><p style='text-align:center;color:rgba(255,255,255,.15);font-size:.68rem'>"
    f"FPL AI Optimizer v3 · Live API + Gradient Boosting + Integer LP · "
    f"{gw_label} · Last updated {status['age_str']}</p>",
    unsafe_allow_html=True,
)
