"""
FPL AI Optimizer - Full Squad Optimizer (15 Players)
=====================================================
Two-phase MILP:
  Phase 1: select 15-player squad (2GK,5DEF,5MID,3FWD) within budget
  Phase 2: pick optimal starting XI across all valid formations
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds


VALID_FORMATIONS = [
    (3,4,3),(3,5,2),(4,3,3),(4,4,2),(4,5,1),(5,3,2),(5,4,1)
]


def _pos_mask(df, pos):
    return (df["position"] == pos).values.astype(float)


def _club_rows(df, max_per_club):
    if "team" not in df.columns:
        return []
    rows = []
    for club in df["team"].unique():
        mask = (df["team"] == club).values.astype(float)
        if mask.sum() > max_per_club:
            rows.append(mask)
    return rows


def _solve(pts, costs, A_rows, b_lo, b_hi, budget, n_players):
    n = len(pts)
    ones = np.ones(n)
    full_A = np.vstack([ones, costs] + ([np.array(A_rows)] if A_rows else [])) if A_rows else np.vstack([ones, costs])
    full_lo = [n_players, 0.0] + b_lo
    full_hi = [n_players, budget] + b_hi
    r = milp(-pts, constraints=LinearConstraint(full_A, full_lo, full_hi),
             integrality=np.ones(n), bounds=Bounds(0, 1))
    return np.round(r.x).astype(int) if r.success else None


def _select_full_squad(df, budget, max_per_club):
    df = df.reset_index(drop=True)
    n = len(df)
    pts = df["predicted_points"].values.astype(float)
    costs = df["cost"].values.astype(float)

    A_rows, b_lo, b_hi = [], [], []
    def eq(row, v): A_rows.append(row); b_lo.append(v); b_hi.append(v)
    def ineq(row, lo, hi): A_rows.append(row); b_lo.append(lo); b_hi.append(hi)

    eq(_pos_mask(df,"GK"), 2)
    eq(_pos_mask(df,"DEF"), 5)
    eq(_pos_mask(df,"MID"), 5)
    eq(_pos_mask(df,"FWD"), 3)
    for row in _club_rows(df, max_per_club):
        ineq(row, 0, max_per_club)

    sol = _solve(pts, costs, A_rows, b_lo, b_hi, budget, 15)
    if sol is None:
        return _greedy_squad(df, budget, max_per_club)
    return df.iloc[np.where(sol == 1)[0]].copy().reset_index(drop=True)


def _pick_starting_xi(squad):
    squad = squad.reset_index(drop=True)
    pts = squad["predicted_points"].values.astype(float)
    n = len(squad)

    best_pts = -np.inf
    best_xi_idx = None
    best_formation = "1-4-4-2"

    for nd, nm, nf in VALID_FORMATIONS:
        A_rows, b_lo, b_hi = [], [], []
        def eq(row, v): A_rows.append(row); b_lo.append(v); b_hi.append(v)
        eq(_pos_mask(squad,"GK"), 1)
        eq(_pos_mask(squad,"DEF"), nd)
        eq(_pos_mask(squad,"MID"), nm)
        eq(_pos_mask(squad,"FWD"), nf)
        sol = _solve(pts, np.zeros(n), A_rows, b_lo, b_hi, 1e9, 11)
        if sol is None:
            continue
        total = pts @ sol
        if total > best_pts:
            best_pts = total
            best_xi_idx = np.where(sol == 1)[0]
            best_formation = f"1-{nd}-{nm}-{nf}"

    if best_xi_idx is None:
        gk_idx = squad[squad["position"]=="GK"]["predicted_points"].idxmax()
        rest = squad[squad["position"]!="GK"].nlargest(10,"predicted_points").index.tolist()
        best_xi_idx = [gk_idx] + rest

    xi_idx = set(best_xi_idx)
    bench_idx = sorted(set(range(n)) - xi_idx)

    xi = squad.iloc[sorted(xi_idx)].copy()
    bench = squad.iloc[bench_idx].copy()
    bench_gk = bench[bench["position"]=="GK"]
    bench_out = bench[bench["position"]!="GK"].sort_values("predicted_points", ascending=False)
    bench = pd.concat([bench_gk, bench_out]).reset_index(drop=True)

    return xi, bench, best_formation


def _greedy_squad(df, budget, max_per_club):
    quotas = {"GK":2,"DEF":5,"MID":5,"FWD":3}
    chosen, spent = [], 0.0
    for pos, q in quotas.items():
        pool = df[df["position"]==pos].sort_values("predicted_points", ascending=False)
        taken = 0
        for _, row in pool.iterrows():
            if taken >= q: break
            if spent + row["cost"] <= budget:
                chosen.append(row); spent += row["cost"]; taken += 1
    return pd.DataFrame(chosen).reset_index(drop=True)


def _assign_captain(xi):
    xi = xi.copy()
    xi["is_captain"] = False
    xi["is_vice_captain"] = False
    if xi.empty:
        return xi
    ranked = xi.sort_values("predicted_points", ascending=False)
    xi.loc[ranked.index[0], "is_captain"] = True
    if len(ranked) > 1:
        xi.loc[ranked.index[1], "is_vice_captain"] = True
    return xi


def optimise_squad(df, budget=100.0, max_per_club=3):
    """
    Main API: returns dict with keys squad, xi, bench, formation, summary.
    """
    squad = _select_full_squad(df, budget, max_per_club)
    xi, bench, formation = _pick_starting_xi(squad)
    xi = _assign_captain(xi)
    xi["role"] = "starter"
    bench["role"] = "bench"
    squad_full = pd.concat([xi, bench]).reset_index(drop=True)
    summary = _build_summary(xi, bench, squad_full, formation)
    return {"squad": squad_full, "xi": xi.reset_index(drop=True),
            "bench": bench.reset_index(drop=True), "formation": formation, "summary": summary}


def _build_summary(xi, bench, squad, formation):
    cap  = xi[xi["is_captain"]].iloc[0]       if xi["is_captain"].any()       else None
    vice = xi[xi["is_vice_captain"]].iloc[0]  if xi["is_vice_captain"].any()  else None
    xi_pts = round(xi["predicted_points"].sum(), 1)
    bench_pts = round(bench["predicted_points"].sum(), 1)
    squad_cost = round(squad["cost"].sum(), 1)
    return {
        "squad_cost": squad_cost,
        "xi_cost": round(xi["cost"].sum(), 1),
        "budget_remaining": round(100.0 - squad_cost, 1),
        "xi_pts": xi_pts,
        "bench_pts": bench_pts,
        "total_pts": round(xi_pts + bench_pts * 0.1, 1),
        "captain": cap["player_name"]  if cap  is not None else "—",
        "vice_captain": vice["player_name"] if vice is not None else "—",
        "formation": formation,
        "n_squad": len(squad),
    }


# ── Insights helpers ────────────────────────────────────────────────────────

def find_differentials(df, threshold=5.0, top_n=8):
    diffs = df[df["selected_by_percent"] < threshold].copy()
    diffs["diff_score"] = diffs["predicted_points"] / diffs["cost"].clip(lower=1)
    return diffs.sort_values("diff_score", ascending=False).head(top_n)


def best_value_picks(df, top_n=10):
    pool = df[df["minutes"] >= 500].copy()
    pool["value_index"] = pool["predicted_points"] / pool["cost"].clip(lower=1)
    return pool.sort_values("value_index", ascending=False).head(top_n)


def risky_picks(df, top_n=6):
    pool = df[df["predicted_points"] > df["predicted_points"].quantile(0.7)].copy()
    pool["consistency"] = pool["pts_per_90"] / pool["predicted_points"].clip(lower=0.1)
    return pool.sort_values("consistency").head(top_n)


def captain_candidates(xi, top_n=3):
    cols = [c for c in ["player_name","position","predicted_points",
                         "pts_per_90","form_score","ict_index",
                         "selected_by_percent","cost"] if c in xi.columns]
    return xi[cols].sort_values("predicted_points", ascending=False).head(top_n)


def generate_key_insights(df, xi):
    insights = []
    pos_avg = df.groupby("position")["predicted_points"].mean()
    top_pos = pos_avg.idxmax()
    insights.append(
        f"🏆 **{top_pos}s dominate** with {pos_avg[top_pos]:.1f} pts avg predicted — "
        f"load up on this position if you can."
    )
    value = df.assign(vpp=df["predicted_points"]/df["cost"].clip(lower=1))
    bv = value.nlargest(1,"vpp").iloc[0]
    insights.append(
        f"💎 **Best value pick:** {bv['player_name']} ({bv['position']}) — "
        f"{bv['predicted_points']:.1f} pts at £{bv['cost']:.1f}M = {bv['vpp']:.1f} pts/£M."
    )
    low = df[df["selected_by_percent"] < 5].nlargest(1,"predicted_points")
    if not low.empty:
        p = low.iloc[0]
        insights.append(
            f"🎯 **Differential alert:** {p['player_name']} ({p['position']}) — "
            f"{p['predicted_points']:.1f} pts but only {p['selected_by_percent']:.1f}% owned. "
            f"Perfect captaincy differential."
        )
    if not xi.empty:
        cap_row = xi.sort_values("predicted_points", ascending=False).iloc[0]
        insights.append(
            f"© **Captain:** {cap_row['player_name']} tops the AI ranking at "
            f"{cap_row['predicted_points']:.1f} pts. With captain double: "
            f"**{cap_row['predicted_points']*2:.1f} pts** projected."
        )
    cs_pool = df[df["position"].isin(["DEF","GK"])].nlargest(1,"cs_rate")
    if not cs_pool.empty:
        p = cs_pool.iloc[0]
        insights.append(
            f"🛡️ **Clean sheet favourite:** {p['player_name']} ({p['position']}) "
            f"has a {p['cs_rate']*100:.0f}% clean sheet rate — strong defensive value."
        )
    mid_pool = df[df["position"]=="MID"].nlargest(1,"creativity_norm")
    if not mid_pool.empty:
        p = mid_pool.iloc[0]
        insights.append(
            f"✨ **Creativity king:** {p['player_name']} (MID) leads all midfielders "
            f"in creativity — prime assist candidate."
        )
    return insights


# ── Legacy shims ─────────────────────────────────────────────────────────────

def optimise_team(df, budget=100.0, **kwargs):
    return optimise_squad(df, budget=budget, max_per_club=kwargs.get("max_per_club",3))["xi"]


def team_summary(selected):
    cap  = selected[selected.get("is_captain",      pd.Series([False]*len(selected))).astype(bool)].head(1)
    vice = selected[selected.get("is_vice_captain",  pd.Series([False]*len(selected))).astype(bool)].head(1)
    counts = selected["position"].value_counts()
    return {
        "total_cost":   round(selected["cost"].sum(), 1),
        "total_pts":    round(selected["predicted_points"].sum(), 1),
        "captain":      cap.iloc[0]["player_name"]  if not cap.empty  else "—",
        "vice_captain": vice.iloc[0]["player_name"] if not vice.empty else "—",
        "formation":    f"1-{counts.get('DEF',0)}-{counts.get('MID',0)}-{counts.get('FWD',0)}",
        "n_players":    len(selected),
    }
