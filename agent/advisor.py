from __future__ import annotations
import json
import requests
import pandas as pd



# ---------------------------------------------------------------------------
# Context builder (RAG)
# ---------------------------------------------------------------------------

def _build_context(selected: pd.DataFrame, all_players: pd.DataFrame | None = None) -> str:
    rows = []
    for _, p in selected.iterrows():
        cap_tag = " [CAPTAIN]" if p.get("is_captain") else ""
        vc_tag  = " [VICE-CAPTAIN]" if p.get("is_vice_captain") else ""
        rows.append(
            f"• {p['player_name']} ({p['position']}) — "
            f"Cost: £{p['cost']:.1f}M | "
            f"Predicted pts: {p['predicted_points']:.1f} | "
            f"Total pts: {p['total_points']} | "
            f"Goals: {p['goals_scored']} Assists: {p['assists']} | "
            f"ICT: {p['ict_index']:.1f} | "
            f"Selected by: {p['selected_by_percent']:.1f}%"
            f"{cap_tag}{vc_tag}"
        )

    context = "=== SELECTED BEST XI ===\n" + "\n".join(rows)

    if all_players is not None:
        top5 = all_players.nlargest(5, "predicted_points")[
            ["player_name", "position", "cost", "predicted_points", "selected_by_percent"]
        ]
        context += "\n\n=== TOP 5 PLAYERS (ALL) ===\n"
        for _, p in top5.iterrows():
            context += (
                f"• {p['player_name']} ({p['position']}) "
                f"£{p['cost']:.1f}M → {p['predicted_points']:.1f} pts "
                f"({p['selected_by_percent']:.1f}% owned)\n"
            )

    return context


# ---------------------------------------------------------------------------
# Core API call
# ---------------------------------------------------------------------------

def _call_llm(system_prompt: str, user_message: str) -> str:
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }
    try:
        resp = requests.post(
            ANTHROPIC_API_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]
    except Exception as e:
        return f"⚠️ AI response unavailable: {str(e)}"


# ---------------------------------------------------------------------------
# Agent capabilities
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Fantasy Premier League (FPL) analyst.
You have access to AI-generated player data including predicted points,
ICT index, goals, assists, and ownership. Give concise, confident,
data-driven answers. Use bullet points where helpful. Keep responses
under 200 words unless asked for more detail."""


def explain_selection(player_name: str, selected: pd.DataFrame, all_players: pd.DataFrame) -> str:
    """Explain why a specific player was selected."""
    context = _build_context(selected, all_players)
    return _call_llm(
        SYSTEM_PROMPT,
        f"{context}\n\nExplain why {player_name} was selected for the Best XI. "
        "Reference their stats and predicted points specifically.",
    )


def recommend_captain(selected: pd.DataFrame, all_players: pd.DataFrame) -> str:
    """Justify the captain and vice-captain choices."""
    context = _build_context(selected, all_players)
    return _call_llm(
        SYSTEM_PROMPT,
        f"{context}\n\nJustify the captain and vice-captain selections. "
        "Consider predicted points, form, and fixture difficulty.",
    )


def suggest_transfers(selected: pd.DataFrame, all_players: pd.DataFrame, budget_remaining: float = 0.0) -> str:
    """Suggest one or two transfer upgrades."""
    context = _build_context(selected, all_players)
    return _call_llm(
        SYSTEM_PROMPT,
        f"{context}\n\nBudget remaining: £{budget_remaining:.1f}M\n\n"
        "Suggest 1–2 transfer improvements. Name specific players to bring in "
        "and who to drop, with a brief reason for each.",
    )


def find_best_differential(all_players: pd.DataFrame) -> str:
    """Identify the best differential pick."""
    diffs = all_players[all_players["selected_by_percent"] < 10].nlargest(5, "predicted_points")
    context = "=== LOW-OWNERSHIP CANDIDATES ===\n"
    for _, p in diffs.iterrows():
        context += (
            f"• {p['player_name']} ({p['position']}) "
            f"£{p['cost']:.1f}M | {p['predicted_points']:.1f} pts | "
            f"{p['selected_by_percent']:.1f}% owned\n"
        )
    return _call_llm(
        SYSTEM_PROMPT,
        f"{context}\n\nWho is the best differential pick from the list above "
        "and why? Consider the risk/reward trade-off.",
    )


def answer(question: str, selected: pd.DataFrame, all_players: pd.DataFrame) -> str:
    """Generic FPL question answering."""
    context = _build_context(selected, all_players)
    return _call_llm(
        SYSTEM_PROMPT,
        f"{context}\n\nUser question: {question}",
    )
