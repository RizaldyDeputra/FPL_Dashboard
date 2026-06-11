# FPL AI-Dashboard  

A production-grade Fantasy Premier League assistant with a **live data**, **ML predictions**, **squad optimization**, and an **AI advisor**.

---

## Architecture

```
fpl_optimizer/
├── app.py                        # Streamlit dashboard (entry point)
├── update_data.py                # CLI automation script
├── requirements.txt
│
├── data/
│   ├── fpl_api.py                # Live FPL API client
│   ├── loader.py                 # Smart loader (live → static fallback)
│   ├── players.csv               # Static fallback dataset
│   ├── raw/                      # Raw JSON from FPL API
│   ├── processed/players.csv     # Live processed dataset
│   ├── predictions/latest.csv    # Latest ML predictions
│   └── cache/metadata.json       # Gameweek + fetch metadata
│
├── models/
│   ├── predictor.py              # RF + Gradient Boosting models
│   ├── train.py                  # Training script
│   ├── predict.py                # Fast inference (uses cache)
│   ├── model_store.py            # Pickle cache layer
│   └── saved/predictor.pkl       # Cached trained model
│
├── optimizer/
│   └── team_selector.py          # Two-phase MILP squad builder
│
├── agent/
│   └── advisor.py                # RAG context builder
│
└── logs/
    ├── pipeline.log              # Rolling app log
    ├── pipeline_runs.jsonl       # Structured run history
    └── training.jsonl            # Model training log
```

---

## Live Data Pipeline

```
FPL API                    FPLAPIClient
bootstrap-static/  ──────▶ fetch_and_save()
fixtures/                │   └─ data/raw/bootstrap_static.json
                         │   └─ data/processed/players.csv
                         │   └─ data/cache/metadata.json
                         ▼
                    loader.py
                    load_and_prepare()
                         │
                         ▼
                    predictor.py  ──── model_store.py (cache)
                    run_ml_pipeline()
                         │
                         ▼
                    team_selector.py
                    optimise_squad()  ◀─── MILP (scipy)
                         │
                         ▼
                    app.py (Streamlit)
```

---


## ML Pipeline

| Model | MAE | RMSE | Notes |
|---|---|---|---|
| Gradient Boosting | ~0.12 | ~0.18 | ✅ Best — sequential error correction |
| Random Forest | ~0.27 | ~0.40 | Baseline — 200 trees |

**Model cache**: trained model saved to `models/saved/predictor.pkl`. Retrains automatically when:
- Data hash changes (new API fetch)
- Model is > 24 hours old

---

## Dashboard Features

| Tab | Description |
|---|---|
| 🏟 My Squad | Pitch view (XI + bench), player list, budget bar, injury/availability status, transfer momentum arrows |
| 💡 Insights | Auto-generated insights, captain picks, best value, risky high-ceiling |
| 📈 Top Players | Filterable ranked list with form bars + squad tags |
| 🎯 Differentials | Low-ownership high-value picks with pts/£M scoring |
| 🤖 AI Advisor | RAG-powered chat interface + 8 quick prompts |
| ⚙️ Pipeline | Live status, force-refresh buttons, automation commands, recent run log |

**Sidebar**:
- Live status badge (🟢 Live / 🟡 Stale / ⚪ No data)
- 🔄 Refresh Live Data button
- Gameweek + deadline display
- Max per club dropdown
- Min minutes slider
- Pipeline status expander

---

## Key Technical Details

- **Staleness threshold**: 6 hours (configurable via `DATA_STALE_HOURS`)
- **Model TTL**: 24 hours (configurable in `model_store.py`)
- **Offline fallback**: cached raw JSON used when API is unreachable
- **Cache invalidation**: `st.cache_data` keyed on file modification time
- **Squad composition**: 2 GK, 5 DEF, 5 MID, 3 FWD = 15 players
- **Budget**: £100M fixed
- **Formation**: auto-selected (tries all 7 valid FPL formations)
