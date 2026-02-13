# ha-intelligence

Home Assistant intelligence engine — collects HA logbook data, builds statistical baselines, trains ML models (scikit-learn), and generates predictions, anomaly detection, and natural-language reports via local LLM (deepseek-r1:8b).

## Components

| Script | Purpose |
|--------|---------|
| `bin/ha-intelligence` | Main engine (~2,200 lines): snapshots, analysis, prediction, ML training, meta-learning, reporting |
| `bin/ha-log-sync` | Syncs HA logbook API to local JSON files (runs every 15min via cron) |

## Architecture

```
HA REST API → ha-log-sync → ~/ha-logs/raw/
                                  ↓
                          ha-intelligence
                          ├── --snapshot / --snapshot-intraday (collect)
                          ├── --analyze (baselines + correlations)
                          ├── --predict (statistical + ML predictions)
                          ├── --score (accuracy tracking)
                          ├── --retrain (sklearn model refresh)
                          ├── --meta-learn (LLM reflection on accuracy)
                          └── --report / --brief / --full (LLM narratives)
```

## Data

All intelligence data lives in `~/ha-logs/intelligence/` (daily snapshots, intraday snapshots, trained models, meta-learning outputs, baselines, predictions, accuracy scores, correlations).

## Schedule (cron)

- Every 15min: `ha-log-sync`
- Every 4h: `ha-intelligence --snapshot-intraday`
- 11:00 PM daily: `ha-intelligence --snapshot`
- 11:30 PM daily: `ha-intelligence --full` (analyze + predict + score + report)
- Monday 1:30 AM: `ha-intelligence --retrain && ha-intelligence --meta-learn`

## ML Stack

- scikit-learn 1.8.0 (GradientBoosting, RandomForest, IsolationForest)
- numpy 2.4.2
- Ramp-up: Days 1-7 stats only → Day 14+ sklearn models → Day 60+ ML weighted higher → Day 90+ ML dominant

## Tests

```bash
python -m pytest tests/
```

## Design Docs

- `docs/ha-intelligence-engine.md` — v1 design (statistical baselines)
- `docs/ha-intelligence-ml-design.md` — v2 design (ML pipeline)
- `docs/home-assistant-claude-code-design.md` — initial HA integration design
