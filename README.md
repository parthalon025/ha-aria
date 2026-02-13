<div align="center">

# ARIA

### Adaptive Residence Intelligence Architecture

**Your home generates 22,000+ events every day.<br/>ARIA learns what they mean.**

[![CI](https://github.com/parthalon025/ha-aria/actions/workflows/ci.yml/badge.svg)](https://github.com/parthalon025/ha-aria/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-683_passing-brightgreen)](https://github.com/parthalon025/ha-aria/actions)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Quick Start](#quick-start) · [Features](#features) · [Architecture](#architecture) · [Dashboard](#dashboard) · [CLI](#cli-reference) · [Contributing](#contributing)

</div>

---

## The Problem

Home Assistant is extraordinary at collecting data — every light switch, door sensor, thermostat adjustment, and motion event gets logged. But **collection isn't intelligence.**

Most HA users end up with:

- Hundreds of hand-written automations, one trigger at a time
- No idea which devices correlate, which patterns repeat, or what's abnormal
- Zero predictions — the system reacts but never anticipates

You're managing a smart home with a clipboard when you need a co-pilot.

## What ARIA Does

ARIA watches, learns, and predicts — **entirely on your local hardware.** No cloud. No subscriptions. No data leaving your network.

| Without ARIA | With ARIA |
|:---|:---|
| You write every automation by hand | ARIA detects patterns and generates ready-to-use YAML |
| Anomalies go unnoticed until something breaks | Statistical baselines flag unusual activity in real time |
| "Is someone home?" needs a dedicated sensor | Bayesian occupancy fuses motion, doors, lights, and media |
| Power consumption is a monthly surprise | Per-outlet profiling with cycle detection and trends |
| Predictions don't exist | 6 ML models forecast what your home will do next |

## Features

### Learn

- **15 entity collectors** — power, climate, occupancy, locks, motion, EV charging, media, and more
- **Statistical baselines** — hourly patterns with confidence ranges, built from your data
- **Entity correlation** — discover which devices activate together and when

### Predict

- **6 ML models** — GradientBoosting, RandomForest, IsolationForest, Prophet, Markov chains, Bayesian occupancy
- **Shadow mode** — predictions scored against reality before any automation runs
- **Concept drift detection** — Page-Hinkley test catches when your home's behavior shifts

### Act

- **Automation suggestions** — LLM-generated Home Assistant YAML from detected patterns
- **Meta-learning** — local Ollama tunes feature engineering based on prediction accuracy
- **Daily intelligence reports** — LLM-summarized insights delivered to Telegram

### See

- **11-page interactive dashboard** — Preact SPA with live WebSocket updates
- **Real-time activity monitor** — 15-minute windowed analysis of every state change
- **Predictions vs actuals** — track how well ARIA's models perform over time

## Quick Start

### 1. Install

```bash
pip install ha-aria
```

Or from source:

```bash
git clone https://github.com/parthalon025/ha-aria.git
cd ha-aria
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,llm,ml-extra,prophet]"
```

### 2. Connect

```bash
export HA_URL="http://your-ha-instance:8123"
export HA_TOKEN="your-long-lived-access-token"
```

### 3. Run

```bash
# Collect your first snapshot
aria snapshot

# Run the full pipeline (snapshot → predict → report)
aria full

# Start the real-time dashboard
aria serve
# → http://localhost:8001
```

ARIA starts learning immediately. Baselines form within 24 hours. ML predictions improve daily. Shadow mode validates accuracy before suggesting any automations.

## Architecture

ARIA has two halves that work together:

```mermaid
flowchart TB
    HA[("Home Assistant\nREST API + WebSocket")]

    subgraph Engine ["Engine — batch jobs"]
        direction TB
        E1["Collect snapshots"]
        E2["Build baselines & features"]
        E3["Train 6 ML models"]
        E4["Generate predictions"]
        E5["LLM reports & suggestions"]
        E1 --> E2 --> E3 --> E4 --> E5
    end

    subgraph Hub ["Hub — real-time service"]
        direction TB
        H1["Entity discovery"]
        H2["Activity monitor\n(WebSocket listener)"]
        H3["Shadow engine\n(predict → compare → score)"]
        H4["Pattern detection\n(Markov sequences)"]
        H5["Data quality\n(entity curation)"]
    end

    subgraph Dash ["Dashboard — Preact SPA"]
        D1["11 interactive pages"]
        D2["Live WebSocket updates"]
    end

    HA -- "scheduled\nvia cron" --> Engine
    HA -- "persistent\nWebSocket" --> Hub
    Engine -- "JSON files" --> Dash
    Hub -- "SQLite cache\n+ WebSocket" --> Dash
```

- **Engine** runs as scheduled batch jobs — collects data, trains models, generates predictions, produces LLM reports
- **Hub** runs as a persistent service — monitors real-time activity via WebSocket, validates predictions in shadow mode, detects patterns
- **Dashboard** presents everything in a live Preact SPA with WebSocket-pushed updates

### How It Learns

ARIA's learning pipeline advances automatically through four stages:

```mermaid
flowchart LR
    S1["Collecting\n(day 1)"]
    S2["Baselines\n(day 2–3)"]
    S3["ML Training\n(day 3–7)"]
    S4["ML Active\n(day 7+)"]

    S1 -- "snapshots\naccumulate" --> S2
    S2 -- "hourly norms\nestablished" --> S3
    S3 -- "models trained\non features" --> S4
    S4 -- "predictions\nvalidated" --> S4
```

**Shadow mode** is the safety net: ARIA makes predictions silently, compares them to what actually happens, and only suggests automations when accuracy thresholds are met. You stay in control.

## Dashboard

The dashboard ships with 11 pages covering the full intelligence pipeline:

| Page | What You See |
|------|-------------|
| **Home** | 3-lane pipeline flowchart with live status chips and journey progress |
| **Intelligence** | Baselines, predictions vs actuals, trends, correlations, LLM insights |
| **Discovery** | Every entity, device, and area HA knows about |
| **Activity** | Real-time event stream with 15-minute windowed analytics |
| **Patterns** | Recurring event sequences detected from your logbook |
| **Predictions** | ML model outputs with confidence scores |
| **Shadow Mode** | Prediction accuracy, high-confidence disagreements, pipeline stage |
| **Automations** | LLM-suggested HA automation YAML from detected patterns |
| **Data Curation** | Entity-level include/exclude tiering for noise control |
| **Settings** | Tunable parameters — retraining schedules, thresholds, model config |

> **Screenshots coming soon** — to preview the dashboard locally, run `aria serve` and open `http://localhost:8001`.

## CLI Reference

| Command | Description |
|---------|-------------|
| `aria serve` | Start real-time hub + dashboard |
| `aria full` | Full daily pipeline (snapshot → predict → report) |
| `aria snapshot` | Collect current HA state |
| `aria predict` | Generate predictions from latest snapshot |
| `aria score` | Score yesterday's predictions against actuals |
| `aria retrain` | Retrain all ML models |
| `aria meta-learn` | LLM meta-learning to tune feature config |
| `aria check-drift` | Concept drift detection (Page-Hinkley test) |
| `aria correlations` | Entity co-occurrence analysis |
| `aria sequences train` | Train Markov chain model from logbook |
| `aria sequences detect` | Detect anomalous event sequences |
| `aria suggest-automations` | Generate HA automation YAML via LLM |
| `aria prophet` | Train Prophet seasonal forecasters |
| `aria occupancy` | Bayesian occupancy estimation |
| `aria power-profiles` | Per-outlet power consumption analysis |
| `aria sync-logs` | Sync HA logbook to local storage |

## Requirements

- **Python** >= 3.12
- **Home Assistant** instance with a [long-lived access token](https://developers.home-assistant.io/docs/auth_api/#long-lived-access-token)
- **Optional:** [Ollama](https://ollama.ai/) for LLM features (daily reports, meta-learning, automation suggestions)
- **Optional:** LightGBM, Prophet for extended ML capabilities

## Project

| | |
|:---|:---|
| **Tests** | 683 (677 passing, CI-enforced) |
| **Code** | 14,451 lines across 63 Python files |
| **Dashboard** | 37 JSX components across 11 pages |
| **Hub modules** | 8 registered (discovery, ML, patterns, shadow, orchestrator, data quality, intelligence, activity) |
| **CI** | Lint → Test (Python 3.12 + 3.13) → Dashboard build → Codecov |

### Built With

[scikit-learn](https://scikit-learn.org/) · [FastAPI](https://fastapi.tiangolo.com/) · [Preact](https://preactjs.com/) · [Tailwind CSS](https://tailwindcss.com/) · [Prophet](https://facebook.github.io/prophet/) · [Ollama](https://ollama.ai/) · [LightGBM](https://lightgbm.readthedocs.io/)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

```bash
git clone https://github.com/parthalon025/ha-aria.git
cd ha-aria
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,llm,ml-extra,prophet]"

# Run tests
pytest tests/ -v

# Lint
ruff check . && ruff format --check .
```

## License

[MIT](LICENSE) — Justin McFarland, 2026
