# ARIA — Adaptive Residence Intelligence Architecture

[![CI](https://github.com/parthalon025/ha-aria/actions/workflows/ci.yml/badge.svg)](https://github.com/parthalon025/ha-aria/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://pypi.org/project/ha-aria/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> ML-powered intelligence for Home Assistant. Learns your home's patterns, predicts what's coming, and spots anomalies — all running locally.

## Features

- **15 entity collectors** — power, climate, occupancy, locks, motion, EV, media, and more
- **ML prediction engine** — GradientBoosting, RandomForest, IsolationForest, Prophet
- **Anomaly detection** — statistical baselines, Markov chain sequence analysis, concept drift
- **Real-time monitoring** — WebSocket activity tracking with 15-minute windowed analysis
- **Shadow mode** — predict-compare-score loop that validates ML accuracy before automating
- **LLM insights** — Ollama-powered daily reports, meta-learning, automation suggestions
- **Interactive dashboard** — Preact + Tailwind SPA with live WebSocket updates
- **Entity correlation** — discover which devices activate together and when
- **Bayesian occupancy** — multi-sensor room occupancy estimation
- **Power profiling** — per-outlet consumption analysis and cycle detection

## Quick Start

### Install

```bash
pip install ha-aria
```

Or install from source:

```bash
git clone https://github.com/parthalon025/ha-aria.git
cd ha-aria
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,llm,ml-extra,prophet]"
```

### Configure

Set your Home Assistant connection:

```bash
export HA_URL="http://your-ha-instance:8123"
export HA_TOKEN="your-long-lived-access-token"
```

### Run

```bash
# Collect a snapshot of your home's current state
aria snapshot

# Run the full daily pipeline (snapshot -> predict -> report)
aria full

# Start the real-time dashboard
aria serve
# Open http://localhost:8001 in your browser
```

## Architecture

```
                    Home Assistant
               (REST API + WebSocket)
              /                       \
     +---------+                +---------+
     |  Engine  |               |   Hub   |
     | (batch)  |               | (live)  |
     |          |               |         |
     | Collect  |               | Discover|
     | Analyze  |               | Monitor |
     | Train    |               | Shadow  |
     | Predict  |               | Pattern |
     | LLM      |               | ML      |
     +----+-----+               +----+----+
          |                          |
          +----------+---------------+
                     |
              +------+------+
              |  Dashboard  |
              | (Preact SPA)|
              +-------------+
```

- **Engine** runs as batch jobs via cron — collects data, trains models, generates predictions
- **Hub** runs as a service — monitors real-time activity, serves dashboard, validates predictions
- **Dashboard** shows live intelligence, predictions vs actuals, shadow mode accuracy, and more

## CLI Reference

| Command | Description |
|---------|-------------|
| `aria snapshot` | Collect current HA state |
| `aria predict` | Generate predictions from latest snapshot |
| `aria full` | Full daily pipeline (snapshot + predict + report) |
| `aria score` | Score yesterday's predictions |
| `aria retrain` | Retrain ML models |
| `aria serve` | Start real-time hub + dashboard |
| `aria correlations` | Entity co-occurrence analysis |
| `aria sequences train` | Train Markov chain model |
| `aria sequences detect` | Detect anomalous sequences |
| `aria suggest-automations` | LLM-generated automation YAML |
| `aria meta-learn` | LLM meta-learning |
| `aria check-drift` | Concept drift detection |
| `aria prophet` | Train Prophet seasonal forecasters |
| `aria occupancy` | Bayesian occupancy estimation |
| `aria power-profiles` | Power consumption profiling |

## Requirements

- Python >= 3.12
- Home Assistant instance with a long-lived access token
- Optional: [Ollama](https://ollama.ai/) for LLM features (reports, meta-learning, automation suggestions)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

[MIT](LICENSE) — Justin McFarland, 2026
