# Contributing to ARIA

Thank you for your interest in contributing to ARIA! This guide will help you get set up and familiar with the project.

## Development Setup

### Prerequisites

- Python 3.12 or later
- Node.js 20+ (for dashboard builds only)
- A Home Assistant instance (for integration testing — not required for unit tests)

### Install

```bash
git clone https://github.com/parthalon025/ha-aria.git
cd ha-aria
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ml-extra]"
```

### Run Tests

```bash
# Full suite
pytest tests/ -v

# Engine tests only
pytest tests/engine/ -v

# Hub tests only
pytest tests/hub/ -v

# Integration tests
pytest tests/integration/ -v
```

### Lint

```bash
ruff check aria/ tests/
ruff format aria/ tests/
```

### Build Dashboard

```bash
cd aria/dashboard/spa
npx esbuild src/index.jsx --bundle --outfile=dist/bundle.js \
  --jsx-factory=h --jsx-fragment=Fragment \
  --inject:src/preact-shim.js --loader:.jsx=jsx --minify
```

## Making Changes

1. **Create a branch** from `main`: `git checkout -b feat/your-feature`
2. **Write tests first** — we follow TDD where practical
3. **Implement your change**
4. **Run the full test suite** — all tests must pass
5. **Run the linter** — `ruff check` and `ruff format` must pass
6. **Open a PR** against `main`

## Code Style

- **Python**: Enforced by [ruff](https://docs.astral.sh/ruff/) (PEP 8 + import sorting)
- **Line length**: 120 characters
- **Type hints**: Encouraged for public APIs
- **Docstrings**: Required for public functions and classes

## Project Layout

```
aria/
├── engine/       # Batch ML pipeline (collectors, models, analysis)
├── hub/          # Real-time service (FastAPI, cache, WebSocket)
├── modules/      # Hub runtime modules (discovery, shadow, activity)
├── dashboard/    # Preact SPA (Tailwind CSS, esbuild)
├── cli.py        # Unified CLI entry point
└── __init__.py   # Package version

tests/
├── engine/       # Engine unit tests
├── hub/          # Hub unit tests
└── integration/  # Cross-package integration tests
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for system design details.

## Reporting Issues

- **Bugs**: Use the [bug report template](https://github.com/parthalon025/ha-aria/issues/new?template=bug_report.md)
- **Features**: Use the [feature request template](https://github.com/parthalon025/ha-aria/issues/new?template=feature_request.md)

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be respectful and constructive.
