.PHONY: lint lint-py lint-spa lint-audit format test

all: lint

lint: lint-py lint-spa lint-audit

lint-py:
	ruff check .

lint-spa:
	cd aria/dashboard/spa && npm run lint

lint-audit:
	pip-audit --progress-spinner off -q 2>/dev/null || true

format:
	ruff format .
	ruff check --fix .
	cd aria/dashboard/spa && npx prettier --write src/

test:
	.venv/bin/python -m pytest --timeout=120 -x -q

