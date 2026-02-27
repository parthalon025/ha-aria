.PHONY: lint format test

all: lint

lint:
	ruff check .
	mypy aria/ --ignore-missing-imports

format:
	ruff format .
	ruff check --fix .

test:
	.venv/bin/python -m pytest --timeout=120 -x -q

