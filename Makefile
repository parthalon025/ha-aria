.PHONY: lint format test

all: lint

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

test:
	.venv/bin/python -m pytest --timeout=120 -x -q

