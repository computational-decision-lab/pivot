.PHONY: reproduce-paper test lint typecheck

reproduce-paper:
	.venv/bin/python scripts/reproduce_paper.py

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check src scripts experiments tests

typecheck:
	.venv/bin/mypy src scripts experiments
