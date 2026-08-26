.PHONY: reproduce-paper test lint typecheck v9-validate v9-analyze v9-figures v9-tables v9-audit

reproduce-paper:
	.venv/bin/python scripts/reproduce_paper.py

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check src scripts experiments tests

typecheck:
	.venv/bin/mypy src scripts experiments

v9-validate:
	.venv/bin/python -m experiments.v9.validate --root . --strict
	.venv/bin/python -m experiments.v9.build_manifest --root .

v9-analyze:
	.venv/bin/python -m experiments.v9.analyze --root .

v9-figures:
	.venv/bin/python scripts/build_v9_figures.py --root .

v9-tables:
	.venv/bin/python scripts/build_v9_tables.py --root .
	.venv/bin/python scripts/build_v9_paper_snippets.py --root .
	.venv/bin/python scripts/sync_v9_paper_assets.py --root .

v9-audit:
	.venv/bin/python scripts/audit_v9_statistics.py --root .
	.venv/bin/python scripts/audit_v9_figures.py --root .
	.venv/bin/python scripts/audit_v9_claims.py --root .
