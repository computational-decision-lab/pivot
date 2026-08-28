.PHONY: reproduce-paper test lint typecheck v9-validate v9-analyze v9-figures v9-tables v9-audit v10-finalize v15-reports v15-figures v15-figure-review v15-finalize v15-release v15-master-loop

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

v10-finalize:
	.venv/bin/python -m experiments.v10.finalize --root .

v15-reports:
	.venv/bin/python -m experiments.v15 reports --root .

v15-figures:
	.venv/bin/python -m experiments.v15 figures --root .

v15-figure-review:
	.venv/bin/python -m experiments.v15 approve-figures --root .

v15-finalize:
	(cd paper/iclr2027 && ./build.sh)
	.venv/bin/python -m experiments.v15 finalize --root .

v15-release: v15-finalize
	.venv/bin/python scripts/build_release_assets.py --root . --public-release --force

v15-master-loop:
	.venv/bin/python -m experiments.v15 master-loop --root .
