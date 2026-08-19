# PIVOT

PIVOT studies **Improvement Fidelity**: whether a policy update that looks beneficial in a cheap or fixed proxy world remains beneficial after deployment changes the environment and, later, after other agents adapt.

```text
PIVOT = Paired Interventional Verification of Optimization Transitions
```

Working paper title: *When Better Gets Worse: Improvement Fidelity for Self-Improving Agents in Adaptive Worlds*.

## Current Status

- Research specification: frozen and documented.
- Implementation: P0-P9 research harnesses are present; P10 LLM/EvoQuant/M3 adapters remain deferred.
- Scientific gates A-F: none formally passed. The calibrated E5 smoke is
  promising at small budgets, but Gate D still requires independent registered
  runs and paired intervals; no smoke result is a paper claim.
- Live trading or external execution: out of scope.

Start with:

- `/opt/projects/research/pivot/docs/pivot.md`
- `/opt/projects/research/pivot/docs/master-goal.md`
- `/opt/projects/research/pivot/docs/superpowers/plans/2026-08-19-pivot-master-implementation.md`
- `/opt/projects/research/pivot/docs/experiments/gates.md`

Run the controlled first milestone with:

```bash
python3 scripts/run_sweep.py --config configs/sweeps/p2.yaml --output results/raw/controlled-first
```

The command writes a Parquet/JSONL transition table, provenance, confidence
intervals, CSV source tables, and PNG diagnostics. Finance and strategic
commands are separate (`experiments/e6_finance_actor.py`, `e7`, `e8`, `e9`),
and all fills remain virtual. See `docs/implementation-status.md` for the
current gate-aware status and known limitations.
