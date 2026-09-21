# HighwayEnv reproduction

This artifact retains the original calibrated cohort and one subsequent DEV
replication. The replication has 20 new calibration seeds and 60 new test seeds;
B=2 is primary and B=4 is secondary. It does not add a benchmark family or a
budget sweep. The frozen analysis plan is in
`source/docs/experiments/highwayenv-calibrated-replication-preregistration-20260921.md`.

## Recompute the reported numbers

From the artifact/repository root, use Python with NumPy and Matplotlib:

```bash
python scripts/build_highway_evidence.py --root .
```

This verifies both evidence manifests, all registered seed/method/budget cells,
query counts, and truth-audit consistency before recomputing every contrast.
It regenerates the table, paired seed CSV, macros, figure, and audit under
`paper/`. All 60 seed pairs are retained. The contrast is Uniform HF ISR minus
PIVOT-KG ISR; positive favors PIVOT-KG. Percentile bootstrap intervals use
4,000 resamples, RNG seed `20260922 + B`, and the registered seed order.
These are individual 95% intervals, not a multiplicity-adjusted joint claim.
They condition on each cohort's fitted calibration model and do not include
uncertainty from resampling the calibration cohort.

## Rerun the frozen simulator code

Use Python 3.12.3 for the recorded cloud environment; the package versions are
pinned in `requirements-lock.txt`. Git is needed to record the exported source
snapshot truthfully. No cloud account, credentials, or historical Git checkout
is required.

```bash
python3.12 -m venv .venv-highway
.venv-highway/bin/python -m pip install -r reproduction/highway/requirements-lock.txt
.venv-highway/bin/python reproduction/highway/run.py --check-only
.venv-highway/bin/python reproduction/highway/run.py --output results/highway-replication
# Optional independent reproduction of the earlier, separate cohort:
.venv-highway/bin/python reproduction/highway/run.py --cohort original --output results/highway-original
```

The launcher first verifies every frozen file and rejects modifications or
unlisted files. It creates a separate working directory and a new Git snapshot
commit, then invokes the unchanged historical runner. The historical source
commits are recorded in `source-manifest.json`; the new run has its own Git
identity. The four algorithm/configuration file hashes and source hash can be
compared to `evidence/highway/*/identity.json`. Only a documentation path was
redacted in the exported source; algorithm files are byte-identical.
`--check-only` checks hashes and imports without evaluating any seed.
Each full run takes roughly one to two CPU hours; it writes checkpoints,
journals, identities, seed registry, query ledger, truth audit, and cost ledger.
Use a new output directory. Do not change seeds or settings to chase a result.

## Budget and evidence boundaries

B counts candidate paired HF queries available to a selection decision. A
paired query compares a candidate with the incumbent in the actor world under
the same seed. Shared incumbent evaluation is reused. Calibration, proxy
rollouts, and post-decision truth audits are outside this logical budget and
are reported separately in the physical evaluation ledger. Physical calls,
environment steps, wall time, and money are different quantities; B is not a
cloud spending cap. The All-HF Oracle uses all five candidates only as an
audit/reference arm. The historical `Proxy Only` label in raw records denotes
a fixed-order budgeted HF arm in this runner; it must not be read as a zero-HF
baseline. The paper contrast uses Uniform HF and calibrated PIVOT-KG.
Uniform HF uses proxy estimates for unqueried candidates; calibrated PIVOT-KG
uses its calibrated posterior. The comparison measures their complete
selection rules under matched query budgets, including calibration and final
selection. It does not isolate acquisition alone.

The original cohort informed this replication design. A technical smoke run
also exposed the first two replication calibration roots and first two test
roots before the full run. They remain in the registered cohorts: no roots
were excluded, and no settings were changed in response. Therefore this is an
external DEV replication, not a wholly blinded confirmatory test. B=4 cannot
replace the primary B=2 conclusion. After this replication, experiment
expansion stops regardless of the observed sign.
