# Revision DEV experiments

Run from this checkout with its `src` on the module path (the shared virtualenv's editable install may point at the historical checkout):

```bash
PYTHONPATH=src <checkout> -m experiments.revision \
  --experiment e3c --seed 11 --root "$PWD" --output results/revision/dev/e3c-seed11
```

Use `--experiment e5c` for the fixed-candidate, one-round budget comparison. Both runs are implementation checks labelled DEV, with four candidates, a maximum of two paired HF queries for PIVOT-VOI/random HF, and eight fantasies/32 posterior samples. No powered scientific claim follows from these runs.

E3C runs two rounds per method and carries the selected immutable Policy into the next candidate generator. The zero option preserves the incumbent. PIVOT-VOI also carries its actual conditioned posterior. Calibration seeds and evaluation seeds are domain-separated. E5C gives every method exactly the same observable candidate rows and calibration prior.

The runner receives only allowlisted proxy features and estimated query costs. Online HF evaluates both policies in actor mode; offline oracle outcomes are evaluated after the decision and joined outside the decision runner. PIVOT-VOI records actual scoring, stopping, query, observation, conditioning and selection events. Random HF and all-HF are named literally; neither is presented as LUCB.

Every actual world call records its returned environment steps and simulator calls plus measured wall time. Calibration, proxy, online HF and offline oracle accounting are separate. Query cost units are environment steps; monetary compute cost is unknown. A durable `evaluation_journal/<attempt>/` writes a fsynced intent before entering the simulator and an atomic result or failure record after it returns. A killed process leaves an unresolved intent, whose unmeasured work remains explicitly unknown.

`raw_results.json`, `audit_trace.json`, `cost_ledger.json`, `config.json`, `dependencies.json`, `run_identity.json` and `DEV_SUMMARY.md` are SHA-256 covered by `manifest.json`. The dependency identity records installed distribution versions and Python version, and the code hash covers all shared `src/pivot` Python sources plus revision/V9 common sources. Use `--resume` only with the same identity; changed code, configuration or dependencies require a new output directory. Damaged or partial derived files with the same run identity are preserved in `recovery/<id>/` and regenerated from a checksum-valid raw checkpoint without rerunning the seed. If the raw checkpoint itself is corrupt, its original bytes and derived files are preserved before rerunning; differing identities are rejected without moving or replacing artifacts. Files are published through fsynced temporary files and atomic hard links, and the manifest is published last. The checkpoint unit is the complete experiment/seed, not an individual round.

Use `--methods pivot_voi pivot_voi_batch pivot_voi_no_update` for DEV controls on the same E5C candidate pool and acquisition RNG seed. Batch scores/acquires once, collects all chosen observations, then conditions without further acquisition; no-update freezes only real posterior coefficient updates while allowing hypothetical VOI fantasy updates and using observed candidate outcomes for selection. Sequential stopping uses analytic conservative `selection_probability_lower` and `acquisition_upper`, rather than the finite Monte Carlo point estimates.

`--environment controlled|congestion_resource|openspiel|physical|mpe2` selects a lazy adapter factory. Optional dependencies must be installed in the executing environment. Actual costs always come from returned rollout counters and adapter metadata. Acquisition uses an adapter's declared paired environment-step estimate/upper bound where available, otherwise the calibration mean paired actor steps; no adapter horizon layout is assumed. The generic two-parameter candidate generator remains a DEV probe, not an environment-specific trained improvement procedure.

World options are explicit JSON: `--environment mpe2 --world-config-json '{"scenario":"simple_spread","horizon":25}'` or `'{"scenario":"simple_adversary","horizon":25}'`. Physical options include `horizon`, `decision_repeat`, and `traffic_density`; controlled options accept `response_strength`. OpenSpiel uses its fixed Kuhn game and accepts an empty object.

Formal baselines remain unfinished. The historical V9 `paired_lucb` and `global_voi` implementations are heuristics; this DEV runner does not relabel them as statistically valid LUCB or a fair formal global-VOI baseline. Calibrated uncertainty validation, environment-specific improvement operators, larger candidate/budget sweeps, and prespecified powered analysis remain separate work.

`evaluation_audit.json` inventories journal files with hashes and totals known work over all invocation attempts, including failed and unresolved evaluations. Successful-unit `cost_ledger.json` still describes that successful attempt; its `failed_queries=0`/`retried_queries=0` must not be used as an aggregate of previous attempts. Audit cost completeness is false for interrupted/failed calls or opaque unmeasured solver work. Resolving actual monetary cost and within-call work after a kill needs provider/worker telemetry; do not replace unknowns with zero. Full experiment-seed recovery can repeat earlier rollouts in an incomplete seed, while completed seed checkpoints are reused without recomputation.
