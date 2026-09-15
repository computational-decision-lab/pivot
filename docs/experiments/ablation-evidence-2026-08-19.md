# Twelve-Ablation Evidence

This record covers the frozen controlled ablation suite at commit
`d519af0`. It closes the implementation gap in the master specification, but
it is not external-validity evidence and does not promote any paper gate.

## Contract and reproduction

- Suite: `controlled-ablation-v1`.
- Configuration: `configs/sweeps/ablations.yaml`.
- Registry: `configs/registered/ablations.yaml`.
- Independent seed sets: three disjoint six-seed runs (`1101-1106`,
  `1201-1206`, `1301-1306`).
- Train/test split and high-fidelity budgets are recorded in every run's
  `provenance.json`; no run selected a subset after inspecting outcomes.
- Every output retains `ablation_rows.jsonl`, `failed_runs.jsonl`, summary, and
  run manifest. All three failure ledgers are empty.
- `paired_vs_unpaired` uses a non-saturated variance fixture
  (`noise_scale=0.5`, `horizon=4`, `reward_bound=100`) so the standard-error
  contrast measures common random numbers rather than reward clipping.
- Finance rows use virtual fills only; they are a fixture comparison, not live
  trading or causal market evidence.

Clean-room root:
`/tmp/pivot-ablations-cleanroom-d519af0`

Commands:

```bash
.venv/bin/python scripts/run_registered.py \
  --registry configs/registered/ablations.yaml \
  --output /tmp/pivot-ablations-cleanroom-d519af0

.venv/bin/python scripts/aggregate_registered.py --experiment ablations \
  --inputs /tmp/pivot-ablations-cleanroom-d519af0/ablation-r01 \
           /tmp/pivot-ablations-cleanroom-d519af0/ablation-r02 \
           /tmp/pivot-ablations-cleanroom-d519af0/ablation-r03 \
  --output /tmp/pivot-ablations-cleanroom-d519af0/ablation-aggregate.json
```

The registered run summary was `n_runs=3`, `n_ok=3`, `n_failed=0`; the
aggregate reported `valid_run_count=3` and `ablation_count=12`. The clean-room
targeted test set reported `7 passed`.

## Aggregated observations

Values below are the bootstrap estimates over the three independent registered
runs. They are descriptive controlled-fixture results.

| Ablation | Main observation |
| --- | --- |
| Paired vs unpaired | Standard error `0.04327` vs `0.06762`; paired is lower. The delta estimates differ because the independent contexts are intentionally different; this is not a comparison of two estimators on identical random draws. |
| Transition vs global value | Transition IDE `6.8691`; global-value IDE `7.1199`. ISC is `0.2708` for both in this fixture, so no broad ranking claim is made. |
| Footprint vs no footprint | IDE `2.06675` vs `2.06686`, ISC `0.8542` for both, IRR `0.2` for both. This is a null on the current controlled grid, not evidence that footprint features are unnecessary. |
| Active vs random HF | PIVOT CTI `-0.3303`, ISR `0.2439`; random CTI `-4.9602`, ISR `4.8738` at the matched one-query budget. |
| PIVOT vs Top Proxy | PIVOT ISR `0.2439`; Top Proxy ISR `3.3746` at the matched one-query budget. |
| Small vs large updates | IRR `0.5` vs `0.75` for the predeclared footprint split. |
| Weak vs strong response | IRR `0.4583` vs `1.0` for the predeclared response split. |
| F1 vs F2 | F2 minus F1 delta `-0.000384` at participation `0.05` in the virtual finance fixture. |
| Fixed vs adaptive competitors | Fixed SIRR `0.0`; adaptive SIRR `1.0`, with adaptive delta `-0.04967`. |
| Single vs multiple response models | Multiple-response IDE `2.6906`, ISC `0.9722`; single-response IDE `4.7509`, ISC `0.75` on the held-out response level. Support is retained in raw rows; some sign metrics have fewer valid groups. |
| Candidate count | ISR `0.2368` at 2 candidates vs `0.2439` at 4 candidates. |
| HF budget | PIVOT ISR is `1.0210`, `0.2439`, `0.2439`, `0.4033`, `0.0` for budgets `0..4`; the non-monotonic fixture curve is retained. |

The values are not a claim that PIVOT dominates every environment. In
particular, the footprint null and non-monotonic budget curve require broader
controlled designs before being used in the main paper.

## Integrity hashes

```text
code commit                         d519af0
configs/sweeps/ablations.yaml       92fe86bf0a35184f4057b9a51f40eafbe89dd4c290c70ee2699175d047bc4360
configs/registered/ablations.yaml   3f145b0c75e2d4b581b664c25815b2e6a02f927d18587346ea6594b74c5de8aa
ablation-aggregate.json             e081f87fd71a187317df42a056afeaafc95dd71d2a2906d42298414f97db4366
ablation-r01/ablation_summary.json  b9a81460db70180f870f9e42c31bb858e83ce3b7baedc4832ecd4c524132150a
ablation-r02/ablation_summary.json  33d2d6a57c8100573535bb79aaa1462368181fe7dfab16cc35658873334a0327
ablation-r03/ablation_summary.json  4d14622f3b376ec12dc44537a55765f09fe94b30b307f00d2b711e7798b73e8b
ablation-r01/ablation_rows.jsonl    b4245b2b5ccceaf5dc5f047b904b692843a5b4a071276de3a37c2fcfd0ca6c60
ablation-r02/ablation_rows.jsonl    cce9896e200a9fba5c16b79b5310fc1a87be493a87cafa57dbee0182cf355546
ablation-r03/ablation_rows.jsonl    437135be63421e66de21411202f9683642104b59579706f42a808b4a88eec96f
```

## Remaining scientific blockers

The suite closes the implementation and evidence-logging gap, not the
submission gap. The remaining blockers are a confirmatory update-generation
rule, causal interactive response data, external strategic validation, and a
broader design for the footprint and budget nulls. LLM/EvoQuant/M3 adapters
remain intentionally deferred.
