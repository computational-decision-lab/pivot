# Reproducing the paper artifact

From the repository root, create the replay environment and install the code:

```bash
python3.10 -m venv /tmp/pivot-paper-replay
/tmp/pivot-paper-replay/bin/python -m pip install -r reproduction/environments/replay-requirements.txt
/tmp/pivot-paper-replay/bin/python -m pip install --no-deps -e .
/tmp/pivot-paper-replay/bin/python reproduction/run.py verify --output /tmp/pivot-paper-check
/tmp/pivot-paper-replay/bin/python reproduction/run.py analyze --experiment leduc_v4 --output /tmp/pivot-paper-analysis
/tmp/pivot-paper-replay/bin/python reproduction/run.py figures --output /tmp/pivot-paper-figures
```

Replace `reproduction/run.py` and the requirements path with absolute paths when the shell is outside the clone. `--python` selects the interpreter for child commands. The output argument is a directory and must be outside `evidence/paper`.

`verify` checks all 1,051 manifest hashes (1,049 original inputs plus two export/provenance records), then replays the saved-label checks in `reproduction/verify_saved_evidence.py`. `analyze` runs that same complete check and writes a selected JSON view of its recomputed aggregates. The `core` statistic pools 90 Kuhn, first Leduc, and Melting Pot roots; a named core cohort also gets its own 30-root tie-aware disagreement and tie counts from the saved summaries. For original Highway, the saved verifier checks file hashes but has no independent original-cohort aggregate calculation. `figures` delegates to `reproduction/figures/run.py` and regenerates plots from saved inputs. These commands do not call a simulator, retrain policies, establish registration timestamps, or certify author approval.

`smoke --experiment NAME` runs a bounded environment check where the corresponding simulator is installed. With no `--experiment`, it checks each listed cohort and stops at the first failure. The controlled check executes a small rollout in its frozen local world and reports `native_environment_step: false` because no external simulator is involved. On a Python replay environment without external simulators, native smoke fails and writes a failure JSON, rather than being counted as a pass.

`full --experiment NAME --output NEW_DIRECTORY` launches the frozen source and protocol for the named cohort. It can take substantial compute and is **not** a bounded smoke. The poker runs generate calibration and confirm panels then run their frozen analyzer. The Highway launchers manage their verified source snapshots; `highway` selects the replication cohort. MetaDrive runs calibration, freezes its model, then runs confirmation and formal scoring. `controlled` calls the five historical E2/E3/E4/E5/E7 confirmatory runners separately. Its five configs match the packaged copies by SHA-256; the shared profile file comes from this release clone's `configs/v9/profiles.yaml`. Full Melting Pot execution fails with an explicit specialist-model, original frozen confirmation-protocol, and crossbench-root gap. No full command was run as part of artifact QA.

Examples below assume the shell is at the release clone root. Every `--output` must name a new directory. OpenSpiel, MetaDrive, and Highway interpreters should be created from their corresponding current smoke locks or a separately recovered historical environment. The smoke locks only show that their bounded checks ran; full study runtime, hardware demand, and byte-level agreement with historical outputs remain unmeasured.

```bash
/tmp/pivot-repro-openspiel/bin/python reproduction/run.py full --experiment kuhn --output /path/to/new-kuhn
/tmp/pivot-repro-openspiel/bin/python reproduction/run.py full --experiment leduc --output /path/to/new-leduc
/tmp/pivot-repro-openspiel/bin/python reproduction/run.py full --experiment leduc_v4 --output /path/to/new-leduc-v4
/tmp/pivot-repro-highway/bin/python reproduction/run.py full --experiment highway --output /path/to/new-highway-replication
/tmp/pivot-repro-highway/bin/python reproduction/run.py full --experiment highway_redesign --output /path/to/new-highway-redesign
/tmp/pivot-repro-metadrive/bin/python reproduction/run.py full --experiment metadrive --output /path/to/new-metadrive
/tmp/pivot-paper-replay/bin/python reproduction/run.py full --experiment controlled --output /path/to/new-controlled
```

The `/tmp/pivot-repro-*` paths are examples from the current QA host. For another machine, create local environments from `reproduction/environments/*-smoke-requirements.txt` and substitute their Python paths. `full --experiment meltingpot` currently exits before execution because the specialist SavedModel assets and crossbench root are absent; its executable source is retained at `evidence/paper/source/core/run_melting_v4.py`.

The supplied supplement README describes the scientific scope of each cohort. In particular, E3/E5 are batch diagnostics, the first MetaDrive primary interval includes zero, and the second Leduc cohort was designed after first-cohort results. A successful saved-evidence check must not be read as an independent full-study replication.
