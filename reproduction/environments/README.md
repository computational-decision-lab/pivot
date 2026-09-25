# Reproduction environments

`replay-requirements.txt` is a lock captured from a tested Python 3.10.12 environment. It covers saved-label verification, aggregation, figure tools, and release tests. It does not install the native simulators.

The delivered Highway and Highway redesign snapshots each include a separate `requirements-lock.txt`. Those historical locks record Python 3.12-era packages including `highway-env==1.12.1`, `gymnasium==1.3.0`, and `numpy==2.5.3`. A separate current Python 3.11 smoke environment using `highway-env==1.12.1`, `gymnasium==1.3.0`, `numpy==2.4.6`, and `PyYAML==6.0.3` passed native steps and both source checks. This does not validate historical full-cohort outputs.

`openspiel-smoke-requirements.txt` records a separate Python 3.10.12 environment in which `open_spiel==1.6.11` passed native Kuhn and Leduc steps plus the frozen second-Leduc panel self-test. It is a current tested smoke environment, not a recovered historical lock or proof of a full-cohort rerun.

`metadrive-smoke-requirements.txt` records a current Python 3.10.12 environment where `metadrive-simulator==0.4.3` passed the frozen adapter's repeated native rollout. `meltingpot-smoke-requirements.txt` records another Python 3.10.12 environment where the official `dm-meltingpot==2.2.0` package passed a native repeated-stag-hunt substrate reset and step; JAX 0.4.18 and NumPy 1.26.4 were needed for its API. Both locks are current smoke environments, not historical full-run locks. Installing a simulator in a new environment does not certify byte-identical historical execution. Melting Pot full execution additionally needs its three specialist model assets and crossbench source root, which the portable paper artifact does not include. The controlled E2/E3/E4/E5/E7 diagnostics are separate historical runs; `full controlled` invokes their five confirmatory runners sequentially.

The original Melting Pot frozen confirmation protocol is also missing; its hash and the three model-file hashes are in `docs/protocol-chronology.md`. The native smoke uses no specialist model and cannot close that full-replay gap.
