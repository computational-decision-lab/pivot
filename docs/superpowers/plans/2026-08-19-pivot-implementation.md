# PIVOT Implementation Plan (Superseded)

> This earlier plan is retained for audit history. It is superseded by [2026-08-19-pivot-master-implementation.md](2026-08-19-pivot-master-implementation.md), which follows the authoritative `docs/master-goal.md` specification and the P0-P10/E1-E9 order.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible research harness that measures improvement fidelity for policy updates, demonstrates Improvement Reversal in a controlled adaptive world, and implements PIVOT's budgeted paired high-fidelity validation before adding finance and strategic extensions.

**Architecture:** Start with a small typed Python package. A `PolicyTransition` is the shared contract across policy operators, evaluators, environments, metrics, acquisition, and experiment logging. Evaluators expose paired deltas rather than forcing consumers to subtract independently estimated policy values. The controlled environment is the only required world for the first milestone; finance, adaptive opponents, LLM/EvoQuant, and M3 are adapters added only after explicit gates pass. The authoritative acronym is PIVOT: *Paired Interventional Verification of Optimization Transitions*.

**Tech Stack:** Python 3.11+, standard-library frozen dataclasses, `numpy`, `pandas`, `scipy`, `scikit-learn`, `PyYAML`, `pytest`, `ruff`, `mypy`, `matplotlib`, and JSONL/Parquet experiment artifacts.

## Global Constraints

- Preserve the frozen research object: a transition `pi -> pi'`, not a standalone policy leaderboard.
- Report `IDE`, `ISC`, `IRR`, `SIRR`, `MTR` (only away from zero denominators), `ISR`, `CTI`, and high-fidelity cost; use documented `tau_sign` and `tau_mtr` thresholds.
- Use paired rollouts with shared initial state, exogenous path, random seed, and opponent initialization whenever the environment supports them.
- Keep the focal agent as the only self-improving agent in the first strategic implementation.
- Do not begin LLM/EvoQuant or M3 integration until the controlled PIVOT budget gate passes.
- Keep all evaluations offline/shadow; no live orders, external side effects, or capital authorization are part of this project.
- Every experiment must emit a config snapshot, code version, seed list, raw result artifact, aggregate metrics, and a manifest with checksums.
- Use deterministic seeds and explicit train/validation/test episode splits; never tune on the final test split.
- Treat learned world models as alternative intervention models, never as ground truth.
- A reported positive result must include uncertainty intervals over independent seeds and the exact high-fidelity query budget.
- Production code must generate seven canonical figures; select the five central figures for the nine-page main text and place diagnostics in the appendix when needed.
- Preserve all nine experiment families `E1` through `E9`, all nine baseline families `B1` through `B9`, and the twelve required ablations from the master specification.
- The implementation order is fixed as `P0` through `P10`; no finance, LLM, or multi-agent code may enter P0/P1.
- Persist failed transitions and discarded-run records; never silently substitute fidelity levels, datasets, or environments.
- Plan against the supplied ICLR 2027 deadlines: abstract 2026-09-18 AOE, full paper 2026-09-25 AOE, and a nine-page main text.

## File Map

The implementation is expected to create the following focused modules:

```text
pyproject.toml
src/pivot/
  __init__.py
  config.py                 # YAML loading and schema defaults
  core/
    policy.py               # policy identity and immutable parameters
    transition.py           # canonical PolicyTransition record
    candidate.py            # candidate batch contracts
    world.py                # World 0/1/2 protocol
    result.py               # result, cost, and provenance schemas
  seeds.py                 # deterministic seed derivation and split registry
  improvers/
    perturbation.py        # controlled synthetic edits
    rl_update.py            # optional performative RL update operator
    typed_finance.py       # one-component finance edits
    llm_optional.py        # deferred typed LLM adapter
  environments/
    performative/           # controlled World 0/1/2 environment
    finance_backtest/       # F0
    execution_replay/       # F1
    interactive_market/     # F2
    strategic_market/       # F4 and S0/S1/S2
  footprint/
    generic.py
    finance.py
  evaluation/
    paired.py
    decomposition.py
    uncertainty.py
  transfer/
    global_value.py
    differential.py
    reversal.py
  acquisition/
    random.py
    top_proxy.py
    footprint.py
    uncertainty.py
    pivot.py
  algorithms/pivot.py
  metrics/improvement.py
  logging/transition_store.py
  adapters/world_model.py # deferred F3; never ground truth
configs/
  controlled/
  finance/
  strategic/
  sweeps/
experiments/
  e1_reversal.py
  e2_phase_diagram.py
  e3_overoptimization.py
  e4_global_vs_local.py
  e5_budget_frontier.py
  e6_finance_actor.py
  e7_strategic_reversal.py
  e8_competition.py
  e9_closed_loop.py
results/
  raw/
  processed/
  figures/
  tables/
tests/
  unit/
  integration/
scripts/
  run_sweep.py
  aggregate_results.py
  make_paper_figures.py
```

The exact file boundaries are part of the plan. A later task may add finance adapters under `src/pivot/environments/finance/`, but it must not move the controlled contracts.

## Task 1: Bootstrap the package and reproducibility contract

**Files:**
- Create: `pyproject.toml`
- Create: `src/pivot/__init__.py`
- Create: `src/pivot/config.py`
- Create: `src/pivot/seeds.py`
- Create: `tests/test_seeds.py`
- Create: `configs/controlled_smoke.yaml`

**Interfaces:**
- `derive_seed(master_seed: int, *labels: str) -> int`
- `make_split(master_seed: int, n_episodes: int) -> dict[str, list[int]]`
- `load_config(path: str | Path) -> dict[str, Any]`

- [ ] **Step 1: Write the failing seed tests.**

```python
def test_derive_seed_is_stable_and_label_sensitive():
    assert derive_seed(7, "train", "episode-1") == derive_seed(7, "train", "episode-1")
    assert derive_seed(7, "train", "episode-1") != derive_seed(7, "test", "episode-1")

def test_split_is_disjoint_and_complete():
    split = make_split(11, 30)
    ids = split["train"] + split["validation"] + split["test"]
    assert len(ids) == 30
    assert len(set(ids)) == 30
    assert set(ids) == set(range(30))
```

- [ ] **Step 2: Run the focused tests and verify the expected import failure.**

Run: `python -m pytest tests/test_seeds.py -q`
Expected: collection fails because `pivot.seeds` does not yet exist.

- [ ] **Step 3: Add the package metadata and deterministic implementation.**

Implement `derive_seed` with a stable SHA-256 digest of the integer seed and UTF-8 labels, reduced to a positive 32-bit integer. Implement `make_split` with fixed proportions `60/20/20`, preserving episode IDs exactly once. The smoke YAML must contain `master_seed: 17`, `n_episodes: 30`, `tau_sign: 1e-9`, and `high_fidelity_budget: 4`.

- [ ] **Step 4: Run the focused tests and lint.**

Run: `python -m pytest tests/test_seeds.py -q && ruff check src tests`
Expected: two tests pass and lint exits zero.

- [ ] **Step 5: Commit the bootstrap.**

```bash
git add pyproject.toml src/pivot/__init__.py src/pivot/seeds.py tests/test_seeds.py configs/controlled_smoke.yaml
git commit -m "chore: bootstrap reproducible pivot package"
```

## Task 2: Define the shared policy-transition and rollout contracts

**Files:**
- Create: `src/pivot/types.py`
- Create: `src/pivot/environments/protocols.py`
- Create: `src/pivot/evaluators/protocols.py`
- Create: `tests/test_types.py`

**Interfaces:**
- `Policy(parameters: Mapping[str, float], metadata: Mapping[str, str] = {})`
- `PolicyTransition(incumbent: Policy, candidate: Policy, operator: str, footprint: Mapping[str, float])`
- `RolloutPair(transition_id: str, incumbent_return: float, candidate_return: float, shared_context_id: str)`
- `EvaluationResult(delta: float, stderr: float, n_pairs: int, metadata: Mapping[str, Any])`
- `Environment.reset(seed: int) -> State`
- `Environment.step(state: State, action: Action, seed: int) -> tuple[State, float, bool, Mapping[str, Any]]`
- `PairedEvaluator.evaluate(transition: PolicyTransition, seeds: Sequence[int]) -> EvaluationResult`

- [ ] **Step 1: Write tests for immutability, stable IDs, and tie-safe serialization.**

```python
def test_transition_copies_input_policy_mappings():
    incumbent_values = {"threshold": 0.1}
    candidate_values = {"threshold": 0.2}
    transition = PolicyTransition(Policy(incumbent_values), Policy(candidate_values), "synthetic", {"l1": 0.1})
    candidate_values["threshold"] = 0.3
    assert transition.candidate.parameters["threshold"] == 0.2

def test_evaluation_result_round_trips_json():
    result = EvaluationResult(delta=0.4, stderr=0.1, n_pairs=4, metadata={"world": "actor"})
    assert EvaluationResult.from_dict(result.to_dict()) == result
```

- [ ] **Step 2: Run tests to confirm the contract is absent.**

Run: `python -m pytest tests/test_types.py -q`
Expected: import/attribute failures for the new types.

- [ ] **Step 3: Implement frozen dataclasses and protocols.**

Copy input mappings into immutable mappings, validate finite numeric policy parameters, require a non-empty `operator`, and derive `transition_id` from canonical JSON. Define protocol-only interfaces so environments and evaluators can be substituted without inheritance coupling.

- [ ] **Step 4: Verify serialization and type checks.**

Run: `python -m pytest tests/test_types.py -q && mypy src/pivot/types.py src/pivot/evaluators/protocols.py`
Expected: tests pass and mypy exits zero.

- [ ] **Step 5: Commit the contracts.**

```bash
git add src/pivot/types.py src/pivot/environments/protocols.py src/pivot/evaluators/protocols.py tests/test_types.py
git commit -m "feat: define policy transition and evaluation contracts"
```

## Task 3: Implement the controlled performative environment

**Files:**
- Create: `src/pivot/environments/controlled.py`
- Create: `tests/test_controlled_environment.py`
- Modify: `src/pivot/environments/protocols.py`

**Interfaces:**
- `ControlledConfig(response_strength: float, noise_scale: float, horizon: int, reward_bound: float)`
- `ControlledPerformativeEnv(config: ControlledConfig)`
- `evaluate_policy(policy: Policy, episode_seed: int, mode: Literal["observer", "actor", "strategic"]) -> float`
- `response_distance(policy_a: Policy, policy_b: Policy) -> float`

- [ ] **Step 1: Write tests for zero-response equivalence and monotonic response sensitivity.**

```python
def test_zero_response_matches_observer_and_actor():
    env = ControlledPerformativeEnv(ControlledConfig(response_strength=0.0, noise_scale=0.0, horizon=8, reward_bound=10.0))
    policy = Policy({"threshold": 0.4})
    assert env.evaluate_policy(policy, 3, "observer") == env.evaluate_policy(policy, 3, "actor")

def test_response_strength_changes_actor_value_for_footprinted_update():
    small = ControlledPerformativeEnv(ControlledConfig(response_strength=0.1, noise_scale=0.0, horizon=8, reward_bound=10.0))
    large = ControlledPerformativeEnv(ControlledConfig(response_strength=1.0, noise_scale=0.0, horizon=8, reward_bound=10.0))
    incumbent = Policy({"threshold": 0.2})
    candidate = Policy({"threshold": 0.8})
    assert small.response_distance(incumbent, candidate) == large.response_distance(incumbent, candidate)
    assert small.evaluate_policy(candidate, 5, "actor") != large.evaluate_policy(candidate, 5, "actor")
```

- [ ] **Step 2: Run the tests and confirm the environment is missing.**

Run: `python -m pytest tests/test_controlled_environment.py -q`
Expected: import failure for `pivot.environments.controlled`.

- [ ] **Step 3: Implement a bounded, analytically inspectable environment.**

Use a finite-horizon scalar state. The policy maps state to a bounded action via a threshold parameter. In observer mode, the transition response is fixed; in actor mode, the candidate's action changes the next-state drift by `response_strength * action`; in strategic mode, a deterministic opponent best-response term is applied. Keep rewards bounded by `reward_bound` and expose the exact response distance used by the theorem experiment.

- [ ] **Step 4: Verify deterministic replay and bounds.**

Run: `python -m pytest tests/test_controlled_environment.py -q`
Expected: all tests pass; add assertions that repeated `(policy, seed, mode)` calls are bitwise equal and rewards never exceed `reward_bound`.

- [ ] **Step 5: Commit the controlled world.**

```bash
git add src/pivot/environments/controlled.py src/pivot/environments/protocols.py tests/test_controlled_environment.py
git commit -m "feat: add controlled performative environment"
```

## Task 4: Add synthetic update operators and footprint extraction

**Files:**
- Create: `src/pivot/operators/synthetic.py`
- Create: `tests/test_operator.py`
- Modify: `src/pivot/types.py`

**Interfaces:**
- `generate_candidates(incumbent: Policy, n_candidates: int, seed: int, scale: float) -> list[PolicyTransition]`
- `compute_footprint(incumbent: Policy, candidate: Policy) -> dict[str, float]`

- [ ] **Step 1: Write tests for one-component edits and controlled footprint ordering.**

```python
def test_candidates_change_one_parameter_and_are_reproducible():
    incumbent = Policy({"threshold": 0.2, "gain": 0.5})
    a = generate_candidates(incumbent, 5, seed=9, scale=0.2)
    b = generate_candidates(incumbent, 5, seed=9, scale=0.2)
    assert [x.transition_id for x in a] == [x.transition_id for x in b]
    assert all(sum(v != 0 for v in (t.candidate.parameters[k] - incumbent.parameters[k] for k in incumbent.parameters)) == 1 for t in a)
```

- [ ] **Step 2: Run the focused test to verify the operator is absent.**

Run: `python -m pytest tests/test_operator.py -q`
Expected: import failure.

- [ ] **Step 3: Implement deterministic typed edits.**

Use a seeded generator, clamp parameters to their declared bounds, change exactly one component per candidate, and compute `l1`, `l2`, action-shift proxy, and support-change indicators. Store the footprint on each `PolicyTransition`.

- [ ] **Step 4: Verify reproducibility and edge cases.**

Run: `python -m pytest tests/test_operator.py -q`
Expected: all tests pass, including zero-scale edits being rejected with `ValueError` and duplicate transition IDs being impossible within one candidate batch.

- [ ] **Step 5: Commit the operator.**

```bash
git add src/pivot/operators/synthetic.py src/pivot/types.py tests/test_operator.py
git commit -m "feat: add controlled policy update operator"
```

## Task 5: Implement paired and proxy evaluators

**Files:**
- Create: `src/pivot/evaluators/controlled.py`
- Create: `src/pivot/evaluators/paired.py`
- Create: `tests/test_paired_evaluator.py`

**Interfaces:**
- `ProxyEvaluator.evaluate(transition, seeds) -> EvaluationResult`
- `PairedRolloutEvaluator(env, mode).evaluate(transition, seeds) -> EvaluationResult`
- `evaluate_transition(transition, evaluator, seeds) -> EvaluationResult`

- [ ] **Step 1: Write a common-random-number variance test.**

```python
def test_paired_delta_cancels_common_mode_noise(noisy_env, transition):
    evaluator = PairedRolloutEvaluator(noisy_env, mode="actor")
    result = evaluator.evaluate(transition, seeds=[1, 2, 3, 4])
    assert result.n_pairs == 4
    unpaired = noisy_env.unpaired_standard_error(transition, seeds=[1, 2, 3, 4])
    assert result.stderr < unpaired
```

- [ ] **Step 2: Run the test and confirm the evaluator is absent.**

Run: `python -m pytest tests/test_paired_evaluator.py -q`
Expected: import failure.

- [ ] **Step 3: Implement paired rollouts.**

For each seed, reset the same episode context, evaluate incumbent and candidate under the same exogenous path, subtract returns inside the evaluator, and compute sample mean and standard error of the paired deltas. Include `world`, `mode`, `seed_ids`, and `shared_context_id` in metadata. The proxy evaluator must use observer mode and may include a policy-independent offset to test differential invariance.

- [ ] **Step 4: Verify paired/unpaired behavior and no seed leakage.**

Run: `python -m pytest tests/test_paired_evaluator.py -q`
Expected: tests pass for variance reduction, exact reproducibility, and distinct train/test seed sets.

- [ ] **Step 5: Commit evaluators.**

```bash
git add src/pivot/evaluators/controlled.py src/pivot/evaluators/paired.py tests/test_paired_evaluator.py
git commit -m "feat: implement paired differential evaluators"
```

## Task 6: Add differential metrics and the first falsification experiment

**Files:**
- Create: `src/pivot/metrics.py`
- Create: `tests/test_metrics.py`
- Create: `scripts/run_controlled.py`
- Modify: `configs/controlled_main.yaml`

**Interfaces:**
- `compute_improvement_metrics(proxy: Sequence[float], true: Sequence[float], tau_sign: float) -> dict[str, float]`
- `bootstrap_ci(values: Sequence[float], seed: int, n_bootstrap: int = 2000) -> tuple[float, float]`
- `run_controlled(config_path: str | Path) -> Path`

- [ ] **Step 1: Write metric tests with hand-computed sign cases.**

```python
def test_reversal_rate_uses_only_positive_proxy_updates():
    metrics = compute_improvement_metrics([1.0, 2.0, -1.0], [0.5, -0.2, -0.4], tau_sign=1e-9)
    assert metrics["irr"] == 0.5
    assert metrics["isc"] == 2 / 3

def test_ties_are_excluded_from_sign_denominator():
    metrics = compute_improvement_metrics([0.0, 1.0], [0.0, -1.0], tau_sign=1e-9)
    assert metrics["n_ties"] == 1
```

- [ ] **Step 2: Run the metric tests to confirm failure.**

Run: `python -m pytest tests/test_metrics.py -q`
Expected: import failure.

- [ ] **Step 3: Implement metrics, bootstrap intervals, and JSONL artifacts.**

Compute `ide`, `isc`, `irr`, `n_reversals`, `n_positive_proxy`, `n_ties`, and paired selection regret. The runner must write `artifacts/runs/<run_id>/config.json`, `raw.jsonl`, `metrics.json`, and `manifest.sha256`; never overwrite an existing run directory.

- [ ] **Step 4: Run a smoke experiment.**

Run: `python scripts/run_controlled.py --config configs/controlled_smoke.yaml --output artifacts/runs`
Expected: a run directory with all four artifacts and a metrics JSON containing `ide`, `isc`, and `irr`.

- [ ] **Step 5: Commit metrics and the smoke runner.**

```bash
git add src/pivot/metrics.py tests/test_metrics.py scripts/run_controlled.py configs/controlled_main.yaml
git commit -m "feat: measure improvement fidelity and run controlled smoke"
```

### Gate A: controlled phenomenon

Run the main controlled grid over at least five independent seeds, three response strengths, and three update scales. Pass only if reversal is present away from the highest response corner, its confidence interval excludes zero in at least one interior cell, and `IRR` increases with both response strength and footprint under a predeclared monotonicity test. A failure means revise the environment/operator before implementing PIVOT.

## Task 7: Implement the differential transfer model and global-fidelity comparison

**Files:**
- Create: `src/pivot/models/differential.py`
- Create: `tests/test_differential_model.py`
- Modify: `src/pivot/runner.py`

**Interfaces:**
- `DifferentialTransferModel.fit(features: DataFrame, target_correction: Series) -> None`
- `DifferentialTransferModel.predict(features: DataFrame) -> ndarray`
- `compare_global_and_differential(proxy_values, proxy_deltas, true_values, true_deltas) -> DataFrame`

- [ ] **Step 1: Write tests for correction identity and held-out prediction.**

```python
def test_constant_value_offset_disappears_in_differential_target(train_frame, test_frame):
    model = DifferentialTransferModel(random_state=3)
    model.fit(train_frame.features, train_frame.true_delta - train_frame.proxy_delta)
    predictions = model.predict(test_frame.features)
    assert predictions.shape == test_frame.true_delta.shape
```

- [ ] **Step 2: Run the focused test and verify failure.**

Run: `python -m pytest tests/test_differential_model.py -q`
Expected: import failure.

- [ ] **Step 3: Implement a leakage-safe baseline model.**

Use a small regularized linear model as the first differential transfer model, with standardized footprint features, explicit train/validation/test fitting, and serialized coefficients. Implement a global value baseline with the same high-fidelity sample count. The comparison table must contain global value MAE/rank correlation and differential IDE/ISC/IRR.

- [ ] **Step 4: Verify held-out evaluation and equal budgets.**

Run: `python -m pytest tests/test_differential_model.py -q && python scripts/run_controlled.py --config configs/controlled_main.yaml --ablation global-vs-differential`
Expected: no test episode appears in model fitting; output records identical high-fidelity budgets for both models.

- [ ] **Step 5: Commit the model comparison.**

```bash
git add src/pivot/models/differential.py tests/test_differential_model.py src/pivot/runner.py
git commit -m "feat: compare global value and differential fidelity"
```

### Gate B: estimand necessity

Pass only if the global evaluator's policy-level rank quality does not completely remove held-out local sign errors at the matched budget, while the differential model improves IDE or ISC. If global fidelity solves the task equally well, narrow or abandon the novelty claim before PIVOT work.

## Task 8: Implement acquisition policies and PIVOT

**Files:**
- Create: `src/pivot/acquisition.py`
- Create: `tests/test_acquisition.py`
- Modify: `src/pivot/runner.py`

**Interfaces:**
- `select_random(candidates, budget, seed) -> list[str]`
- `select_top_proxy(candidates, budget) -> list[str]`
- `select_largest_footprint(candidates, budget) -> list[str]`
- `select_uncertain(candidates, model, budget) -> list[str]`
- `select_pivot(candidates, model, budget, cost_key="cost") -> list[str]`
- `run_pivot_round(incumbent, candidates, proxy_evaluator, hf_evaluator, policy, budget) -> RoundResult`

- [ ] **Step 1: Write tests for budget compliance and decision-change priority.**

```python
def test_pivot_never_exceeds_budget():
    selected = select_pivot(candidates, model, budget=3)
    assert len(selected) == 3

def test_pivot_prefers_candidate_that_can_change_argmax():
    selected = select_pivot([safe, ambiguous, low_value], model, budget=1)
    assert selected == ["ambiguous"]
```

- [ ] **Step 2: Run focused tests and verify failure.**

Run: `python -m pytest tests/test_acquisition.py -q`
Expected: import failure.

- [ ] **Step 3: Implement deterministic acquisition policies.**

Represent each candidate with proxy delta, footprint, predicted correction mean/variance, evaluation cost, and current selected rank. Define PIVOT's score as expected reduction in update-selection regret per cost, using a closed-form approximation from correction uncertainty and the margin to the incumbent/runner-up. Always return exactly `min(budget, n_candidates)` unique IDs and record the score used.

- [ ] **Step 4: Verify all baselines and round accounting.**

Run: `python -m pytest tests/test_acquisition.py -q && python scripts/run_controlled.py --config configs/controlled_main.yaml --policy pivot`
Expected: every policy consumes the same budget; each round records candidate IDs, queried IDs, proxy decisions, corrected decisions, and selection regret.

- [ ] **Step 5: Commit PIVOT.**

```bash
git add src/pivot/acquisition.py tests/test_acquisition.py src/pivot/runner.py
git commit -m "feat: add budgeted pivot acquisition"
```

### Gate C: budget frontier

Compare Proxy Only, Random HF, Top Proxy HF, Largest Footprint, Uncertainty, and PIVOT over identical budgets and seeds. Pass only if PIVOT's confidence interval for CTI or selection regret is better than Random HF and Top Proxy HF at one or more non-saturated budgets, without claiming superiority at the all-HF endpoint.

## Task 9: Add validation, manifests, and canonical figures

**Files:**
- Create: `src/pivot/validation.py`
- Create: `src/pivot/plotting.py`
- Create: `tests/test_validation.py`
- Create: `scripts/aggregate_results.py`
- Create: `scripts/make_figures.py`

**Interfaces:**
- `validate_run_manifest(run_dir: Path) -> ValidationReport`
- `aggregate_runs(run_dirs: Sequence[Path]) -> DataFrame`
- `make_canonical_figures(table: DataFrame, output_dir: Path) -> list[Path]`

- [ ] **Step 1: Write tests for checksums, seed coverage, and canonical output names.**

```python
def test_manifest_rejects_modified_raw_artifact(tmp_path):
    run_dir = make_valid_run(tmp_path)
    (run_dir / "raw.jsonl").write_text("tampered\n")
    report = validate_run_manifest(run_dir)
    assert not report.ok
    assert "raw.jsonl" in report.errors
```

- [ ] **Step 2: Run focused validation tests and verify failure.**

Run: `python -m pytest tests/test_validation.py -q`
Expected: import failure.

- [ ] **Step 3: Implement validation and five figure builders.**

Require config snapshot, raw results, metrics, manifest, code version, and disjoint seed sets. Produce exactly these stable names: `fig1_reversal_scatter.png`, `fig2_wrong_world.png`, `fig3_global_vs_improvement.png`, `fig4_budget_frontier.png`, and `fig5_world_ladder.png`. Include data tables next to each image so plots are regenerable and auditable.

- [ ] **Step 4: Verify an end-to-end aggregate and visual artifact set.**

Run: `python -m pytest tests/test_validation.py -q && python scripts/aggregate_results.py artifacts/runs --output artifacts/aggregate.parquet && python scripts/make_figures.py artifacts/aggregate.parquet --output artifacts/figures`
Expected: validation passes, aggregate exists, and all five canonical images plus their CSV data files are present.

- [ ] **Step 5: Commit reproducibility tooling.**

```bash
git add src/pivot/validation.py src/pivot/plotting.py tests/test_validation.py scripts/aggregate_results.py scripts/make_figures.py
git commit -m "feat: add reproducible result validation and figures"
```

## Task 10: Add finance replay and interactive actor adapters

**Files:**
- Create: `src/pivot/environments/finance/__init__.py`
- Create: `src/pivot/environments/finance/replay.py`
- Create: `src/pivot/environments/finance/interactive.py`
- Create: `tests/test_finance_adapters.py`
- Create: `configs/finance_smoke.yaml`

**Interfaces:**
- `ParticipationConfig(participation_rate: float, fee_bps: float, slippage_bps: float)`
- `ReplayMarket.evaluate_policy(policy, session_id, seed) -> float`
- `InteractiveMarket.evaluate_policy(policy, session_id, seed) -> float`
- `participation_rate(order_volume, market_volume) -> float`

- [ ] **Step 1: Write adapter contract tests.**

```python
def test_zero_participation_reduces_actor_to_replay():
    replay, actor = make_markets(participation_rate=0.0)
    assert actor.evaluate_policy(policy, "session-1", 4) == replay.evaluate_policy(policy, "session-1", 4)
```

- [ ] **Step 2: Run the tests and verify adapters are absent.**

Run: `python -m pytest tests/test_finance_adapters.py -q`
Expected: import failure.

- [ ] **Step 3: Implement offline, deterministic adapters.**

Use a supplied historical session fixture, explicit fee/slippage accounting, and virtual fills only. The interactive adapter must apply the focal order's footprint to available liquidity and record participation, impact, and fill metadata. No network, broker, exchange, or credential access is permitted.

- [ ] **Step 4: Verify the participation sweep.**

Run: `python -m pytest tests/test_finance_adapters.py -q && python scripts/run_controlled.py --config configs/finance_smoke.yaml --world-ladder`
Expected: W0/W1/W2 deltas are emitted with virtual accounting and no live-order markers.

- [ ] **Step 5: Commit the finance adapters.**

```bash
git add src/pivot/environments/finance tests/test_finance_adapters.py configs/finance_smoke.yaml
git commit -m "feat: add offline finance replay and actor adapters"
```

### Gate D: finance plausibility

Run the participation sweep over a predeclared defensible range. Pass only if the proxy/actor gap changes systematically with participation, survives fee/slippage controls, and does not depend on a single session or seed. If reversal requires implausibly large orders, narrow the finance claim and keep the controlled result primary.

## Task 11: Add adaptive opponents and strategic reversal experiment

**Files:**
- Create: `src/pivot/environments/finance/strategic.py`
- Create: `src/pivot/environments/opponents.py`
- Create: `tests/test_strategic_response.py`
- Modify: `src/pivot/environments/protocols.py`

**Interfaces:**
- `OpponentConfig(adaptation_strength: float, learning_rate: float, update_interval: int)`
- `AdaptiveOpponent.reset(seed: int) -> None`
- `AdaptiveOpponent.respond(observation: Mapping[str, float]) -> Action`
- `StrategicMarket.evaluate_policy(policy, session_id, seed) -> float`
- `opponent_sensitivity(incumbent, candidate, config) -> float`

- [ ] **Step 1: Write tests for fixed-opponent equivalence and adaptation monotonicity.**

```python
def test_zero_adaptation_matches_actor_world():
    actor, strategic = make_markets(adaptation_strength=0.0)
    assert strategic.evaluate_policy(policy, "s", 2) == actor.evaluate_policy(policy, "s", 2)

def test_opponent_response_is_seeded_and_observable():
    opponent = AdaptiveOpponent(OpponentConfig(adaptation_strength=0.5, learning_rate=0.1, update_interval=1))
    opponent.reset(8)
    first = opponent.respond({"flow": 1.0})
    opponent.reset(8)
    assert opponent.respond({"flow": 1.0}) == first
```

- [ ] **Step 2: Run focused tests and verify failure.**

Run: `python -m pytest tests/test_strategic_response.py -q`
Expected: import failure.

- [ ] **Step 3: Implement one focal agent plus adaptive competitors.**

Use a deterministic bounded best-response approximation with configurable adaptation strength and learning rate. Log opponent actions and state updates so a strategic reversal can be attributed to response, not hidden randomness. Keep all fills virtual.

- [ ] **Step 4: Run the strategic grid.**

Run: `python -m pytest tests/test_strategic_response.py -q && python scripts/run_controlled.py --config configs/finance_smoke.yaml --world strategic --adaptation-grid`
Expected: output includes `delta_proxy`, `delta_actor`, `delta_strategic`, and `opponent_sensitivity` for every grid cell.

- [ ] **Step 5: Commit strategic adapters.**

```bash
git add src/pivot/environments/finance/strategic.py src/pivot/environments/opponents.py tests/test_strategic_response.py src/pivot/environments/protocols.py
git commit -m "feat: add adaptive strategic response world"
```

### Gate E: strategic evidence

Prefer a stable interior region satisfying `Delta_proxy > 0`, `Delta_actor > 0`, and `Delta_strategic < 0`. If the adaptive opponents only inflate variance, move the result to the appendix and do not claim systematic strategic reversal.

## Task 12: Add deferred world-model and candidate-operator adapters

**Files:**
- Create: `src/pivot/evaluators/world_model.py`
- Create: `src/pivot/operators/typed.py`
- Create: `tests/test_deferred_adapters.py`
- Create: `docs/experiments/deferred-adapters.md`

**Interfaces:**
- `WorldModelEvaluator(model_path: Path, matching_engine: Callable)`
- `TypedEditOperator.generate(incumbent, edit_type, seed) -> list[PolicyTransition]`
- `compare_worlds(evaluators, transitions, seeds) -> DataFrame`

- [ ] **Step 1: Write adapter tests using local fakes only.**

```python
def test_world_model_is_labeled_as_alternative_intervention_model():
    result = WorldModelEvaluator(fake_model, fake_engine).evaluate(transition, [1])
    assert result.metadata["ground_truth"] is False

def test_typed_operator_records_edit_type():
    transitions = TypedEditOperator().generate(policy, "execution", seed=1)
    assert all(t.metadata["edit_type"] == "execution" for t in transitions)
```

- [ ] **Step 2: Run tests and verify deferred modules are absent.**

Run: `python -m pytest tests/test_deferred_adapters.py -q`
Expected: import failure until this task begins.

- [ ] **Step 3: Implement local-file adapters without external model downloads.**

The world-model adapter consumes a versioned local model and matching-engine callable, emits `world_model_id`, and sets `ground_truth: false`. The typed operator supports `signal`, `entry`, `exit`, `threshold`, `position`, `holding`, `rebalance`, and `execution`; one component changes per candidate.

- [ ] **Step 4: Verify cross-world disagreement artifacts.**

Run: `python -m pytest tests/test_deferred_adapters.py -q && python scripts/run_controlled.py --config configs/finance_smoke.yaml --compare-worlds`
Expected: a table of sign agreement/disagreement among replay, actor, strategic, and alternative model evaluators.

- [ ] **Step 5: Commit only after Gates A-E are recorded.**

```bash
git add src/pivot/evaluators/world_model.py src/pivot/operators/typed.py tests/test_deferred_adapters.py docs/experiments/deferred-adapters.md
git commit -m "feat: add deferred world-model and typed operator adapters"
```

## Final verification checklist

- [ ] `python -m pytest -q` passes with no skipped core tests.
- [ ] `ruff check src tests` and `mypy src` pass.
- [ ] A clean smoke run reproduces identical metrics for the same config and seed list.
- [ ] Every run has a valid checksum manifest and fails validation after artifact tampering.
- [ ] Controlled figures reproduce from saved aggregate tables without source-data access.
- [ ] Gates A-E are recorded in `docs/experiments/gates.md` with run IDs, configs, seeds, budgets, and confidence intervals.
- [ ] The main narrative does not depend on M3, LLM/EvoQuant, or strategic evidence until their gates are actually met.
- [ ] No artifact contains credentials, private logs, raw vendor L2, or live-order records.

## Dependency and milestone order

```text
Task 1 -> Task 2 -> Task 3 -> Task 4 -> Task 5 -> Task 6
                                                   |
                                                   v
                                             Gate A -> Task 7
                                                          |
                                                          v
                                                     Gate B -> Task 8
                                                                  |
                                                                  v
                                                             Gate C -> Task 9
                                                                          |
                                                                          v
                                                                     Gate D -> Task 11
                                                                          |
                                                                          v
                                                                     Gate E -> Task 12
```

Task 9 may be developed in parallel with Task 8 after the artifact schema is frozen, but its release verification depends on Task 8 output. Tasks 10 and 11 are finance stages; Task 11 cannot start until Task 10's adapter contract and Gate D pass. Task 12 is intentionally last.

## Risk register and mitigations

| Risk | Detection | Mitigation |
| --- | --- | --- |
| Reversal is only a noise artifact | Seed bootstrap intervals, response/footprint grid, deterministic replay checks | Increase paired episodes, simplify the analytic environment, and stop before method claims if the effect disappears |
| Global evaluator is already sufficient | Matched-budget held-out comparison in Gate B | Narrow the estimand claim or replace the proxy with a deliberately misspecified but documented observer world |
| PIVOT exploits an implementation artifact | Equal-cost baselines, frozen candidate batches, logged acquisition scores | Add a second controlled response family and rerun without tuning on test seeds |
| Finance effect needs unrealistic participation | Participation plausibility bounds and session-level replication | Keep controlled evidence primary and report finance as a bounded testbed result |
| Strategic opponents add variance only | Compare mean response and variance separately across adaptation strengths | Move strategic result to appendix and avoid claiming systematic strategic reversal |
| Artifact leakage or non-reproducibility | Manifest validation, clean-room rerun, seed-disjointness checks | Fail the run, regenerate from config, and preserve the failed manifest for audit |

## Execution handoff

The first execution batch is Tasks 1-3 only. It should end with a deterministic controlled environment and no optimizer, LLM, finance data, or world-model dependency. After that batch, record Gate A readiness in `docs/experiments/gates.md` before starting model or acquisition work.
