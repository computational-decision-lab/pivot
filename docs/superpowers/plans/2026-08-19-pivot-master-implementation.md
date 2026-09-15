# PIVOT Master Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a rigorous, reproducible framework that measures whether a self-improving policy update remains beneficial after the update changes the environment and, later, after other agents adapt.

**Architecture:** `PolicyTransition` is the only scientific unit passed between improvement operators, worlds, evaluators, footprint extractors, transfer models, acquisition policies, metrics, and artifact storage. World 0 produces proxy deltas, World 1 produces endogenous actor deltas, and World 2 produces strategic deltas. PIVOT estimates transition-level corrections and allocates a fixed high-fidelity budget; it never becomes an action-authorization or abstention gate.

**Tech Stack:** Python 3.10+, typed dataclasses or Pydantic-style schemas, NumPy/Pandas/SciPy/scikit-learn, PyYAML, pytest, Ruff, mypy, Matplotlib, Parquet, JSON/YAML provenance, and SHA-256 manifests.

**Live implementation note (2026-08-20):** P0-P9 harnesses, registered fixture
runs, a seven-session single-asset public calibration, and a frozen 12-session
three-asset public expansion are present in the checkout. The checkboxes below
remain the original execution contract rather than a live completion ledger;
current implementation and gate status are tracked in
`docs/implementation-status.md`, `docs/experiments/gates.md`,
`docs/experiments/public-finance-evidence-2026-08-19.md`, and
`docs/experiments/public-expansion-evidence-2026-08-19.md`. The public depth
audits are observational, not causal actor-world validation. P10 remains
intentionally deferred. The IMPROVE-X V5 platform extension (transition
contracts, ImprovementBench, multi-round trajectories, failure taxonomy, and
PIVOT-X scoring) is complete as a separately verified vertical slice; its
confirmatory external evaluation is still open.

## Global Constraints

- The authoritative research object is `pi_t -> pi'_{t,j}`. Do not substitute a policy leaderboard, action gate, authorization decision, or trajectory score.
- PIVOT means **Paired Interventional Validation of Optimization Transitions**.
- Distinguish `Delta_proxy`, `Delta_actor`, and `Delta_strategic`; unavailable values are explicit `null`, never silently substituted.
- Use paired evaluation with matched initial states, scenarios, exogenous seeds, market days, order flow, and opponent initialization whenever possible.
- Report IDE, ISC, IRR, SIRR, MTR with a denominator threshold, ISR, CTI, and high-fidelity cost.
- Preserve all transition-level raw rows, including failed and discarded transitions; do not filter inconvenient seeds after the fact.
- Every run persists config, seed, git commit, dependency versions, environment version, dataset/version ID, timestamp, machine information, raw data, processed data, and a checksum manifest.
- No live orders, credentials, external execution, production trading, or capital authorization.
- No LLM, EvoQuant, F3 learned world model, finance, or multi-agent code in P0/P1.
- Do not claim a theorem or empirical pattern until the corresponding gate evidence exists.
- The supplied ICLR 2027 planning dates are abstract 2026-09-18 AOE, full paper 2026-09-25 AOE, and nine-page main text.

## Authoritative Vocabulary

| Concept | Contract |
| --- | --- |
| World 0 | Observer/fixed world; outputs `Delta_proxy` |
| World 1 | Actor/endogenous world; outputs `Delta_actor` |
| World 2 | Strategic/adaptive-opponent world; outputs `Delta_strategic` |
| F0/F1/F2/F3/F4 | Backtest, execution replay, interactive actor, alternative generative model, strategic market |
| S0/S1/S2 | Fixed, reactive, finite-step adaptive opponents |
| P0-P10 | Mandatory implementation order |
| E1-E9 | Mandatory experiment order |
| B1-B9 | Mandatory baseline families |

## Experiment and Baseline Map

| ID | Experiment |
| --- | --- |
| E1 | Does Improvement Reversal exist? |
| E2 | Response strength x update footprint phase diagram |
| E3 | Performative overoptimization |
| E4 | Global Fidelity versus Improvement Fidelity |
| E5 | PIVOT budget frontier |
| E6 | Financial mechanical reversal |
| E7 | Strategic Improvement Reversal |
| E8 | Competition-strength sweep |
| E9 | Closed-loop self-improvement |

| ID | Baseline |
| --- | --- |
| B1 | Proxy Only |
| B2 | Random High Fidelity |
| B3 | Top Proxy |
| B4 | Largest Footprint |
| B5 | Global Value Model |
| B6 | Global Ranking Model |
| B7 | Uncertainty Sampling |
| B8 | All-HF Oracle |
| B9 | PIVOT |

## Repository Contract

```text
pivot/
├── README.md
├── pyproject.toml
├── configs/{controlled,finance,strategic,sweeps,registered}/
├── src/pivot/
│   ├── core/{policy.py,transition.py,candidate.py,world.py,result.py}
│   ├── improvers/{perturbation.py,rl_update.py,typed_finance.py,llm_optional.py}
│   ├── environments/{performative,finance_backtest,execution_replay,interactive_market,strategic_market}/
│   ├── footprint/{generic.py,finance.py}
│   ├── evaluation/{paired.py,decomposition.py,uncertainty.py}
│   ├── transfer/{global_value.py,differential.py,reversal.py}
│   ├── acquisition/{random.py,top_proxy.py,footprint.py,uncertainty.py,pivot.py}
│   ├── algorithms/pivot.py
│   ├── metrics/improvement.py
│   ├── logging/transition_store.py
│   └── adapters/world_model.py
├── experiments/e{1_reversal,2_phase_diagram,3_overoptimization,4_global_vs_local,5_budget_frontier,6_finance_actor,7_strategic_reversal,8_competition,9_closed_loop}.py
├── scripts/{run_sweep.py,aggregate_results.py,make_paper_figures.py}
├── tests/{unit,integration}/
├── results/{raw,processed,figures,tables,registered}/
└── docs/
```

## Canonical Schemas

### `PolicyTransition`

The Parquet transition table must contain at least:

```text
transition_id, round_id, incumbent_policy_id, candidate_policy_id,
candidate_index, improvement_operator, edit_type,
proxy_world_id, high_fidelity_world_id,
proxy_incumbent_value, proxy_candidate_value, delta_proxy,
actor_incumbent_value, actor_candidate_value, delta_actor,
strategic_incumbent_value, strategic_candidate_value, delta_strategic,
mechanical_effect, competition_effect,
improvement_reversal, strategic_improvement_reversal,
update_footprint, footprint_components,
response_strength, competition_strength, opponent_context,
hf_queried, hf_query_reason, hf_query_cost,
seed, paired_seed_ids, config_id, git_commit, timestamp
```

### Run provenance

```text
config_snapshot, master_seed, split_ids, git_commit,
dependency_versions, environment_version, dataset_version,
timestamp_utc, machine_info, hf_budget, tau_sign, tau_mtr
```

### Metrics

```text
IDE  = mean(abs(delta_proxy - delta_true))
ISC  = P(sign(delta_proxy) == sign(delta_true))
IRR  = P(delta_true < 0 | delta_proxy > 0)
SIRR = P(delta_strategic < 0 | delta_actor > 0)
MTR  = delta_true / delta_proxy when abs(delta_proxy) > tau_mtr
ISR  = max_j delta_true_j - delta_true_selected
CTI  = sum_t selected_delta_true_t
```

Every aggregate includes counts, paired/unpaired status, bootstrap or analytic confidence intervals, and high-fidelity transition/rollout/step/call cost.

## Phase P0: Core Contracts, Paired Evaluation, Metrics, and Logging

**Purpose:** Establish the estimand and artifact contract before any environment complexity.

**Files:**
- Create: `src/pivot/core/policy.py`
- Create: `src/pivot/core/transition.py`
- Create: `src/pivot/core/candidate.py`
- Create: `src/pivot/core/world.py`
- Create: `src/pivot/core/result.py`
- Create: `src/pivot/evaluation/paired.py`
- Create: `src/pivot/evaluation/decomposition.py`
- Create: `src/pivot/evaluation/uncertainty.py`
- Create: `src/pivot/footprint/generic.py`
- Create: `src/pivot/metrics/improvement.py`
- Create: `src/pivot/logging/transition_store.py`
- Create: `tests/unit/test_transition.py`
- Create: `tests/unit/test_metrics.py`
- Create: `tests/unit/test_paired.py`
- Create: `tests/unit/test_decomposition.py`
- Create: `tests/unit/test_generic_footprint.py`
- Create: `tests/unit/test_store.py`

**Interfaces:**
- `Policy.from_mapping(values: Mapping[str, float]) -> Policy`
- `PolicyTransition.to_record() -> Mapping[str, Any]`
- `World.evaluate(policy: Policy, context: RolloutContext) -> RolloutResult`
- `PairedEvaluator.evaluate(transition: PolicyTransition, contexts: Sequence[RolloutContext]) -> PairedEvaluation`
- `compute_improvement_metrics(rows: DataFrame, tau_sign: float, tau_mtr: float) -> DataFrame`
- `compute_update_footprint(pi: Policy, pi_prime: Policy, evaluation_states: Sequence[State]) -> Footprint`
- `TransitionStore.append(row: Mapping[str, Any]) -> None`
- `TransitionStore.finalize() -> Manifest`

- [ ] **Step 1: Write failing tests for immutable policy IDs, full-schema serialization, paired subtraction, known decomposition, metric formulas, and component-level footprint output.**

```python
def test_policy_transition_round_trips_with_explicit_nulls():
    transition = make_transition(delta_actor=None, delta_strategic=None)
    record = transition.to_record()
    assert "delta_actor" in record and record["delta_actor"] is None
    assert PolicyTransition.from_record(record) == transition

def test_paired_evaluator_subtracts_inside_each_shared_context():
    result = PairedEvaluator(fake_world).evaluate(transition, [ctx1, ctx2])
    assert result.deltas == [ctx1.candidate_return - ctx1.incumbent_return,
                             ctx2.candidate_return - ctx2.incumbent_return]

def test_known_decomposition_is_exact():
    values = decompose(direct=2.0, actor=1.25, strategic=0.5)
    assert values.mechanical == -0.75
    assert values.competition == -0.75

def test_footprint_preserves_components():
    result = compute_update_footprint(pi, pi_prime, evaluation_states)
    assert {"mean_kl", "max_kl", "action_shift", "entropy_change",
            "occupancy_divergence", "support_expansion",
            "trajectory_divergence", "episode_length_change"} <= result.components.keys()
```

- [ ] **Step 2: Run `python -m pytest tests/unit/test_transition.py tests/unit/test_metrics.py tests/unit/test_paired.py tests/unit/test_decomposition.py -q` and confirm collection failures.**

- [ ] **Step 3: Implement frozen typed schemas, canonical JSON IDs, paired differences, zero-tolerance sign handling, MTR denominator handling, paired bootstrap intervals, component-level footprint extraction, cost accounting, and append-only Parquet/JSONL storage. `PairedEvaluation` must expose incumbent value, candidate value, delta, standard error, confidence interval, rollout count, and paired seed IDs.**

- [ ] **Step 4: Verify `python -m pytest tests/unit -q`, `ruff check src tests`, and `mypy src`; tamper with a stored row and confirm manifest validation fails.**

- [ ] **Step 5: Commit `feat: establish pivot transition and evaluation contracts`.**

**P0 exit criterion:** A unit-only toy transition can be serialized, paired-evaluated, decomposed, scored, and persisted without any environment-specific import.

The paired-decomposition tests must use an analytically solvable toy world where `Delta_direct`, `mechanical_effect`, and `competition_effect` are known exactly; a mock that only returns arbitrary numbers is insufficient coverage.

## Phase P1: Controlled Performative World

**Purpose:** Build the first scientifically complete world with known policy-dependent dynamics. No finance, LLM, or multi-agent code.

**Files:**
- Create: `src/pivot/environments/performative/config.py`
- Create: `src/pivot/environments/performative/world.py`
- Create: `src/pivot/environments/performative/proxy.py`
- Create: `src/pivot/improvers/perturbation.py`
- Create: `src/pivot/improvers/rl_update.py`
- Create: `configs/controlled/smoke.yaml`
- Create: `configs/controlled/main.yaml`
- Create: `tests/unit/test_performative_world.py`
- Create: `tests/unit/test_perturbation.py`
- Create: `tests/unit/test_rl_update.py`
- Create: `tests/integration/test_first_round.py`

**Interfaces:**
- `PerformativeConfig(response_strength: float, noise_scale: float, horizon: int, reward_bound: float, optimization_strength: float)`
- `PerformativeWorld.evaluate(policy, seed, mode: Literal["observer", "actor"]) -> RolloutResult`
- `SyntheticPerturbation.propose(incumbent, scale, num_candidates, seed) -> list[PolicyTransition]`
- `RLUpdateOperator.propose(incumbent, context, num_candidates, optimization_strength, seed) -> list[PolicyTransition]`
- `response_distance(pi, pi_prime) -> float`

- [ ] **Step 1: Write tests for observer/actor equality at zero response, deterministic seeds, bounded rewards, and response changes for a fixed footprint.**

- [ ] **Step 2: Run the focused tests and confirm the new modules are absent.**

- [ ] **Step 3: Implement a finite-horizon scalar-state environment with explicit `response_strength`, fixed observer dynamics, policy-dependent actor drift, and an analytically inspectable reward. Implement tiny/small/medium/large one-component perturbations and an ordinary policy-gradient update operator with configurable optimization strength; preserve all footprint components. State-of-the-art RL performance is not an objective.**

- [ ] **Step 4: Run `python -m pytest tests/unit/test_performative_world.py tests/unit/test_perturbation.py tests/integration/test_first_round.py -q`; verify repeated `(policy, seed, mode)` calls are identical and no result exceeds `reward_bound`.**

- [ ] **Step 5: Commit `feat: add controlled performative world and perturbation operator`.**

**P1 exit criterion:** One local command creates a transition table containing `round`, incumbent/candidate IDs, proxy and true values/deltas, reversal flag, footprint, response strength, and seed; no finance, LLM, or multi-agent import exists in the P1 path.

## Phase P2: E1/E2 Phenomenon and Structure

**Purpose:** Falsify or support the existence and structure of Improvement Reversal before building PIVOT.

**Files:**
- Create: `experiments/e1_reversal.py`
- Create: `experiments/e2_phase_diagram.py`
- Create: `experiments/e3_overoptimization.py`
- Create: `scripts/run_sweep.py`
- Create: `scripts/aggregate_results.py`
- Create: `docs/experiments/p2_protocol.md`
- Create: `tests/integration/test_e1_e2_outputs.py`
- Create: `tests/integration/test_e3_overoptimization.py`

- [ ] **Step 1: Register a grid over at least five independent seeds, three response strengths, three update scales, and three optimization strengths. Freeze the grid in `configs/sweeps/p2.yaml`.**

- [ ] **Step 2: Write output-schema tests requiring the transition columns, confidence intervals, failed-run records, and five first-milestone artifacts.**

- [ ] **Step 3: Implement E1 scatter generation and E2 response-by-footprint aggregation without tuning to force a zero crossing. Run E3 only after E1/E2 rows exist: repeatedly apply the fixed update operator and track `J_V(pi_t)` and `J_*(pi_t)` without assuming the true curve must deteriorate.**

- [ ] **Step 4: Run `python experiments/e1_reversal.py --config configs/sweeps/p2.yaml`, `python experiments/e2_phase_diagram.py --config configs/sweeps/p2.yaml`, and `python experiments/e3_overoptimization.py --config configs/sweeps/p2.yaml`; validate all artifacts from a clean process.**

- [ ] **Step 5: Record Gate A (phenomenon) and Gate B (structure) in `docs/experiments/gates.md` with run IDs, config hashes, seeds, budgets, and intervals.**

**P2 gate:** Proceed only if reversal appears under non-pathological settings and varies systematically with response and/or footprint. A null or noise-only result is a valid stop/narrow-claim outcome.

**First-milestone command:**

```bash
python scripts/run_sweep.py --config configs/sweeps/p2.yaml --milestone first
```

It must create one Parquet transition dataset and exactly these first-milestone outputs: `proxy_vs_true_scatter`, `irr_vs_response`, `irr_vs_footprint`, `response_footprint_heatmap`, and a confidence-interval table. No finance, LLM, or multi-agent module may be imported by this command.

## Phase P3: E4 Global Fidelity versus Improvement Fidelity

**Purpose:** Test whether the new estimand is necessary rather than assumed.

**Files:**
- Create: `src/pivot/transfer/global_value.py`
- Create: `src/pivot/transfer/differential.py`
- Create: `src/pivot/transfer/reversal.py`
- Create: `experiments/e4_global_vs_local.py`
- Create: `configs/sweeps/e4.yaml`
- Create: `tests/unit/test_transfer.py`
- Create: `tests/integration/test_e4_budget_match.py`

**Interfaces:**
- `GlobalValueModel.fit(policy_features, high_fidelity_values) -> None`
- `DifferentialModel.fit(transition_features, high_fidelity_corrections) -> None`
- `predict_correction(transition) -> CorrectionPrediction`
- `compare_global_vs_local(test_rows, budget) -> DataFrame`

- [ ] **Step 1: Write held-out tests proving train/test transition IDs are disjoint and both models consume identical high-fidelity budgets.**

- [ ] **Step 2: Implement ridge regression, gradient-boosted trees, and the mandatory global ranking baseline; a small MLP is allowed only if the dataset size justifies it. Do not begin with transformers.**

- [ ] **Step 3: Run E4 and emit policy-value MAE, policy rank correlation, IDE, ISC, IRR, and ISR.**

- [ ] **Step 4: Verify `python experiments/e4_global_vs_local.py --config configs/sweeps/e4.yaml` and inspect the saved budget ledger.**

- [ ] **Step 5: Record Gate C (new estimand). If the global evaluator completely solves the update problem, preserve that null result and narrow the paper claim before PIVOT.**

## Phase P4: Minimal PIVOT and Baselines

**Purpose:** Add transition-level correction and budget-aware acquisition only after E1-E4 evidence exists.

**Files:**
- Create: `src/pivot/acquisition/random.py`
- Create: `src/pivot/acquisition/top_proxy.py`
- Create: `src/pivot/acquisition/footprint.py`
- Create: `src/pivot/acquisition/uncertainty.py`
- Create: `src/pivot/acquisition/pivot.py`
- Create: `src/pivot/algorithms/pivot.py`
- Create: `experiments/e5_budget_frontier.py`
- Create: `configs/sweeps/e5.yaml`
- Create: `tests/unit/test_acquisition.py`
- Create: `tests/integration/test_pivot_round.py`

**Interfaces:**
- `select_random(candidates, budget, seed) -> list[str]`
- `select_top_proxy(candidates, budget) -> list[str]`
- `select_largest_footprint(candidates, budget) -> list[str]`
- `select_uncertainty(candidates, model, budget) -> list[str]`
- `select_pivot(candidates, model, budget, cost_key) -> list[str]`
- `run_pivot_round(incumbent, candidates, proxy, hf, acquisition, budget) -> RoundResult`

- [ ] **Step 1: Write budget-compliance and decision-change tests.**

```python
def test_all_selectors_obey_identical_budget():
    for selector in selectors:
        assert len(selector(candidates, budget=3)) == 3

def test_pivot_prioritizes_a_candidate_that_can_change_selection():
    assert select_pivot(candidates, model, budget=1, cost_key="cost") == ["ambiguous"]
```

- [ ] **Step 2: Implement correction target `Delta_H - Delta_proxy`, uncertainty, expected decision-change reduction per cost, and exact query ledgers.**

- [ ] **Step 3: Implement B1-B9 without unnecessary weak baselines: Proxy Only, Random HF, Top Proxy, Largest Footprint, Global Value, Global Ranking, Uncertainty, All-HF Oracle, PIVOT.**

- [ ] **Step 4: Run E5 with multiple seeds and paired evaluation where possible; produce CTI and ISR against matched HF budgets.**

- [ ] **Step 5: Record Gate D (method). PIVOT must beat Random HF and Top Proxy HF at a fixed budget on CTI or ISR; otherwise stop and revise the acquisition mechanism.**

## Phase P5: E5 Production Frontier and Figure Pipeline

**Purpose:** Make all core controlled evidence reproducible from scripts and tables.

**Files:**
- Create: `scripts/make_paper_figures.py`
- Create: `src/pivot/validation.py`
- Create: `tests/integration/test_figure_reproduction.py`
- Create: `docs/experiments/figure_schema.md`

- [ ] **Step 1: Write tests for seven canonical figure names, adjacent data tables, manifest checksums, and clean-room regeneration.**

- [ ] **Step 2: Implement stable outputs `fig1_when_better_gets_worse`, `fig2_reversal_phase_diagram`, `fig3_optimizing_wrong_world`, `fig4_policy_vs_improvement_fidelity`, `fig5_pivot_budget_frontier`, `fig6_observer_actor_strategic`, and `fig7_strategic_reversal`. Each image has a same-stem CSV/Parquet source table. Figures 6-7 remain explicitly unavailable until P8 rather than being fabricated from controlled data.**

- [ ] **Step 3: Run `python scripts/make_paper_figures.py --input results/processed --output results/figures` and verify every figure has a source table and config hash.**

- [ ] **Step 4: Record the first three milestones and update the gate ledger; do not claim any empirical gate as passed without its evidence record.**

## Phase P6: F0/F1 Finance Replay

**Purpose:** Add finance only after the controlled estimand and PIVOT budget frontier are established.

**Files:**
- Create: `src/pivot/environments/finance_backtest/`
- Create: `src/pivot/environments/execution_replay/`
- Create: `src/pivot/footprint/finance.py`
- Create: `src/pivot/improvers/typed_finance.py`
- Create: `experiments/e6_finance_actor.py`
- Create: `configs/finance/f0_f1.yaml`
- Create: `tests/unit/test_finance_replay.py`

- [ ] **Step 1: Write tests that F0 has no endogenous response, F1 adds spread/fees/slippage/partial fills/queue when available, and all fills are virtual.**

- [ ] **Step 2: Implement a versioned local fixture, explicit strategy-frequency versus simulation-frequency metadata, and one-component typed edits for signal, entry, exit, threshold, position size, risk size, holding horizon, rebalance frequency, urgency, and participation.**

- [ ] **Step 3: Implement finance footprint components: turnover, size, participation, holding period, rebalance frequency, order distribution, urgency, aggressive/passive ratio, liquidity consumption, concentration, inventory duration, and spread crossing.**

- [ ] **Step 4: Run F0/F1 across sessions and seeds; preserve failed sessions and emit cost/provenance fields.**

- [ ] **Step 5: Record finance readiness before F2.**

## Phase P7: F2 Interactive Actor and Participation Experiment

**Purpose:** Test mechanical response with a physically interpretable causal knob.

**Files:**
- Create: `src/pivot/environments/interactive_market/`
- Create: `configs/finance/f2_participation.yaml`
- Modify: `experiments/e6_finance_actor.py`
- Create: `tests/integration/test_participation_sweep.py`

- [ ] **Step 1: Write tests for zero participation equivalence to replay and increasing participation changing impact/liquidity/recovery metadata.**

- [ ] **Step 2: Implement F2 with minimal validated endogenous impact, liquidity depletion, recovery/reversion, and execution-state feedback; do not build an unnecessary exchange simulator.**

- [ ] **Step 3: Hold `pi -> pi'` fixed and sweep participation, size, urgency, and turnover; compare `Delta_proxy`, `Delta_replay`, and `Delta_actor`.**

- [ ] **Step 4: Run the plausibility checks and record Gate E (finance). If reversal requires absurd footprint, keep finance secondary and do not proceed as if the gate passed.**

## Phase P8: Add F4 Strategic Market and S0/S1/S2 Opponents

**Purpose:** Add competition as a higher response layer, with one focal self-improver, without yet interpreting a strategic-reversal result.

**Files:**
- Create: `src/pivot/environments/strategic_market/`
- Create: `configs/strategic/s0_fixed.yaml`
- Create: `configs/strategic/s1_reactive.yaml`
- Create: `configs/strategic/s2_adaptive.yaml`
- Create: `src/pivot/footprint/generic.py` (strategic sensitivity support)
- Create: `tests/unit/test_opponents.py`

- [ ] **Step 1: Write tests for S0 equivalence to F2, deterministic S1 response rules, and finite-step S2 adaptation with logged opponent state.**

- [ ] **Step 2: Implement noise traders, one liquidity provider, and one adaptive competitor; only the focal policy runs PIVOT.**

- [ ] **Step 3: Implement sweepable adaptation steps `K`, learning rate, opponent count, and market-share sensitivity; expose `S_-i` and SIRR computation but do not tune parameters against a desired sign crossing.**

- [ ] **Step 4: Run deterministic S0/S1/S2 smoke tests and freeze the strategic environment version before P9.**

## Phase P9: E7/E8 Strategic Evidence, Then Closed-Loop E9

**Purpose:** First measure strategic reversal and competition strength, then test repeated self-improvement only after E1-E8 are complete.

**Files:**
- Create: `experiments/e7_strategic_reversal.py`
- Create: `experiments/e8_competition.py`
- Create: `experiments/e9_closed_loop.py`
- Create: `configs/sweeps/e9.yaml`
- Create: `tests/integration/test_strategic_experiments.py`
- Create: `tests/integration/test_closed_loop.py`

- [ ] **Step 1: Run E7 on identical transitions across F2 and F4, then E8 over opponent count, adaptation steps/rate, and market-share sensitivity. Record Gate F; if competition only adds variance, move it to secondary analysis and narrow the claim.**

- [ ] **Step 2: Write a test for `pi_t -> candidates -> proxy -> PIVOT -> HF query -> pi_{t+1}` with a fixed budget ledger.**

- [ ] **Step 3: Implement promotion and consequence logging without converting PIVOT into an authorization gate. Compare final `J_H(pi_T)` and `CTI_T` under equal HF budgets across B1-B9.**

- [ ] **Step 4: Preserve all rounds, including harmful updates and rejected candidates, in the transition store.**

## Phase P10: Deferred LLM/EvoQuant and F3 Adapters

**Purpose:** Add realistic candidate operators and alternative learned worlds only after all core gates.

**Files:**
- Modify: `src/pivot/improvers/typed_finance.py`
- Create: `src/pivot/improvers/llm_optional.py`
- Create: `src/pivot/adapters/world_model.py`
- Create: `tests/unit/test_deferred_adapters.py`
- Create: `docs/experiments/deferred-adapters.md`

- [ ] **Step 1: Write tests requiring typed executable diffs, and requiring every world-model result to carry `ground_truth: false`.**

- [ ] **Step 2: Adapt the existing typed finance edits to an optional EvoQuant-compatible candidate interface without reproducing EvoQuant's evolution system.**

- [ ] **Step 3: Add optional LLM persistence for prompt, response, edit, compilation result, and strategy artifact; never place an LLM in the event-level execution hot path.**

- [ ] **Step 4: Add local-file F3 adapter and sign-disagreement comparison without downloading or training a market foundation model.**

- [ ] **Step 5: Run only after P2-P9 evidence is recorded; otherwise leave adapters unexecuted.**

## Required Ablation Matrix

Every final report includes:

```text
paired vs unpaired
transition vs global value
with vs without footprint
active vs random HF
PIVOT vs Top Proxy
small vs large updates
weak vs strong response
F1 vs F2
fixed vs adaptive competitors
single vs multiple response models
candidate count
HF budget
```

## Go/No-Go Ledger

| Gate | Requirement | Evidence |
| --- | --- | --- |
| A Phenomenon | Reversal in non-pathological controlled settings | E1 run IDs, intervals |
| B Structure | IRR relates to footprint/response | E2 heatmap and regression |
| C Estimand | Global policy evaluation does not fully solve local transition quality | E4 matched-budget table |
| D Method | PIVOT beats Random HF and Top Proxy at fixed budget | E5 CTI/ISR intervals |
| E Finance | F0/F1 versus F2 difference at plausible footprint | participation sweep |
| F Competition | Strategic response adds systematic effect beyond mechanical response | E7/E8 SIRR and sensitivity |

## Final Verification

- [ ] `python -m pytest -q` passes unit and integration tests.
- [ ] `ruff check src tests` and `mypy src` pass.
- [ ] A clean-room run reproduces transition rows, metrics, and all figure source tables from configuration and seeds.
- [ ] All seven figures regenerate from scripts; five can be selected for the nine-page main text.
- [ ] Every result includes high-fidelity budget and uncertainty; no reversal is based on one trajectory.
- [ ] `docs/experiments/gates.md` records each gate as `Not run`, `Pass`, `Fail`, or `Narrowed`, with evidence links.
- [ ] No deferred adapter is used to support a core claim before P10.
- [ ] No credential, private log, raw vendor L2, live order, or production artifact is present.
