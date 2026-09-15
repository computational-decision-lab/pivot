# PIVOT Next-Step Benchmark Plan

## Purpose

This two-week sprint turns the repository into a single, reproducible execution baseline for the ICLR paper. The objective is not to maximize benchmark count. It is to close the mechanism loop around **Improvement Fidelity**: whether an update that improves a cheap verifier remains an improvement after deployment changes the world and other agents respond.

The scientific object is a directed update transition:

$$\pi_t \rightarrow \pi_{t+1}$$

We distinguish proxy improvement ($\Delta_V$), deployment improvement ($\Delta_*$), and improvement reversal ($\Delta_V>0$ but $\Delta_*<0$). The primary questions are: when reversal is real rather than noise; whether response strength separates proxy from deployment; and whether PIVOT reduces update-selection mistakes at matched high-fidelity (HF) validation cost.

## Execution baseline

The GitHub repository is the sole implementation baseline. Freeze a sprint branch/tag such as `benchmark/iclr-2027` and record implementation, candidate generation, baselines, random seeds, paired rollout rules, HF budget, stopping rule, adaptation horizon, and statistics implementation.

Every paired rollout must reuse the same initial state, random seed, external randomness, and opponent initialization for incumbent and candidate. Each formal run emits a manifest containing `experiment_id`, `git_commit`, benchmark, environment, method, seed, candidate count, HF budget, adaptation horizon, response strength, timestamps, and status. Write each completed seed immediately as an independent artifact.

## Day 1 sanity gate

Before adding benchmarks, reproduce reduced versions of the existing operator-shift and closed-loop results. Use 5–10 seeds for the check. If direction and rough scale are not consistent, stop new benchmark work and debug code, configuration, or statistical units first.

## Priority benchmarks

### 1. MPE2 with adaptive opponents

Use fixed, short, medium, and long adaptation horizons. Discovery uses 10–12 seeds per condition; after freezing the protocol, confirmation uses 30 new seeds. The pre-specified hypothesis is that increasing adaptation strength or horizon will increase proxy–deployment divergence, and that PIVOT’s relative value will increase where deployment response materially affects update selection. Null or opposite results are retained as boundary evidence.

### 2. OpenSpiel fixed opponent to best response

Use 2–3 games and a fixed candidate count (for example, 8). Compare a fixed-opponent proxy world with a deployment world where the opponent computes an exact best response where tractable; otherwise uses a pre-specified approximate best-response learner with response quality recorded. The clean target is direct improvement against the old opponent but strategic reversal after response. Use the same discovery and confirmation seed policy as MPE2.

### 3. MetaDrive replay to reactive traffic

Compare replayed, non-responsive traffic with responsive traffic that brakes, follows, or avoids based on ego behavior. Do not begin with a complex learned traffic agent. If a clean reproducible experiment is not available after one engineering day, switch to HighwayEnv.

### 4. Melting Pot adaptive population

Select only one or two substrates with clear response. Compare fixed and adaptive populations. If the adaptive population is an extension, label it explicitly as an experiment-specific adaptive extension. Allow at most two engineering days; then switch to Overcooked with an adaptive partner.

## Backups and deferred work

HighwayEnv is the MetaDrive backup. Overcooked with an adaptive partner is the Melting Pot backup. EvoPolicyGym is stretch work after the four primary benchmarks. ABIDES-MARL is deferred because it is engineering-heavy and the current finance evidence is already observational. SUMO-RL, RESCO, and SMAX/JaxMARL adaptive variants are deferred because their incremental mechanism coverage is limited.

## Unified methods and metrics

Compare Proxy Only, Uniform HF, Paired LUCB, Global-VOI, PIVOT-VOI, and All-HF. All-HF is an HF reference, not a deployable method or universal ground truth.

The primary decision metric is update-selection regret at matched HF cost: **ISR** (Improvement-Selection Regret) for single candidate-set experiments and **CISR** (Cumulative Improvement-Selection Regret) for repeated closed-loop experiments. In a one-round budget experiment, CISR reduces to ISR. Secondary metrics are **IRR** (Improvement Reversal Rate), **SIRR** (Strategic Improvement Reversal Rate), **ISC** (Improvement Sign Consistency), **IDE** (Improvement Delta Error), and **CTI** (Cumulative True Improvement).

Statistical inference is at seed, trajectory, or opponent-seed-cluster level. Transition rows are internal observations, not independent replicates. Bootstrap confidence intervals must use the corresponding independent unit.

## Discovery and confirmation

Discovery (8–12 seeds) is for debugging, variance estimation, validating that the response manipulation changes the environment, and fixing the final protocol. Primary benchmarks are not dropped based on effect direction. It is not evidence for the primary claim. Confirmation starts only after protocol freeze and uses independent seeds, normally 30 per condition. Before confirmation begins, define a precision-based extension rule to 60 seeds, for example extension only if the 30-seed confidence-interval half-width exceeds a pre-specified tolerance. Report both the original 30-seed estimate and the expanded estimate when extension occurs; do not keep adding seeds to chase significance.

## Two-week schedule

- **Day 0:** handoff, branch/tag, environment freeze, configs, metrics.
- **Day 1:** sanity reproduction and go/no-go gate.
- **Days 2–3:** MPE2 discovery and confirmation.
- **Days 3–4:** OpenSpiel strategic-response experiment.
- **Days 4–6:** MetaDrive; switch to HighwayEnv after one blocked day.
- **Days 6–8:** Melting Pot; switch to Overcooked after two blocked days.
- **Days 8–9:** unified aggregation and mechanism review.
- **Days 9–11:** concentrate compute on informative conditions; expand seeds only where justified.
- **Days 11–12:** paired/unpaired, horizon, response-strength, and HF-budget robustness.
- **Days 12–13:** freeze experiments and generate bootstrap summaries, tables, and figures.
- **Days 13–14:** integrate results, limitations, appendix, and reproducibility material into the paper.

## Core figures

1. Response strength or adaptation horizon versus IRR/SIRR.
2. The mechanism chain $\Delta_V \rightarrow \Delta_{actor} \rightarrow \Delta_{strategic}$ across benchmarks.
3. HF validation cost versus CISR for all methods.
4. A cross-benchmark forest plot of normalized within-environment effects, such as FER (Fraction of Excess Regret Removed) only when candidate count, decision horizon, and environment are matched, or standardized paired effect sizes. Raw regret remains reported separately in native units.

## Stop/go rules and success criterion

Do not change seeds, horizons, or metrics to chase significance. Do not abandon MPE2 because of a null. Limit new-benchmark infrastructure debugging to one or two days. Prefer four orthogonal benchmarks over many repeated ones. Every main result must rebuild from a fixed commit, config, and seed manifest.

Success does not require PIVOT to dominate across all benchmarks. The sprint succeeds if it establishes—or clearly bounds—the relationship between deployment response and Improvement Fidelity. The central hypothesis is that when deployment response is weak, proxy and deployment improvements remain close and transition-aware validation offers limited incremental value; as environmental or strategic response strengthens, the two can diverge, making decision-sensitive paired validation increasingly valuable. Null and negative results are retained as evidence about the boundary of the mechanism rather than treated as failed benchmarks.
