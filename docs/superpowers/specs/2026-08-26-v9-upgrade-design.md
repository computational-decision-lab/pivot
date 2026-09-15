# PIVOT V9 Upgrade Design

**Status:** Approved for implementation on 2026-08-26

**Goal:** Upgrade the frozen V7 PIVOT study into a multi-environment, operator-shift, learned-evaluator, evidence-efficiency, and strategic-adaptation study without changing the V7 evidence or its claims.

## Scientific contract

The unit of analysis remains a replacement transition `pi -> pi'`. V9 tests when global policy-value fidelity fails to imply operator-relative improvement fidelity, whether that failure varies with operator distribution shift and endogenous response, and whether paired high-fidelity queries are evidence-efficient. V9 does not assume that PIVOT must win; powered nulls are frozen as results.

Every experiment terminates in exactly one of `IMPLEMENTATION_FAILURE`, `DESIGN_INVALID`, `UNDERPOWERED`, `HYPOTHESIS_SUPPORTED`, or `HYPOTHESIS_NOT_SUPPORTED`. Confirmatory labels are determined from preregistered gates and independent seed/cluster counts, never from visual appeal.

## Architecture

V9 is an additive layer over the existing typed `Policy`, `PolicyTransition`, paired evaluator, PIVOT acquisition, and metric contracts. The new `src/pivot/v9/` package provides deterministic environment families, operator families, learned evaluator adapters, hierarchical bootstrap utilities, calibration diagnostics, and a schema-preserving artifact writer. The existing V7 directories are read-only inputs for the frozen MPE2/null and baseline audit.

The execution path is:

```text
profile/config -> seeded candidate transitions -> observer/actor/strategic worlds
-> raw transition rows -> grouped statistics and bootstrap CIs
-> scientific decision + failure ledger -> figures/tables/claim audit
-> paper snapshot and reproducible submission package
```

## Environment and operator scope

The primary V9 environment families are:

1. Frozen MPE2/external adaptive fixture, treated as a preserved null.
2. Independent performative control, where deployed action changes response stock and reward dynamics.
3. Independent congestion/resource world, where allocation changes queue load and future latency/cost.

The E2C operator families are local random mutation, gradient-informed proposals, and evolutionary/population proposals. Shift is controlled by a frozen concentration parameter and never tuned after confirmatory labels are observed.

Strategic evaluation uses fixed, reactive, finite-step best-response, gradient-adaptive, and RL-like/evolutionary adaptive opponent mechanisms. The focal agent is the only self-improving agent; opponent seeds and adaptation strengths are held out.

## Statistical design

The primary confirmatory profile uses 30 independent seeds per cell where computationally feasible, `T=40`, `K=8`, and explicit hierarchical bootstrap over environment/operator, seed, trajectory, and transition groups. V9 reports raw independent counts, transition counts, means, medians, standard deviations, 95% confidence intervals, and effect sizes. E2C uses 10 fixed shift levels. E4C uses trajectory, environment, operator, and response-regime holdouts; no row-wise random split is permitted.

The primary hypothesis family is H1--H5 from the V9 protocol, with Holm correction where formal tests are used. Effect estimates and uncertainty are primary; p-values are secondary.

## Artifact and claim boundaries

Each V9 run stores a config snapshot, code/dependency provenance, deterministic seed list, raw transition records, processed source tables, metrics, scientific decision, failure ledger, and SHA-256 manifest. Large transition tables use Parquet when available and gzipped JSONL as the portable fallback. Every plotted number is generated from an artifact, and every result claim is registered with an allowed and forbidden scope.

The V7 paper, figures, results, PDF, supplementary ZIP, and finance negative boundary are copied or hash-referenced under `snapshot/v9_preupgrade/` before V9 changes. V9 may strengthen or narrow claims, but may not rewrite historical outcomes. If E3C/E4C/E5C are null, the paper states the null explicitly and distinguishes estimand-level motivation from universal method superiority.

## Completion gates

V9 is complete only after P0--P10 have evidence in the repository: at least three E2C environments and operator families (or a documented construct-valid reason for a smaller scope), the independent E3C performative and congestion worlds, learned evaluator OOD/calibration diagnostics, matched-budget E5C methods including LUCB and Global-VOI, multiple strategic opponent mechanisms, publication-quality vector figures with metadata/source tables, automated statistical/claim/reproducibility audits, and a rebuilt anonymous PDF within nine main-text pages.
