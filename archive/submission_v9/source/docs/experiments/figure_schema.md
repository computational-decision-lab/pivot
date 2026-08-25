# Figure Artifact Schema

The paper figures are generated from source tables, never from hand-edited
values. Each available figure has `<stem>.png` and `<stem>.csv`; the validation
script checks that the PNG has a valid signature and that its source table is
present. A deferred figure has `<stem>.unavailable` with a reason.

Canonical stems:

1. `fig1_when_better_gets_worse`
2. `fig2_reversal_phase_diagram`
3. `fig3_optimizing_wrong_world`
4. `fig4_policy_vs_improvement_fidelity`
5. `fig5_pivot_budget_frontier`
6. `fig6_observer_actor_strategic` (E7 source after P8)
7. `fig7_strategic_reversal` (E8 source after P8)

Before P8, Figure 6/7 remain explicitly unavailable. Once E7/E8 artifacts are
present, the same script generates both figures and their source tables.
