# V10 Reviewer Attack Audit

Status: **PASS**

## Attack 1: Is this merely another policy evaluation metric?

**Paper location:** Problem Formulation and Contributions 1--2

**Supported answer:** No. Policy evaluation estimates J(pi); Improvement Fidelity evaluates the sign/order of a replacement under the operator-induced transition law, and PIVOT allocates paired interventions for selection regret.

**Scope limitation:** The paper does not claim policy-value models are unnecessary; global fidelity is a sufficient but stronger condition.

## Attack 2: Is Global Fidelity Blindness only an adversarial construction?

**Paper location:** Proposition 2 and operator-shift experiment

**Supported answer:** The proposition is a sharp non-implication. The empirical shift experiment separately shows structured local deterioration while global rank remains competitive.

**Scope limitation:** The construction proves possibility, not prevalence; prevalence is limited to tested operator/world families.

## Attack 3: Why not just use a global value model?

**Paper location:** Value Fidelity versus Improvement Fidelity; OOD null

**Supported answer:** A global model is a strong baseline and sometimes wins. The target can still be misaligned because the self-improver visits local directed transitions rather than the global policy population.

**Scope limitation:** The registered differential learner underperforms globally in the tested OOD splits; the paper retains that powered null.

## Attack 4: Why not just use LUCB?

**Paper location:** PIVOT VOI and evidence-efficiency figure

**Supported answer:** LUCB targets confidence around a best arm; PIVOT-VOI uses paired differential posterior regret, footprint/context, and heterogeneous evaluation cost. Paired LUCB is explicitly compared.

**Scope limitation:** PIVOT is not uniformly favorable against LUCB in the frozen results.

## Attack 5: Why does PIVOT-VOI not always win?

**Paper location:** Closed-loop outcomes and Stress Tests

**Supported answer:** Acquisition quality depends on posterior fit, candidate gaps, response structure, and cost. The method is designed for decision-sensitive evidence allocation, not universal dominance.

**Scope limitation:** Claims are restricted to Proxy Only contrasts and registered cells with supported paired effects.

## Attack 6: Are the adaptive environments hand-designed to produce reversal?

**Paper location:** Experiments design and Figure 1

**Supported answer:** The worlds are transparent controlled mechanisms with preregistered response/shift sweeps and independent seeds. They make the phenomenon falsifiable and expose where the sign changes.

**Scope limitation:** They are not presented as natural-world prevalence or market realism; the external reference null is preserved.

## Attack 7: Are Figure 4 conditions comparable?

**Paper location:** Figure 4 caption and metadata

**Supported answer:** Top panels fix K=8 within each environment and plot matched query cost. The lower panel reports paired fixed-budget effects; heterogeneous cells are not connected.

**Scope limitation:** The all-HF line is an oracle reference, not a method trajectory, and environments retain separate axes.

## Attack 8: Why do CISR scales differ?

**Paper location:** Metrics and Scale; metric-scale audit

**Supported answer:** CISR is in native reward units. Environment dynamics and reward scales differ; closed-loop CISR sums rounds while the budget study has T=1.

**Scope limitation:** Raw magnitudes are never compared across environments; only within-environment paired effects or stable matched-cell FER are allowed.

## Attack 9: Is strategic reversal hard-coded?

**Paper location:** Strategic response and Figures A/D

**Supported answer:** The outcome is measured across independent opponent-seed clusters and five mechanisms. Fixed opponents show a near-zero effect, while adaptive families show negative effects with cluster intervals.

**Scope limitation:** Mechanisms are finite controlled adaptations, not proof of equilibrium behavior or real market ecology.

## Attack 10: What does the negative OOD result imply?

**Paper location:** OOD evaluator contrast and Figure B

**Supported answer:** It rejects the claim that the registered transition learner dominates the global evaluator. It does not reject the transition-level estimand or paired decision target.

**Scope limitation:** No claim is made for untested model classes, data regimes, or domains.

## Attack 11: How is this different from prior Self-Improvement Reversal?

**Paper location:** Related Work: Self-improvement and verification

**Supported answer:** Prior post-training reversal concerns benchmark/capability regression. This paper defines a directed policy-update estimand under policy-induced world response and validates it with paired interventions.

**Scope limitation:** The paper does not claim the phrase 'reversal' itself; novelty is the estimand, response decomposition, and budgeted validator.

## Attack 12: How does AI4AI-Bench relate to this paper?

**Paper location:** Related Work: Self-improvement and verification

**Supported answer:** AI4AI-Bench evaluates whether agents redesign training algorithms under a hidden evaluator. This paper asks whether a proposed replacement remains beneficial after endogenous/strategic response.

**Scope limitation:** PIVOT is not evaluated on AI4AI-Bench, so the relation is conceptual and complementary.

## Attack 13: Is finance causal evidence?

**Paper location:** Finance audit and boundary; Figure E

**Supported answer:** No. It is an observational historical-path/virtual-fill/depth-proxy stress test and reports 0/7 primary and 0/5 holdout causal reversals.

**Scope limitation:** Causal market impact, replenishment, strategic response, profitability, and live trading are explicitly not identified.
