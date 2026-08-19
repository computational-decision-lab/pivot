# Experiment Protocol

## Mandatory Order

1. E1: Does Improvement Reversal exist?
2. E2: Response strength x update footprint.
3. E3: Performative overoptimization.
4. E4: Global Fidelity versus Improvement Fidelity.
5. E5: PIVOT budget frontier.
6. E6: Financial mechanical reversal.
7. E7: Strategic Improvement Reversal.
8. E8: Competition strength.
9. E9: Closed-loop self-improvement.

P0-P10 in the master implementation plan controls when each experiment may run.

## Statistical Rules

- Use multiple independent seeds.
- Preserve transition-level raw data, failed transitions, discarded runs, and sample counts.
- Register exclusions before inspecting final outcomes.
- Use paired bootstrap or analytically justified paired confidence intervals when evaluation is paired.
- Separate sampling noise from systematic response effects.
- Never report reversal from one trajectory.
- Hold HF budgets equal across selection baselines.
- Freeze candidate batches when comparing acquisition policies.

## Baselines

```text
B1 Proxy Only
B2 Random HF
B3 Top Proxy
B4 Largest Footprint
B5 Global Value Model
B6 Global Ranking Model
B7 Uncertainty Sampling
B8 All-HF Oracle
B9 PIVOT
```

## Required Ablations

Paired/unpaired, transition/global-value, footprint/no-footprint, active/random, PIVOT/Top Proxy, small/large updates, weak/strong response, F1/F2, fixed/adaptive competitors, single/multiple response models, candidate count, and HF budget.

## Finance Causal Test

Hold the transition fixed and vary only participation:

```text
rho = agent trading volume / market volume
```

Evaluate the same update in F0, F1, F2, and F4. Do not tune the response model to force a sign crossing.
