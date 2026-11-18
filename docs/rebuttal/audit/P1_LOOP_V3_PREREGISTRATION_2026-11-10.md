# P1 multi-round loop v3 preregistration

Date: 2026-11-10  
Status: development protocol; confirmation roots remain untouched until development, calibration-noise audit, and code review are complete.

## Motivation

V2's confirmation showed that contextual calibration alone was better than carrying its online regression update forward, while the online variant did not establish superiority over both targeted baselines. Inspection found that v2 used cross-validated correction-model RMSE as the variance of one HF observation. Those quantities are different: the former includes model misspecification, while the latter is rollout sampling noise. V3 separates them and estimates single-rollout observation noise using repeated paired HF rollouts on calibration contexts.

V3 is a new method variant, `PIVOT-KG-Loop-v3`. It does not replace the v1 or v2 results. No outcome is assumed; all response regimes and negative findings will be reported.

## Fixed protocol

- Five deployment rounds, four candidates per round, three response regimes: low `0.35`, medium `0.60`, and high `0.85`.
- Calibration uses 240 roots, IDs `510000` through `510239`, stratified over pressure `[0.05, 0.70]`, with response strengths balanced across roots and root-seeded incumbent parameters.
- For each calibration root and candidate slot, eight independent paired HF rollout deltas are collected. Their mean is the calibration target; their sample variance estimates single-rollout observation noise. The rollout seeds, values, means, and variances are saved and audited.
- Development uses 50 roots per regime, IDs beginning at `800000` with disjoint 10,000-ID regime blocks.
- Confirmation uses 120 roots per regime, IDs beginning at `900000` with the same disjoint regime blocks. Confirmation roots are not used for model or protocol selection.
- Test initial pressure is `0.10 + 0.015 * (root_offset mod 7)`; deployment carries the resulting final state into the next round.
- The three primary online methods receive two paired HF queries per round: one common-rule sentinel at `candidate_index = round_id mod 4`, followed by one method-specific query. The sentinel is included in, not added to, the budget.
- PIVOT-KG-Loop-v3, IVR-Online, and Top-Proxy-Online share the same contextual model, noise estimates, sentinel rule, terminal selector, total budget, root-level initial conditions, and exogenous random streams. Their second query uses exact one-step Gaussian KG, IVR, or the largest remaining proxy delta, respectively.
- `Contextual-KG` conditions within-round but resets the regression at the next round. `Static-PIVOT-KG-v1` uses the original narrow-state prior and is a legacy reference, not part of the matched contextual-model comparison. Every method still receives two queries per round.
- Evaluation streams are shared by root, round, and candidate index; deployment streams by root and round; query streams by root, round, and candidate ID.
- Primary outcome is per-root cumulative five-round ISR. Secondary outcomes are cumulative selected deployment utility, harmful-adoption rate, and incumbent-retention rate.
- The six primary contrasts are IVR-Online minus PIVOT-KG-Loop-v3 ISR and Top-Proxy-Online minus PIVOT-KG-Loop-v3 ISR in each of the three regimes. Report paired-root bootstrap 95% intervals and sign-flip tests with Holm correction across all six tests.
- A superiority claim within a regime requires both paired ISR interval lower bounds to exceed zero and both Holm-adjusted p-values to be below `0.05`. Utility is reported alongside ISR; no utility non-inferiority claim is planned.

## Variance model

The target is the proxy-to-HF correction `g = HF delta - proxy delta`. The contextual Bayesian linear model uses pressure, response strength, incumbent parameters, candidate parameter changes, proxy delta, pressure/action and response/action interactions, and normalized round.

Observation variance is estimated separately for each response-strength/candidate-slot cell from within-context variance over the eight calibration rollouts. Calibration targets average those eight rollouts, so their label-noise variance is the measured single-rollout variance divided by eight. Five-fold root-level out-of-fold MSE estimates total prediction error; the model-residual component is that MSE minus the label-noise variance, floored at `1e-6`. Bayesian coefficient covariance is scaled by this model-residual component. Online updates use the sum of coefficient, model-residual, and single-rollout observation variance.

Ridge is fixed at `0.5`; no hyperparameter is selected using confirmation results. Standardized residuals are logged for diagnosis without a test-tuned trigger threshold.

## Execution order

1. Run and audit the development split, repeated-rollout noise estimates, shared seeds, query budgets, and posterior replay.
2. Record any code or protocol amendment before confirmation. Development results remain exploratory.
3. Freeze runner, analyzer, calibration-source, and protocol hashes, then run the untouched confirmation split once.
4. Report all regimes, all baselines, uncertainty, utility trade-offs, and negative findings.

Runner: `run_multiround_loop_v3.py`  
Analyzer: `analyze_multiround_loop_v3.py`
