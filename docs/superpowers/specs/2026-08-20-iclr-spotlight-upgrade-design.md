# ICLR Spotlight-Level Narrative Upgrade Design

> **Status:** approved by the supplied Codex Goal v3 specification
>
> **Goal:** Reframe the anonymous ICLR 2027 paper around transition fidelity as the correct evaluation object for self-improvement, while adding a decision-preservation proposition, a distinct value-versus-transition experiment, and stronger visual/claim discipline without changing the title or central thesis.

## Locked identity

The paper title remains *When Better Gets Worse: Improvement Fidelity of Self-Improvement Operators in Adaptive Worlds*. The central phenomenon remains Improvement Reversal, the statistical object remains the policy transition \(\tau=(\pi_t,\pi_{t+1})\), and PIVOT remains the first practical implementation rather than the definition of the problem.

## Narrative architecture

The introduction opens with the replacement operation optimized by a self-improvement loop and states that a verifier may rank policies correctly while ranking improvements incorrectly. The abstract follows four paragraphs: hidden transfer assumption, Improvement Fidelity estimand, decision-preserving PIVOT, and bounded evidence/limitations. Contributions are ordered as failure mode, evaluation criterion, method, and evidence.

## Theory addition

After the existing global-fidelity and response-footprint propositions, add a cautious proposition named “Decision Preservation Under Differential Error.” For a candidate pair with true margin \(m=\Delta_{*,i}-\Delta_{*,j}>0\), if each differential estimate has absolute error less than \(m/2\), their ordering is preserved. The proof is a two-line triangle-inequality argument. It motivates PIVOT directly without claiming universal optimality.

## Method distinction

Add a subsection explicitly contrasting global active learning with PIVOT. Active learning targets global prediction error for \(y\); PIVOT targets the probability of changing the selected update by estimating \(\Delta_* - \Delta_V\). State that PIVOT minimizes neither global world-model error nor an unconditional value loss.

## New controlled evidence

Extend E4 with a reproducible “Value Fidelity vs Improvement Fidelity” comparison. Construct two deterministic evaluator surrogates from the held-out transition rows: evaluator A preserves policy-value accuracy while injecting a transition-correlated differential bias; evaluator B has a small policy-value offset/noise while preserving the differential correction. Report policy-value MAE, rank correlation, IDE, ISC, IRR, and ISR for both, with the transition rows and construction parameters frozen in the snapshot. The experiment may expose a null; the paper must report it honestly.

## Figure changes

Keep Figure 1 but label correct improvement, false improvement, and improvement reversal regions. Render Figure 2 as a continuous response-by-footprint heatmap with a zero-reversal contour and colorbar. Add a central Figure 4 panel/table for the new evaluator contrast while retaining the current figure contract and appendix diagnostics.

## Finance and claims

Rename the section to “Stress Tests Beyond Controlled Environments.” Describe finance as observational, non-causal boundary validation, retain 0/7 primary and 0/5 holdout reversal nulls, and do not imply real-market reversal. Preserve all limitations, deferred LLM/EvoQuant/M3 adapters, and open external scientific gates.

## Verification contract

The upgraded source must keep the exact title, remain anonymous, compile to at most nine main pages, have no undefined references or overfull boxes, include the new proposition and explicit PIVOT distinction, regenerate the new experiment/figure artifacts from scripts, pass the full project tests plus Ruff and mypy, rebuild the supplementary archive, and pass the ICLR package audit. Only after those checks may the release be committed and pushed.
