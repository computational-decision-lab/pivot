# Algorithm 1 and v3c call-chain audit

Date: 2026-11-14

This is a source-level audit. It checks the implementation anchors and does not
count source assertions as simulator evidence.

## Finding

`PIVOT-KG-Loop-v3c` executes the single-round logic of the submitted
Algorithm 1 inside each round: proposer, proxy rows, calibrated posterior,
method-specific acquisition, paired HF observation, noisy Gaussian
conditioning, terminal posterior-mean selection, independent evaluation,
deployment, state transition and incumbent update are all present.

The five-round protocol remains a controlled extension. It resets the
contextual correction model at each round for the primary confirmation, has no
common sentinel, and permits later candidate panels to diverge after method
specific incumbent updates. Therefore the audit supports the wording “faithful
single-round logic inside a controlled five-round extension”; it does not
support “confirmation of static Algorithm 1 in arbitrary self-improvement
systems.”

## Source anchors

| Step | Paper Algorithm 1 | v3c runner | Classification |
| --- | ---: | ---: | --- |
| Candidate/proxy evaluation | 453 | 108-112 | Exact single-round match |
| Calibrated posterior | 454 | 113-122 | Exact single-round match |
| Acquisition | 456-460 | 125-129 | Exact logic; method-specific wrapper |
| Paired HF observation | 461 | 131-135 | Exact logic; controlled seed schedule |
| Gaussian conditioning | 462 | 154-158 | Exact logic with positive observation variance |
| Terminal selection | 465 | 172 | Exact logic |
| Independent evaluation | implicit truth evaluation | 173-180 | Controlled extension ledger |
| Deployment/state transition | outside single-round algorithm | 183-229 | Five-round extension |
| Next-round incumbent | outside single-round algorithm | 226-229 | Five-round extension |

The executable checker is
`docs/rebuttal/audit/audit_algorithm1_v3c_call_chain.py`. It verifies the
source anchors and records the controlled-extension boundaries in JSON.
