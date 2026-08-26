# V10 Metric and Scale Audit

Status: **PASS**

| Metric | Definition | Aggregation | Unit |
| --- | --- | --- | --- |
| IDE | mean absolute difference |Delta_V - Delta_*| | transition rows within the declared operator law | native environment reward |
| ISC | sign agreement after excluding proxy/true ties at tau | comparable transition rows | probability |
| IRR | P(Delta_* < -tau | Delta_V > tau) | proxy-positive transition rows | conditional probability |
| ISR | max_j Delta_*j - Delta_*selected | one incumbent/candidate decision set | native environment reward |
| CTI | sum_t Delta_*selected,t | trajectory in the closed loop; one selected delta when T=1 | native environment reward |
| CISR | sum_t ISR_t | trajectory for repeated improvement; equals ISR in the one-set budget study | native environment reward |
| FER | 1 - (CISR_method-CISR_allHF)/(CISR_proxy-CISR_allHF) | only matched environment, K, horizon, and cost protocol | dimensionless fraction of excess regret removed |

## Scale boundary

The Proxy Only cumulative regret means are `0.025273021055140617` and `4.263059574924713` in two different native reward systems (ratio `168.7` where available). Their raw magnitudes are not compared across environments.

Closed-loop CTI/CISR sum 40 replacement decisions. The efficiency frontier uses one candidate set, so its stored CISR is one-set ISR. FER is computed only within a matched cell and is withheld when its oracle denominator is not meaningful.

## Inference units

- `operator_shift`: independent seed; cell summaries preserve transition counts
- `closed_loop`: trajectory seed
- `ood`: registered held-out unit and fitted family; descriptive split spans are not pseudo-CIs
- `efficiency`: paired environment x K x seed decision set
- `strategic`: opponent-seed cluster
- `finance`: session-level observational diagnostic

## Code trace

- `IDE_ISC_IRR_ISR`: `src/pivot/v9/statistics.py:58`
- `closed_loop_CTI_CISR`: `experiments/v9/e3c_closed_loop.py:249`
- `budget_set_CTI_CISR`: `experiments/v9/e5c_efficiency.py:67`
