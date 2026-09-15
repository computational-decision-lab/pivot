# Resource Plan

This is the locked resource plan.  The registered primary design is two
operator families, two task families, 30 independent trajectories per major
unit, `30` rounds, and `4` candidates per round.
The pinned external runtime has completed bounded DEV smoke runs.  Confirmatory
execution remains unopened, so DEV counts below are validation evidence only.

Local adapter probes:
- Inspect AI: `available` (0.1.dev1+g9fa31ccbc)
- mini-SWE-agent: `available` (2.4.6)
- Pi: `available` (ccfe79ed238674f760c986e3a61493aab794000a)

DEV manifests: transition `COMPLETED`, promotion
`COMPLETED`, closed loop
`COMPLETED`, Pi
`COMPLETED`, response
`PARTIAL`, ablations
`COMPLETED`.  These runs recorded no
assessment access outside the terminal DEV assessor and no outcome chasing.
Reducing scope before confirmatory data is allowed; changing scope after
outcomes to obtain a preferred result is forbidden.

## Accounting

| Resource | Pre-outcome estimate | Observed before confirmatory run |
|---|---|---|
| LLM calls | 6480 | 0 |
| Inspect evaluations | 14400 | 0 |
| Container executions | 28800 | 0 |
| Task instances | 12 | not opened |
| T x K | 30 x 4 | locked |
| CPU / GPU | 1 vCPU per container / none required by protocol; model provider may impose external quota | not run |
| Token budget | 2048 | not run |
| Storage | estimate after trace-size calibration | not run |
| Wall clock | estimate after adapter dry-run and provider latency calibration | not run |
| Estimated cost | estimate_after_external_smoke | 0 confirmed calls |
