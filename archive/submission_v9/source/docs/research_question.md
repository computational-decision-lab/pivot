# Research Question

## Primary Question

When does a policy update that appears better under a cheap or fixed verifier remain better in the adaptive world created by deploying that update?

The statistical object is:

```text
pi_t -> pi'_{t,j}
```

The first failure event is Improvement Reversal:

```text
Delta_proxy > 0 and Delta_true < 0
```

The stronger failure event is Strategic Improvement Reversal:

```text
Delta_actor > 0 and Delta_strategic < 0
```

## Falsifiable Questions

1. Does reversal occur in non-pathological controlled settings?
2. Does IRR vary systematically with update footprint and environment response?
3. Does a policy-level value/ranking evaluator fully explain local transition quality at an equal HF budget?
4. Does paired transition-level modeling reduce local improvement error?
5. Does PIVOT reduce true selection regret per unit HF budget versus Random HF and Top Proxy?
6. Does finance show a structured F0/F1 versus F2 difference at plausible participation?
7. Does strategic adaptation add a systematic effect beyond mechanical response?

The project must accept a null answer to any question. Its first objective is to determine whether Improvement Fidelity is a distinct measurable problem, not to prove PIVOT works.
