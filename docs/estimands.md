# Estimands and Metrics

## World-Specific Deltas

```text
Delta_proxy     = J_V(pi') - J_V(pi)
Delta_actor     = J(pi'; M[pi']) - J(pi; M[pi])
Delta_strategic = J_i(pi_i', BR_-i(pi_i')) - J_i(pi_i, BR_-i(pi_i))
```

When identifiable:

```text
Delta_direct       = J(pi'; M) - J(pi; M)
mechanical_effect  = Delta_actor - Delta_direct
competition_effect = Delta_strategic - Delta_actor
```

Unavailable counterfactual components remain `null`.

## Primary Metrics

```text
IDE  = E[abs(Delta_proxy - Delta_true)]
ISC  = P[sign(Delta_proxy) = sign(Delta_true)]
IRR  = P[Delta_true < 0 | Delta_proxy > 0]
SIRR = P[Delta_strategic < 0 | Delta_actor > 0]
MTR  = Delta_true / Delta_proxy when abs(Delta_proxy) > tau_mtr
ISR  = max_j Delta_true_j - Delta_true_selected
CTI  = sum_t Delta_true_selected_t
```

`tau_sign` determines ties. Ties are counted and reported rather than forced into a sign. MTR is missing when its denominator is unstable.

## Cost Estimand

Every method reports:

- HF transitions queried;
- HF paired rollouts;
- environment steps;
- simulator calls;
- compute cost where meaningful.

PIVOT is compared only at equal or explicitly cost-normalized HF budgets.
