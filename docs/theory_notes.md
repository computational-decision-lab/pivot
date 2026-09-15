# Theory Notes

## T1: Absolute Fidelity Is Sufficient

If:

```text
sup_pi abs(J_V(pi) - J_*(pi)) <= epsilon
```

then:

```text
abs(Delta_V - Delta_*) <= 2 epsilon
```

The converse need not hold. A constant policy-independent offset may make absolute values inaccurate while preserving every update delta.

## T2: Response Sensitivity x Update Footprint

Target assumptions and bound:

```text
D(M[pi'], M[pi]) <= L_M d(pi', pi)
abs(Delta_actor - Delta_direct) <= C L_M d(pi', pi)
```

`C` must be derived from the chosen environment and objective. The implementation measures `d`, response strength, and improvement error, but no unproven bound may be hard-coded or claimed.

## T3: Strategic Sensitivity

```text
S_-i = D(BR_-i(pi_i'), BR_-i(pi_i)) / (d(pi_i', pi_i) + epsilon)
```

E8 tests whether higher `S_-i` is associated with SIRR and whether small focal updates can cause disproportionately large competitor responses. This is an extension, not a prerequisite for the core estimand.

## Proof/Empirical Separation

- Propositions require formal assumptions and proofs.
- Controlled experiments test consequences, not prove theorems.
- Finance and strategic results test external relevance, not universal validity.
- A violated Lipschitz assumption is a boundary result, not an implementation failure to hide.
