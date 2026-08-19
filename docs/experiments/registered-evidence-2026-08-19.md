# Registered Evidence Record

This is a controlled-fixture evidence record generated after the registered
runner and independent aggregation were implemented. It is not a claim that
the effects transfer to real finance data or to arbitrary adaptive worlds.

## Provenance

- Output root: `/tmp/pivot-registered.W4OJHm`
- Registry runs: 3 per experiment, disjoint seed sets
- Registry SHA-256:

```text
p2  c385f77bb12380ded7d481193964b3da803c43ceebc45aa0bf7f53bd4e239667
e4  0d624e2e0197d9f17f39adc2a74e4d04c64f88c08dcf60320f8d20a59398a5d5
e5  0cda2869c436513fd707f18c82ca44708af8c14b4620ab4b07b1b14771e41bfe
e6  832133d6f5d936159da64da550d920da348a1f17127d2bca864f0c58e15e9055
e7  cd269d6ec21ad869dccc5c9b7c15d2de748889309646986c4b5f17c5f423b807
e8  f2432c1c8c7f29b1ea8e7eb6e743d91cd2c6105796039247cd28820907150f29
e9  b5f47987f3431300d5e7db190362f903ab04c9eb9cc351995c3c6c863b3764d4
```

Run IDs are `p2-r01..r03`, `e4-r01..r03`, `e5-r01..r03`,
`e6-r01..r03`, `e7-r01..r03`, `e8-r01..r03`, and `e9-r01..r03`.

## Fixture-level gate output

| Gate | Result | Registered metric |
| --- | --- | --- |
| A | Fixture Pass | high-response IRR `0.9722`, 95% CI `[0.9722, 0.9722]` |
| B | Fixture Pass | high-minus-low response IRR `0.9722`, 95% CI `[0.9722, 0.9722]` |
| C | Fixture Pass | local-minus-global ISC `0.1204`, 95% CI `[0.1204, 0.1204]`; ISR difference tied |
| D | Fixture Pass | PIVOT ISR gain over Random `[4.4401, 4.8694]`; over Top Proxy `[3.1282, 3.1322]` |
| E | Fixture Pass | zero-participation F2-F1 `0`; target-participation effect `-0.0041473` |
| F | Fixture Pass | E7 SIRR `1.0`; E8 competition effect `-0.04052`; sensitivity contrast `-0.10822` |

Gate F intentionally does not require E8 SIRR: in this registered fixture the
actor delta is already negative at the E8 participation level, so the SIRR
denominator is empty. The predeclared E8 criterion is instead a negative
competition effect with a negative high-minus-low strategic-sensitivity
contrast.

E9 produced three isolated closed-loop artifacts; it is summarized for
reproducibility and is not used to promote a new gate.

## Interpretation boundary

The current records establish that the implementation can execute and
aggregate the planned estimands under frozen synthetic fixtures. They do not
establish external validity, realistic market calibration, or a universal
PIVOT superiority claim. Those remain explicit follow-up requirements.
