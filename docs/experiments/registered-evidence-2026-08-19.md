# Registered Evidence Record

This is a controlled-fixture evidence record generated after the registered
runner and independent aggregation were implemented. It is not a claim that
the effects transfer to real finance data or to arbitrary adaptive worlds.

## Provenance

- Verified commit: `e51ac38d0d6654c16acb789868ce209afeac149a`
- Output root: `/tmp/pivot-registered-final.MPuoXC`
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

Aggregated summary SHA-256:

```text
p2  663e18c3a2f9256a31bdff50e3e93018991ec0a13733d7466d9591f5c01c02da
e4  87d5353ac9ad1fa9085f7975acf394c5cc8a757c83d3af1c75985f6aa9e03320
e5  cb754d0917e6ac42f052aaa4cec2ec89b94eaf0e022a88e6f51cb807edcc4ea9
e6  3278f3b6446389e87666d32cfe4943ead4943e622b3d9c20c4666096d4b81e59
f   35ca414464d28bce201fc0a54acaaea16e9b3eb3cdeafc67050133bedd830edf
e9  37614f59a0cf5c03bd967eec1e1ce41517f41ba9197c4a6395daf60d36ec8c99
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
