# Registered Evidence Record

> Amendment: commit `4c1677c` corrected F2 from repeated holding impact to
> fill-only execution impact and reran E6--E9. The table below uses the
> corrected E/F values. P2, E4, and E5 remain unaffected. Full amendment and
> public-data evidence are in `public-finance-evidence-2026-08-19.md`.

This is a controlled-fixture evidence record generated after the registered
runner and independent aggregation were implemented. It is not a claim that
the effects transfer to real finance data or to arbitrary adaptive worlds.

## Provenance

- P2/E4/E5 verified commit: `e51ac38d0d6654c16acb789868ce209afeac149a`
- P2/E4/E5 output root: `/tmp/pivot-registered-final.MPuoXC`
- Corrected E6-E9 verified commit: `4c1677cf71ed88825e902eaff5631da3427ab83c`
- Corrected E6-E9 output root: `/tmp/pivot-evidence-4c1677c.qDSzix`
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
e6  e103a8f4c01bd1d10a83094cd5a97dfb38712e0d022067f0341da51f694c5a4d
f   20dad33cc8ef122f407470601847b1febe9bb74e53c8babae7dc8b8acb36c7be
e9  4cb3a8ab32a6e3a65c827fac5c495d36e905c329e7b8c1cedb94864ffdeea4d7
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
| E | Fixture Pass | zero-participation F2-F1 `0`; 5 percent effect `-0.0002303999863` |
| F | Fixture Pass | E7/E8 SIRR `1.0`; E8 actor delta `0.00170453706`; competition effect `-0.04052`; sensitivity contrast `-0.10822` |

The old repeated-holding-impact E6/F summaries from commit `e51ac38` are
superseded and must not be used for the current code or a paper claim.

E9 produced three isolated closed-loop artifacts; it is summarized for
reproducibility and is not used to promote a new gate.

## Interpretation boundary

The current records establish that the implementation can execute and
aggregate the planned estimands under frozen synthetic fixtures. They do not
establish external validity, realistic market calibration, or a universal
PIVOT superiority claim. The separate public audit improves execution
plausibility but does not identify causal market response. Those remain
explicit follow-up requirements.
