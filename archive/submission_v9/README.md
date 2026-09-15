# PIVOT V9 Baseline Archive

This directory is an immutable, hash-addressed snapshot of the V9 submission
surface before the V7 upgrade. The source tree was produced from git commit
`7428d04a1cf63c08cb6ca231eb05b5ee2880ba8c` and includes the tracked manuscript,
figures, configurations, registered theory artifact, code, tests, and the
ignored local V9 result/build files that were present at archive time.

The V9 baseline intentionally preserves these scientific boundaries:

- E3 is a null fixture because both proxy and true values saturate.
- PIVOT-H is a transparent heuristic, not a Bayesian VOI method.
- Strategic reversal is fixture-level evidence only.
- The finance boundary test remains observational and negative (0/7 primary,
  0/5 holdout reversals).
- The submission was `CONDITIONAL GO`; external interactive, strategic, and
  confirmatory holdout gates were open.

`manifest.sha256` hashes every file under `source/`. Do not edit this archive;
new work belongs in the live tree and must be compared against this snapshot.
