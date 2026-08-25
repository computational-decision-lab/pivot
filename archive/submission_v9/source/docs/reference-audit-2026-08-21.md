# Reference Audit

Date: 2026-08-21 (Asia/Shanghai)

The ICLR bibliography was checked against primary publisher or repository
metadata before the final paper freeze. All 46 arXiv identifiers cited in
`paper/iclr2027/references.bib` resolved through the official arXiv API. The
recent 2025--2026 entries use the title and author metadata returned by those
records. The M3 entry is checked against the authors and title on the
project's paper PDF because that work does not expose an arXiv identifier in
the current release.

The audit found and corrected two stale identifiers: the bibliography entry for
`A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning` now
points to `https://arxiv.org/abs/1711.00832` (the previous `1706.05371` URL was
for an unrelated paper), and `Price Dynamics in a Markovian Limit Order
Market` now points to `https://arxiv.org/abs/1104.4596` (the previous
`1002.4316` URL was for an unrelated paper). The final bibliography contains 50 entries, including
verified ICLR 2024 entries (AgentBench, WebArena, SWE-bench, and TD-MPC2), the
NeurIPS 2022 FinRL-Meta benchmark, and recent 2025--2026 work on performative
learning, self-improvement, world-model fidelity, and market simulation.

## Verification scope

- Official arXiv API: every arXiv identifier in the bibliography (46/46).
- Primary M3 paper PDF: title and five listed authors.
- DOI/publisher URLs in the remaining entries were retained as primary
  references; BibTeX compilation and citation resolution pass in the final
  LaTeX build.
- No invented venue or paper title is used for the new entries. Preprints are
  labelled as arXiv preprints unless a verified conference venue is available.
