# PIVOT ICLR 2027 Curated Release

This directory contains the anonymous manuscript and the deterministic
supplementary archive and the 2026-09-18 sealed experiment update.
The release preserves positive, null, and negative response-world results
and the decision-relevance bridge derived from the sealed roots.

## Files

- `paper.pdf`: anonymous manuscript.
- `supplementary.zip`: sanitized source, figures, tables, and audit artifacts.
- `revision_evidence_audit.json`: reviewer-safe hashes, seals, and recomputation checks.
- `submission_verification.json`: compact machine audit summary.
- `SHA256SUMS`: hashes for every release file except the checksum file.

No raw Inspect traces, sandbox trees, candidate archives, private paths,
credentials, or local runtime installations are included. Rebuild the
paper, supplement, and audit with `make v15-finalize` from the repository root.
