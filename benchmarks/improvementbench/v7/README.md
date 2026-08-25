# ImprovementBench V7

This release contains 380,960 transition-level records from the final valid
E3b, E4b, and E7b outputs. The complete JSONL is stored as
`transitions.jsonl.gz` so the public package stays below repository file-size
limits. Restore it with `gzip -dk transitions.jsonl.gz`; the uncompressed
SHA-256 is represented by the source-run manifest and the compressed member is
hash-bound by this release manifest.

Splits are assigned by connected components of trajectory, environment,
operator, opponent-family, and response-regime groups; no row is reused across
a different split. E3b and E4b rows may contain explicit nulls for unavailable
world layers, preserving the evaluation contract rather than imputing them.
