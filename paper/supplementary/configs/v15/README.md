# V15 Protocol Inputs

The versioned YAML, runtime, and public provenance files in this directory
define the V15 execution contract. The full `task_manifest.json` is a sealed
local input and is intentionally ignored by Git; it contains task files and
instructions that must not be exposed to the operator, PIVOT, or the final
assessment package.

The anonymous supplement and `release/v15/` include a redacted manifest with
task IDs, family labels, and source hashes. A confirmatory run requires the
authorized sealed manifest whose SHA-256 is recorded in
`experiments/v15/confirmatory_lock.json`.
