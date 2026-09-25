# ICLR 2027 reproducibility release v1

The release contains a full research-source ZIP, an anonymous supplementary
ZIP, package checksums, and a validation report. Large generated ZIPs are
published as release assets rather than committed into Git history.

Build with `python scripts/build_submission_release.py --output OUTPUT_DIR`
from any working directory, using the script's path in this checkout.
The builder rejects unsafe paths, excludes operational material, produces
deterministic archives, and writes file-by-file hashes inside each ZIP.

The current manuscript is a read-only snapshot in `reproduction/manuscript`.
No release command writes to the Overleaf-managed `paper/` directory.

Acceptance checks cover fresh-environment installation, frozen-file hashes,
saved-result recomputation, core algorithms, figure output, and bounded native
smokes. `validation.json` distinguishes checks actually passed from full study
runs that were not repeated. Historical missing inputs remain documented.
