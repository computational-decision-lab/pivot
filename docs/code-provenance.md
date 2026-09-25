# Research source provenance (2026-09-26)

This release preserves historical research code supplied in the coauthor's server ZIP and a separate snapshot of the current local research tree. The two archives are independent: no local file overwrites a server version, and neither archive replaces the canonical source tree. Archived code is provided for inspection and reproducibility history; importing it does not establish that every historical experiment can be rerun in the current environment.

## Sources and mapping

- Server input: `服务器运行代码汇总_20260926.zip` (not included in the release), SHA-256 `2a1c4c6edec400e59575b8329db9f181e1610caeef565d45580544a3056babf1`.
- Server archive: `archive/server-code-20260926/`. Its `source-manifest.json` records every original ZIP file path, byte count, SHA-256, disposition, and destination or exclusion reason. Original ZIP filenames were decoded from CP437-stored UTF-8 bytes. The ten top-level version groups have readable ASCII destination names; paths inside each group are preserved. The sole ZIP symlink remains a symlink, with its target and link-byte hash recorded in the manifest.
- Local source: the current files under `src/`, `experiments/`, `scripts/`, `tests/`, and `configs/`, plus root `pyproject.toml` and `uv.lock`, from `/opt/projects/research/pivot` on 2026-09-26. `archive/local-code-20260926/source-manifest.json` records each file, disposition, hash, and destination. For each archived local file it also records the corresponding path in server group 02 when present and whether the bytes match.

The ZIP has 1,904 file entries: 1,019 macOS resource-fork entries and 885 ordinary files in ten version groups. The release archives 876 ordinary files (513 unique SHA-256 values). Nine ordinary files are excluded: two remote result-sync scripts, two cloud environment bootstrap scripts, one cloud budget watchdog, one remote SSH collection utility, two external LLM API/review utilities, and one paper-asset sync utility. Their paths, hashes, and individual reasons remain in the server manifest. The local snapshot archives 406 files; it excludes four cloud provisioning files, one paper-asset sync utility, and five generated egg-info files, each documented in its manifest.

Relative to the later server repository group, 302 local files are byte-identical, 38 have the same relative path but different bytes, and 66 have no same-path counterpart. The full contents of all 38 differing local files are retained in the local archive. The comparison is path-based and does not imply that absent files are scientifically new.

## Validation and handling

`scripts/import_research_sources.py` verifies the input ZIP CRC, hashes each source file and its archived copy, rejects unsafe or colliding paths, checks regular Python files with AST parsing without executing them, and scans included files for common literal credential and private-endpoint patterns. The raw ZIP is intentionally not published: it includes macOS metadata and operational scripts with private environment details. No credential replacement was needed in the research sources that were archived. The manifests contain hashes and paths for excluded files, not their contents.

To regenerate the archives from the same local sources, run:

```bash
python3 scripts/import_research_sources.py \
  --source-zip /opt/projects/research/pivot/服务器运行代码汇总_20260926.zip \
  --local-root /opt/projects/research/pivot \
  --output-root .
```

The source ZIP hash and manifest hashes provide the integrity check if a future copy is made from a different workspace. A different local snapshot will naturally change local hashes and comparison counts.

The Git archive preserves the one original symlink. The downloadable full ZIP
materializes that link as its target's exact file contents for portable
extraction on systems that do not restore Unix links. `RELEASE-LINK-EXPORTS.json`
records that export-only conversion; package checksums cover the materialized
file. Original source-manifest hashes continue to describe the supplied link.
