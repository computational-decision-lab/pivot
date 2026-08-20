# PIVOT Working Paper

This directory contains an anonymous working-paper build. The paper consumes
only the hash-indexed files in `snapshot/`; it does not read `/tmp`, network
data, credentials, or live execution endpoints.

Build from this directory:

```bash
./build.sh
```

Outputs:

- `build/main.pdf`: compiled paper;
- `pivot_working_paper.pdf`: copied handoff PDF;
- `verification.json`: page, text, font, metadata, and raster checks;
- `snapshot/manifest.json`: source artifact hashes.

The PDF is an anonymous working paper, not a claim that the public finance
proxy is causal ground truth. The strategic and public-data limitations in the
paper are intentional.
