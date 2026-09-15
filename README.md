# IMPROVE-X / PIVOT

<p align="center">
  <a href="docs/pivot.md"><strong>English Documentation</strong></a>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="docs/zh/00_项目入口与当前状态.md"><strong>中文文档</strong></a>
</p>

**Improvement Fidelity in Adaptive Worlds**

PIVOT (Paired Interventional Validation of Optimization Transitions) studies whether a self-improvement update that looks beneficial in a cheap proxy world remains beneficial after deployment changes the environment and other participants respond.

The statistical object is a directed transition `pi -> pi'`, rather than two isolated policy scores.

## Current status

| Area | Status |
| --- | --- |
| Research question and protocol | Frozen and documented |
| Theory, core algorithms, and controlled experiments | Implemented; local checks pass |
| V9 registered controlled evidence | Frozen; claims are scoped to registered mechanisms |
| V15 external-agent study | Engineering and DEV checks complete; confirmatory execution not opened |
| Paper, supplement, and release package | Local machine checks pass; manual submission gates remain |
| Overall scientific state | `BLOCKED` pending external confirmatory evidence and manual gates |

## Start here

- **English research guide:** [`docs/pivot.md`](docs/pivot.md)
- **English research question:** [`docs/research_question.md`](docs/research_question.md)
- **English estimands and metrics:** [`docs/estimands.md`](docs/estimands.md)
- **English experiment protocol:** [`docs/experiment_protocol.md`](docs/experiment_protocol.md)
- **Chinese project guide:** [`docs/zh/00_项目入口与当前状态.md`](docs/zh/00_项目入口与当前状态.md)
- **Chinese theory and algorithms:** [`docs/zh/01_研究问题与理论.md`](docs/zh/01_研究问题与理论.md), [`docs/zh/02_算法与代码架构.md`](docs/zh/02_算法与代码架构.md)
- **Chinese experiments and operations:** [`docs/zh/03_实验设计与指标.md`](docs/zh/03_实验设计与指标.md), [`docs/zh/04_运行手册.md`](docs/zh/04_运行手册.md)
- **Chinese evidence and handoff:** [`docs/zh/05_结果与证据边界.md`](docs/zh/05_结果与证据边界.md), [`docs/zh/06_同事交接清单.md`](docs/zh/06_同事交接清单.md)
- **Version and file map:** [`docs/zh/版本与审计索引.md`](docs/zh/版本与审计索引.md), [`docs/zh/目录与文件地图.md`](docs/zh/目录与文件地图.md)

## Method

Each round generates candidate transitions, measures proxy deltas and update footprints, selects a fixed high-fidelity query budget, evaluates incumbent and candidate in paired contexts, and records corrected estimates, selection, cost, and provenance.

```text
incumbent -> candidate transitions -> proxy + footprint
           -> PIVOT/PIVOT-VOI acquisition
           -> paired high-fidelity evaluation
           -> correction, selection, metrics, manifest
```

The stable public interface is [`src/pivot_core/`](src/pivot_core/). The platform layer is [`src/improve_x/`](src/improve_x/).

## Quick checks

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src scripts experiments tests
```

Build and evaluate ImprovementBench:

```bash
.venv/bin/python scripts/build_improvementbench.py \
  --config configs/improve_x/benchmark.yaml \
  --output /tmp/improvementbench-v1
```

Rebuild the paper and curated release:

```bash
make v15-finalize
make v15-release
```

The underlying finalizer is also available as
`.venv/bin/python -m experiments.v15 finalize --root .`.

## Repository layout

| Directory | Purpose |
| --- | --- |
| `src/` | Core libraries and stable facades |
| `experiments/` | P0-P9, V9, V10, and V15 runners and audits |
| `configs/` | Registered protocols, experiment parameters, and runtime locks |
| `results/` | Version-isolated raw and canonical results |
| `paper/` | Canonical ICLR 2027 submission source, supplement, and release artifacts |
| `release/` | Sanitized anonymous release package |
| `docs/zh/` | Chinese team documentation and handoff guide |
| `tests/` | Unit and integration tests |

## Evidence boundary

V9 results apply only to their preregistered controlled mechanisms. V15 stages marked `DEV_ONLY`, `UNDERPOWERED`, or `NOT_RUN` are not confirmatory evidence. The public finance audit is observational and does not identify causal market response. Live trading and unauthorized external execution are out of scope.
