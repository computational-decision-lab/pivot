# PIVOT 文档体系整理实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变科研证据路径的前提下，建立中文主文档、算法/理论/实验说明和同事交接流程，并推送到项目 GitHub 远程。

**Architecture:** 以 `docs/zh/` 作为唯一中文导航层，根 README 只提供入口；既有英文审计、论文、配置和结果继续保留并作为证据源。文档通过相对链接连接源码、命令和状态文件，不移动原文件。

**Tech Stack:** Markdown、现有 Python 3.10 虚拟环境、pytest、ruff、Makefile、Git。

## Global Constraints

- 中文主文档与英文权威原文并存。
- 不移动或重命名现有 `src/`、`experiments/`、`configs/`、`results/`、`paper/`、`release/` 文件。
- 不把 DEV、UNDERPOWERED、NOT_RUN 或 BLOCKED 改写为确认性证据。
- 只提交 `/opt/projects/research/pivot` 目录下的变更。
- 推送前必须执行测试、静态检查、差异审阅和 `git diff --check`。

### Task 1: 写入中文项目入口和文件地图

**Files:**
- Create: `docs/zh/00_项目入口与当前状态.md`
- Create: `docs/zh/目录与文件地图.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: `README.md`、`docs/archive/v15/V15_FINAL_REPORT.md`、`docs/archive/v15/V15_IMPLEMENTATION_MATRIX.md`、`release/v15/submission_verification.json`、`research/research_state.json`。
- Produces: 新同事可从根 README 进入中文总览，并能定位每类源码和证据。

- [ ] 从现有状态文件提取当前状态、证据等级和阻塞项。
- [ ] 编写中文入口、推荐阅读顺序、项目边界和文件地图。
- [ ] 在根 README 顶部加入 `docs/zh/00_项目入口与当前状态.md` 链接。
- [ ] 检查所有链接指向现有文件或本任务新增文件。

### Task 2: 编写理论与算法说明

**Files:**
- Create: `docs/zh/01_研究问题与理论.md`
- Create: `docs/zh/02_算法与代码架构.md`

**Interfaces:**
- Consumes: `docs/theory_notes.md`、`docs/estimands.md`、`docs/claim_boundary.md`、`src/pivot/`、`src/pivot_core/`、`src/improve_x/`。
- Produces: 理论符号、命题边界、PIVOT/PIVOT-VOI/IMPROVE-X 算法流程与源码映射。

- [ ] 说明 `pi -> pi'`、`Delta_V`、`Delta_*`、响应分解和 Improvement Fidelity。
- [ ] 逐一记录 T1-T6 或现有理论目标的条件、结论和证据级别。
- [ ] 说明 transition record、配对评估、footprint、差分后验、VOI 获取和停止规则。
- [ ] 将算法步骤映射到稳定 facade 和具体模块。
- [ ] 明确哪些是理论充分条件、哪些是受控实验假设、哪些仍待外部验证。

### Task 3: 编写实验设计、运行和结果边界

**Files:**
- Create: `docs/zh/03_实验设计与指标.md`
- Create: `docs/zh/04_运行手册.md`
- Create: `docs/zh/05_结果与证据边界.md`

**Interfaces:**
- Consumes: `docs/experiment_protocol.md`、`docs/experiments/`、`configs/`、`experiments/v9/`、`experiments/v15/`、`V15_*` 审计文件。
- Produces: 实验矩阵、指标定义、命令目录、输出目录和可引用结论边界。

- [ ] 整理 World 0/1/2、F0-F2、V9 和 V15 的目的、输入、输出及状态。
- [ ] 记录配对/非配对对照、基线、预算、bootstrap 单位和终止状态。
- [ ] 给出从环境检查到 smoke、正式审计、发布验证的可执行命令。
- [ ] 汇总已完成、UNDERPOWERED、NOT_RUN、BLOCKED 和人工门槛。
- [ ] 对每条重要结论给出允许写法和禁止写法。

### Task 4: 编写交接和维护规范

**Files:**
- Create: `docs/zh/06_同事交接清单.md`
- Create: `docs/zh/文档维护规范.md`

**Interfaces:**
- Consumes: 前三项中文文档、`docs/iclr2027-execution-schedule.md`、`configs/v15/confirmatory.yaml`、`release/v15/confirmatory_lock.json`。
- Produces: 新同事按顺序执行的接手清单，以及后续更新文档/证据的规则。

- [ ] 按“先理解、再本地复现、再审计、最后申请确认性运行”的顺序列出任务。
- [ ] 每个任务写明输入、命令、输出、验收条件和不可越过的门槛。
- [ ] 规定权威文件优先级、状态变更、哈希变更和发布包更新规则。
- [ ] 增加提交前检查清单。

### Task 5: 验证、提交和推送

**Files:**
- Modify: 仅本项目内需要同步的文件

**Interfaces:**
- Consumes: 全部新增中文文档和现有项目验证入口。
- Produces: 可审阅 Git 提交并推送到 `origin`。

- [ ] 执行 Markdown 路径检查和 `git diff --check`。
- [ ] 执行 `.venv/bin/pytest -q`。
- [ ] 执行 `.venv/bin/ruff check src scripts experiments tests`。
- [ ] 审阅 `git diff --stat`、受控状态和待提交文件，排除父目录其他项目。
- [ ] 提交并推送当前分支到 `https://github.com/computational-decision-lab/pivot`。
