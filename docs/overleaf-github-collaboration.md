# PIVOT 论文协作说明：GitHub ↔ Overleaf 双向同步

更新日期：2026-09-17。本文描述当前已部署的配置，供论文合作者使用。

**可以在 Overleaf 写论文，也可以通过 GitHub 更新论文文件。两端通过自动任务同步，但不是实时共同编辑；切换编辑平台前，请先确认上一次同步成功。**

## 1. 常用入口

| 入口 | 链接与用途 |
| --- | --- |
| Overleaf 论文 | [打开项目](https://www.overleaf.com/project/6aab7ef4c95659076d75b579)：在线编辑、讨论和编译 |
| GitHub 仓库 | [computational-decision-lab/pivot](https://github.com/computational-decision-lab/pivot)：研究代码、实验与版本记录 |
| GitHub 论文目录 | [master 分支的 paper/](https://github.com/computational-decision-lab/pivot/tree/master/paper)：同步的论文源文件 |
| 正向同步任务 | [Sync paper to Overleaf](https://github.com/computational-decision-lab/pivot/actions/workflows/sync-paper-to-overleaf.yml)：GitHub → Overleaf |
| 反向同步任务 | [Sync Overleaf to GitHub](https://github.com/computational-decision-lab/pivot/actions/workflows/sync-overleaf-to-github.yml)：Overleaf → GitHub |
| 提交历史 | [GitHub master 历史](https://github.com/computational-decision-lab/pivot/commits/master/)：核对每次更新与责任来源 |

链接本身不授予访问权限。请联系项目维护者开通相应的 Overleaf 和 GitHub 权限；仅在 Overleaf 写作的合作者不需要自行配置 Git token。

## 2. 为什么需要双向同步

- **各用合适的工具。** 合作者在 Overleaf 共同写作、查看 PDF；研究人员和 Agent 在 GitHub 管理论文源文件、代码以及实验产物。
- **减少手工传文件。** 无需反复下载 ZIP、上传图表或复制 LaTeX，降低两边版本不一致的概率。
- **保留可追溯记录。** Overleaf 导入会在 GitHub 生成提交，便于查看改动、比较版本和人工恢复。
- **保护双方修改。** 同步遇到冲突会停止，不通过强制推送覆盖已有历史。

这套流程同步的是文件，不会自动判断实验结论是否正确，也不会替代作者审阅。

## 3. 为什么可行，以及已验证到哪一步

项目通过 Overleaf Git 接口读写论文，GitHub Actions 负责运行同步任务。认证凭据保存在 GitHub Actions Secret 中，不放入论文或本说明。

2026-09-17 已完成真实双向实验：

| 实验 | 操作与结果 | 证据 |
| --- | --- | --- |
| GitHub → Overleaf | 在 GitHub 新增独立测试文件，push 自动触发同步；重新克隆 Overleaf 后，确认内容逐字节一致 | [自动同步运行](https://github.com/computational-decision-lab/pivot/actions/runs/35225880510)、[远端内容校验](https://github.com/computational-decision-lab/pivot/actions/runs/35225893671) |
| Overleaf → GitHub | 通过 Overleaf Git 接口修改测试文件；反向任务自动创建 GitHub 提交，核对内容一致 | [反向运行](https://github.com/computational-decision-lab/pivot/actions/runs/35225962926)、[自动提交 efb2d19](https://github.com/computational-decision-lab/pivot/commit/efb2d1910983f19a5808ae9063454fa154bfa67b) |
| 实验清理 | 删除测试文件后同步，并确认两端均无残留；临时实验 workflow 已删除，论文正文未改动 | [清理验证](https://github.com/computational-decision-lab/pivot/actions/runs/35226036466) |

**验证边界：** 反向实验是手动触发同一个同步任务，证明实际导入链路可用；没有以这次实验验证等待定时调度的触发时间，也没有模拟 Overleaf 网页编辑器操作。计划任务配置为每 15 分钟检查一次，实际启动可能排队延迟。

## 4. 自动更新的路径和触发方式

```text
GitHub：master 分支 / paper/ 中的论文文件
       │
       │ push 涉及 paper/ 时自动触发
       ▼
Overleaf：本项目 main 分支 / 项目根目录
       │
       │ GitHub Actions 每 15 分钟计划检查一次
       ▼
GitHub：master 分支 / paper/ 中的论文文件
```

| 方向 | 触发条件 | 更新方式 |
| --- | --- | --- |
| GitHub → Overleaf | 向 master 推送涉及 paper/ 的改动，或修改正向 workflow；也可手动运行 | 检查 Overleaf 同步基线，再提交到 Overleaf 的 main 分支 |
| Overleaf → GitHub | 定时检查，或手动运行反向任务 | 比较上次同步内容；无冲突时直接提交到 GitHub master，不创建 PR |

本地修改但没有 push，不会触发正向同步。在其他分支修改，需要合并到 master 才进入自动同步。

反向任务只在有文件变化时生成论文提交；正常无变化时成功退出。两个同步任务共用并发锁，避免彼此同时运行。反向导入还会更新同步基线，正常情况下不会形成来回重复提交。

## 5. 合作者如何使用 Overleaf

1. 打开 [Overleaf 项目](https://www.overleaf.com/project/6aab7ef4c95659076d75b579)，确认此前同步没有失败，再开始编辑。
2. 正常修改正文、参考文献或图片，等待 Overleaf 保存。建议多人主要在 Overleaf 同一个项目内共同写作。
3. 修改完成后执行 Recompile，检查 PDF、引用和排版。**同步成功不等于论文编译成功。**
4. 等待下一次反向检查；如需立即同步，请有 GitHub 操作权限的人打开 [反向同步任务](https://github.com/computational-decision-lab/pivot/actions/workflows/sync-overleaf-to-github.yml)，点击 **Run workflow**，选择 `master` 并运行。
5. 在 [提交历史](https://github.com/computational-decision-lab/pivot/commits/master/) 查看 `Sync manuscript changes from Overleaf` 提交，打开具体文件确认内容已到达。

无需每次下载 ZIP，也无需把 token 发给其他合作者。

## 6. 如何从 GitHub 或本地更新

1. 开始本地编辑前，确认 Overleaf 的最新修改已反向同步，再拉取 GitHub 最新版本。
2. 修改 `paper/` 中需要同步的文件，检查差异和论文编译结果。
3. 提交并 push 到 `master`；团队使用 PR 时，合并到 `master` 后触发同步。
4. 在 [正向同步任务](https://github.com/computational-decision-lab/pivot/actions/workflows/sync-paper-to-overleaf.yml) 确认本次运行成功，再打开 Overleaf 检查文件和 PDF。

下面仅适用于独立的 PIVOT 仓库克隆，且本地工作区干净；不要在包含其他项目的大工作树中直接照抄：

```bash
# 在 PIVOT 仓库根目录，开始编辑前
 git switch master
 git pull --ff-only origin master

# 编辑和本地检查后，逐个添加实际修改的文件
 git diff -- paper/
 git add paper/main.tex
 git commit -m "Revise manuscript discussion"
 git push origin master
```

如果使用其他分支、工作区有未提交内容或 Git 报冲突，应按团队 Git 流程处理，不要强制推送。

## 7. 哪些内容同步，哪些不同步

| 内容 | 当前行为 |
| --- | --- |
| 论文源文件、参考文献、样式、图片 | 同步 `.tex`、`.bib`、`.sty`、`.bst`、`.cls`、`.pdf`、`.png`、`.jpg`、`.jpeg`、`.eps` |
| 目录映射 | GitHub `paper/main.tex` 对应 Overleaf 根目录 `main.tex`；其他受支持文件保留相对目录 |
| 新增、修改、删除 | 受支持文件可同步这些变化；重命名通常表现为删除旧文件、增加新文件；反向任务拒绝删除 `main.tex` |
| 归档、构建、补充材料、快照 | 排除 `archive/`、`build/`、`supplementary/`、`snapshot/` 目录 |
| 提交版 PDF | 名称以 `pivot_iclr2027_submission` 开头的文件不导出 |
| 实验代码、原始结果、paper/ 外的文件 | 不在同步范围；须先按研究流程生成或复制所需论文产物到 paper/ |
| Markdown、CSV、JSON、SVG 等其他格式 | 不属于当前论文内容同步白名单 |
| Overleaf 评论、聊天、账户设置、协作者权限 | 不同步到 GitHub |
| Overleaf `.github-sync.json` 和 `latexmkrc` | 同步机制维护的状态和编译配置，不要手动修改或删除 |

为适配 Overleaf 根目录，`main.tex` 中的表格路径会在同步时转换：GitHub 的 `../tables/` 对应 Overleaf 的 `tables/`。两边路径文字不完全一致可能是预期行为；项目根目录的 `tables/` 本身不属于同步范围，维护者须保证需要的表格已在 `paper/tables/` 中准备好。

## 8. 冲突、失败和协作约定

**推荐约定：同一时间尽量只在一端修改同一个文件。** 切换平台前确认同步完成；若大量正文集中在 `main.tex`，应提前约定谁在 GitHub 修改、谁在 Overleaf 修改。

- **Overleaf 有尚未导入的修改：** 正向任务可能先停止保护文件。先运行反向任务，成功后再运行正向任务。
- **双方同时修改同一文件、内容不同：** 反向任务停止，不会自动挑选一方。判断按文件进行，即使改了同一文件的不同段落，也可能需要人工合并。
- **双方修改不同文件：** 反向任务可导入 Overleaf 的变化并保留 GitHub 单边变化；若之前正向任务失败，需要再运行一次正向任务。
- **权限、网络或并发推送失败：** 查看失败步骤；内容未冲突时可在最新状态上重试。不要用 `push --force` 解决。

遇到红色失败状态，请把对应 Actions 运行链接和冲突文件名发给维护者。维护者应保留双方版本、人工核对并合并内容，再协调更新同步基线和恢复双向任务；不要删除状态文件或仅反复点击重试来跳过冲突。

手动任务入口中的 **Run workflow** 需要相应仓库权限。合作者看不到按钮时，可请维护者运行。

## 9. 日常协作检查清单

- 开始编辑前：确认目标平台内容已更新，没有未处理的同步失败。
- 修改完成后：确认保存或 push 成功，并检查 PDF。
- 切换平台前：确认同步任务成功，抽查刚改过的文件。
- 发生冲突时：暂停该文件的双端编辑，把失败链接交给维护者。

当前同步任务不包含每次完整编译、科学结论审查或自动解决冲突。投稿前仍需统一运行论文构建与审查流程。
