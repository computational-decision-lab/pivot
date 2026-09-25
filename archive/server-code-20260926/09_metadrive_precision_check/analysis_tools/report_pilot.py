"""Report the prespecified MetaDrive mechanism pilot; make no PIVOT efficacy claim.

Audit arrays contain paired candidate-minus-own-incumbent gains.  The independent
fixed-world audit is used for response-gap and reversal checks so ordinary proxy
Monte Carlo error is not mistaken for a response effect.  The observed proxy is
also retained and reported separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 78233


def bootstrap(values, seed=BOOTSTRAP_SEED):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("root-level bootstrap requires finite, nonempty 1D input")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(len(x), size=(BOOTSTRAP_DRAWS, len(x)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi),
            "n_independent_roots": int(len(x)), "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": int(seed)}


def _array(root, key, shape):
    a = np.asarray(root[key], dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"seed {root.get('seed')}: {key} expected finite shape {shape}, got {a.shape}")
    return a


def _engine_pass(checks):
    """Reject absent/nonboolean checks instead of inferring success from a file."""
    if not isinstance(checks, dict):
        raise ValueError("checks.json must be an object")
    flags = checks.get("checks")
    if not isinstance(flags, dict) or not flags:
        raise ValueError("checks.json requires a nonempty checks object of boolean values")
    if any(type(v) is not bool for v in flags.values()):
        raise ValueError("checks.json checks values must be explicit booleans")
    declared = checks.get("all_pass")
    if type(declared) is not bool:
        raise ValueError("checks.json requires boolean all_pass")
    if declared != all(flags.values()):
        raise ValueError("checks.json all_pass disagrees with individual checks")
    return declared


def build_report(roots, protocol, checks):
    expected = [int(x) for x in protocol["root_seeds"]]
    ids = [int(w["seed"]) for w in roots]
    if len(set(ids)) != len(ids) or sorted(ids) != sorted(expected):
        raise ValueError(f"pilot must contain exactly registered roots {expected}; got {ids}")
    roots = sorted(roots, key=lambda w: int(w["seed"]))
    adaptation = [int(x) for x in protocol["response"]["adaptation_steps"]]
    if adaptation != [4, 12]:
        raise ValueError("this fixed pilot requires adaptation_steps [4,12]")
    names = list(protocol["candidate_names"])
    k, audit_n = len(names), int(protocol["audit_episodes_per_split"])
    s, val_n = int(protocol["selection_episodes"]), int(protocol["response_validation_episodes"])
    if k != 6:
        raise ValueError("this fixed pilot requires six candidates")
    root_rows, reversal_rows, bg_values, gap_values, observed_gap_values = [], [], [], [], []
    scatter = []
    for w in roots:
        rid = int(w["seed"])
        if list(w["candidate_names"]) != names:
            raise ValueError(f"seed {rid}: candidate order differs from protocol")
        if list(w.get("adaptation_steps", adaptation)) != adaptation:
            raise ValueError(f"seed {rid}: adaptation order differs from protocol")
        proxy = _array(w, "proxy", (k,))
        _array(w, "selection", (2, k, s))  # validate only; never used to estimate deployment gain here
        aa = _array(w, "audit_a", (2, k, audit_n))
        ab = _array(w, "audit_b", (2, k, audit_n))
        fa = _array(w, "audit_fixed_a", (k, audit_n))
        fb = _array(w, "audit_fixed_b", (k, audit_n))
        validation = _array(w, "response_validation", (k + 1, 3, val_n))
        response_indices = _array(w, "response_indices", (k + 1, 2))
        a, b, fixed_a, fixed_b = aa.mean(axis=2), ab.mean(axis=2), fa.mean(axis=1), fb.mean(axis=1)
        deployment, fixed = (a + b) / 2.0, (fixed_a + fixed_b) / 2.0
        # Paired audit episode streams, including the respective adapted incumbent.
        paired_gap = ((aa - fa[None, :, :]).mean(axis=2) +
                      (ab - fb[None, :, :]).mean(axis=2)) / 2.0
        abs_gap = np.abs(paired_gap).mean(axis=1)
        bg = validation.mean(axis=(0, 2))
        bg_values.append(bg.tolist())
        gap_values.append(abs_gap.tolist())
        observed_gap_values.append(np.abs(deployment - proxy[None, :]).mean(axis=1).tolist())
        stable_flip = (fixed_a > 0) & (fixed_b > 0) & (a[1] < 0) & (b[1] < 0)
        for i in range(k):
            row = {"root": rid, "candidate_index": i + 1, "candidate_name": names[i],
                   "observed_proxy_gain": float(proxy[i]), "heldout_fixed_gain": float(fixed[i]),
                   "heldout_fixed_A_gain": float(fixed_a[i]), "heldout_fixed_B_gain": float(fixed_b[i]),
                   "short_gain": float(deployment[0, i]), "long_gain": float(deployment[1, i]),
                   "long_A_gain": float(a[1, i]), "long_B_gain": float(b[1, i]),
                   "stable_fixed_positive_long_negative": bool(stable_flip[i])}
            scatter.append(row)
            if stable_flip[i]:
                reversal_rows.append(row)
        root_rows.append({
            "root": rid, "background_heldout_mean_returns_fixed_short_long": bg.tolist(),
            "background_long_minus_fixed": float(bg[2] - bg[0]),
            "mean_abs_paired_gap_short_long": abs_gap.tolist(),
            "long_minus_short_mean_abs_paired_gap": float(abs_gap[1] - abs_gap[0]),
            "n_stable_reversal_candidates": int(stable_flip.sum()),
            "response_indices_incumbent_first": response_indices.astype(int).tolist(),
            "response_changed_from_short_to_long_count": int(np.sum(response_indices[:, 0] != response_indices[:, 1])),
            "candidate_response_differs_from_incumbent_fraction_short_long":
                np.mean(response_indices[1:] != response_indices[0][None, :], axis=0).tolist(),
            "unique_adapted_profiles_among_incumbent_and_candidates_short_long":
                [int(len(np.unique(response_indices[:, hi]))) for hi in range(2)],
            "observed_proxy_gain": proxy.tolist(), "heldout_fixed_gain": fixed.tolist(),
            "deployment_gain_short_long": deployment.tolist(),
            "runner_cost_accounting": w.get("cost_accounting", w.get("costs")),
            "runner_wallclock_seconds": w.get("seconds", w.get("wallclock_seconds")),
        })
    bg_values, gap_values = np.asarray(bg_values), np.asarray(gap_values)
    observed_gap_values = np.asarray(observed_gap_values)
    bg_contrast = bootstrap(bg_values[:, 2] - bg_values[:, 0])
    gap_contrast = bootstrap(gap_values[:, 1] - gap_values[:, 0])
    gate_values = {
        "engine_checks_required": _engine_pass(checks),
        "heldout_background_long_minus_fixed_positive": bool(bg_contrast["mean"] > 0),
        "mean_abs_gap_long_greater_than_short": bool(gap_contrast["mean"] > 0),
        "at_least_one_proxy_positive_to_long_negative_flip_stable_in_a_and_b": bool(reversal_rows),
    }
    required = protocol["pilot_go_rule"]
    if set(required) != set(gate_values) or any(v is not True for v in required.values()):
        raise ValueError("pilot_go_rule differs from the fixed four registered requirements")
    return {
        "schema": "metadrive_mechanism_pilot_report_v1", "phase": "mechanism_pilot",
        "n_independent_roots": len(roots), "root_seeds": sorted(ids), "candidate_names": names,
        "pilot_go": bool(all(gate_values.values())), "pilot_checks": gate_values,
        "on_no_go": protocol["on_no_go"], "engine_checks": checks,
        "background_heldout_return": {
            str(h): bootstrap(bg_values[:, i]) for i, h in enumerate([0, 4, 12])},
        "background_long_minus_fixed": bg_contrast,
        "mean_abs_paired_gap_vs_heldout_fixed": {
            str(h): bootstrap(gap_values[:, i]) for i, h in enumerate(adaptation)},
        "long_minus_short_mean_abs_paired_gap": gap_contrast,
        "mean_abs_gap_vs_observed_proxy_reported_only": {
            str(h): bootstrap(observed_gap_values[:, i]) for i, h in enumerate(adaptation)},
        "n_stable_reversal_root_candidate_pairs": len(reversal_rows),
        "candidate_conditioned_response_diagnostics_reported_only": {
            "candidate_response_differs_from_incumbent_fraction": {
                str(h): bootstrap([r["candidate_response_differs_from_incumbent_fraction_short_long"][hi]
                                   for r in root_rows]) for hi, h in enumerate(adaptation)},
            "unique_adapted_profiles_mean": {
                str(h): float(np.mean([r["unique_adapted_profiles_among_incumbent_and_candidates_short_long"][hi]
                                      for r in root_rows])) for hi, h in enumerate(adaptation)},
            "note": "descriptive only; identical response choices do not demonstrate candidate-specific feedback; no additional gate",
        },
        "stable_reversals": reversal_rows, "per_root": root_rows, "candidate_points": scatter,
        "definitions": {
            "gain": "candidate native cumulative return minus own-incumbent native cumulative return; each is evaluated in its corresponding response world",
            "heldout_fixed_gain": "mean of paired gains across independent fixed-world audit_A and audit_B; separately reported from observed proxy",
            "paired_gap": "response-world paired gain minus fixed-world paired gain on matched audit episode streams",
            "response_strength": "number of predetermined controller profiles evaluated (4 or12); not PPO gradient steps or an exact best response",
            "background_validation": "fixed-initial-roster background native return; average over incumbent+six candidate branches and held-out validation episodes within each root",
            "stable_reversal": "at least one root/candidate has fixed_A>0, fixed_B>0, long_A<0, long_B<0",
            "gate_interval_rule": "gates use registered mean signs; bootstrap intervals are reported, not added as extra gates",
        },
        "interpretation_limits": [
            "Four-root exploratory mechanism pilot, not confirmation or a powered efficacy comparison.",
            "No PIVOT selection comparison is run or claimed in this report.",
            "Confidence intervals bootstrap independent roots, not correlated candidate/episode observations.",
            "A sign stable across two small audit splits is a pilot criterion, not a statistically proven reversal.",
            "A failed gate stops formal expansion under this protocol; it does not establish absence of a response mechanism.",
            protocol["scientific_scope"],
        ],
    }


def plot_report(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 220})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    roots = report["per_root"]
    n = report["n_independent_roots"]
    files = []
    def save(fig, name):
        fig.tight_layout()
        for ext in ("pdf", "png"):
            target = output / f"{name}.{ext}"
            fig.savefig(target, bbox_inches="tight")
            files.append(str(target))
        plt.close(fig)
    def ci_plot(ax, x, stats, color="#205493"):
        means = np.array([v["mean"] for v in stats])
        lo = np.array([v["lo"] for v in stats]); hi = np.array([v["hi"] for v in stats])
        ax.errorbar(x, means, yerr=np.vstack([np.maximum(means-lo,0), np.maximum(hi-means,0)]),
                    color=color, marker="o", linewidth=2, capsize=4,
                    label=f"Mean and root-bootstrap 95% CI (n={n})")
    fig, ax = plt.subplots(figsize=(5.8, 4.1))
    for r in roots:
        ax.plot([0, 4, 12], r["background_heldout_mean_returns_fixed_short_long"],
                color="#91b1cc", alpha=.5, linewidth=.9)
    ci_plot(ax, [0, 4, 12], [report["background_heldout_return"][str(h)] for h in (0,4,12)])
    ax.set(xticks=[0,4,12], xticklabels=["Fixed (0)", "Short (4)", "Long (12)"],
           xlabel="Response search: controller profiles evaluated",
           ylabel="Held-out mean background native return",
           title="Does the response improve its own held-out return?")
    ax.legend(fontsize=8, loc="best")
    save(fig, "01_background_response_return")
    fig, ax = plt.subplots(figsize=(5.8, 4.1))
    for r in roots:
        ax.plot([4,12], r["mean_abs_paired_gap_short_long"], color="#d6aa7f", alpha=.5, linewidth=.9)
    ci_plot(ax, [4,12], [report["mean_abs_paired_gap_vs_heldout_fixed"][str(h)] for h in (4,12)], color="#bb6c25")
    ax.set(xticks=[4,12], xticklabels=["Short (4)", "Long (12)"],
           xlabel="Response search: controller profiles evaluated",
           ylabel="Mean absolute response-induced gain gap",
           title="Does a stronger response separate deployment from proxy?")
    ax.legend(fontsize=8, loc="best")
    save(fig, "02_proxy_deployment_gap")
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    x = np.array([p["heldout_fixed_gain"] for p in report["candidate_points"]])
    y_short = np.array([p["short_gain"] for p in report["candidate_points"]])
    y_long = np.array([p["long_gain"] for p in report["candidate_points"]])
    ax.scatter(x, y_short, s=28, marker="o", color="#205493", alpha=.65, label="Short response (4)")
    ax.scatter(x, y_long, s=38, marker="^", color="#bb6c25", alpha=.75, label="Long response (12)")
    combined = np.concatenate([x, y_short, y_long, [0]])
    low, high = float(combined.min()), float(combined.max())
    margin = max((high-low)*.06, .05)
    low, high = low-margin, high+margin
    ax.plot([low,high], [low,high], linestyle="--", linewidth=1, color=".4", label="Equal gain")
    ax.axhline(0,color=".75",linewidth=.7); ax.axvline(0,color=".75",linewidth=.7)
    ax.set(xlim=(low,high), ylim=(low,high), xlabel="Fixed-world proxy gain (held-out audit)",
           ylabel="Deployment gain (held-out audit)",
           title=f"Each point is one root/candidate (n={n} roots)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, loc="best")
    save(fig, "03_proxy_vs_deployment")
    return files


def write_readme(report, output):
    bg, gap = report["background_long_minus_fixed"], report["long_minus_short_mean_abs_paired_gap"]
    outcome = "通过预设机制门槛，可按冻结流程进入后续阶段。" if report["pilot_go"] else "未通过预设机制门槛，按协议停止正式扩展。"
    lines = ["# MetaDrive 机制试运行结果", "", outcome, "",
             f"本次只有 {report['n_independent_roots']} 个独立 root；这是探索性的机制检查，还没有比较 PIVOT 与其他选择方法。", "",
             f"- 背景车辆长响应相对固定行为的平均收益差：{bg['mean']:.6g}，root bootstrap 95% 区间 [{bg['lo']:.6g}, {bg['hi']:.6g}]。",
             f"- 长响应减短响应的平均绝对 proxy/deployment gap：{gap['mean']:.6g}，区间 [{gap['lo']:.6g}, {gap['hi']:.6g}]。",
             f"- 在 audit A 和 B 中都保持“固定世界收益差为正、长响应世界收益差为负”的 root/候选对：{report['n_stable_reversal_root_candidate_pairs']}。", "",
             "门槛判断：", ""]
    lines += [f"- `{key}`：{str(value).lower()}" for key,value in report["pilot_checks"].items()]
    lines += ["", "三个图均单独提供 PDF 和 PNG：", "",
              "1. 背景车辆收益：检查响应是否在独立场景里提高了自身收益；浅色线是各 root。",
              "2. 收益差的分离：同一批独立审计场景中，响应环境与固定环境的策略升级收益差相差多少。",
              "3. 固定环境与部署环境散点：每个点是一个 root/候选，两边都用独立 audit 均值；右下角表示固定环境认为变好、部署后变差。", "",
              "图里的 fixed-world proxy 使用 held-out fixed audit；生成候选时观察到的 proxy 数字另列在 JSON 中。二者不能混写。", "",
              "这里的 short/long 是评估 4/12 个预设控制器配置后择优，不是训练神经网络 4/12 步，也不是求得精确 best response。", "",
              "全部门槛依据预先规定的均值符号及双 split 符号稳定性；区间是补充报告，未额外加入通过条件。小样本通过不等于确认研究结论，失败也不证明机制不存在。", "",
              "完整数字、逐 root 结果和运行检查见 `pilot_report.json`；源输出保留在 `roots/`。", ""]
    (Path(output) / "README_试运行结果.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checks", type=Path, required=True)
    args = parser.parse_args()
    protocol, checks = json.loads(args.protocol.read_text()), json.loads(args.checks.read_text())
    paths = sorted((args.out / "roots").glob("seed_*/root.json"))
    roots = [json.loads(p.read_text()) for p in paths]
    report = build_report(roots, protocol, checks)
    report["provenance"] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "checks_sha256": hashlib.sha256(args.checks.read_bytes()).hexdigest(),
        "root_files_sha256": {str(p.relative_to(args.out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "report_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "pilot_report.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    figures = plot_report(report, args.out / "figures")
    write_readme(report, args.out)
    print(json.dumps({"pilot_go": report["pilot_go"], "pilot_checks": report["pilot_checks"],
                      "n_independent_roots": report["n_independent_roots"],
                      "pilot_report": str(target), "figures": figures}, ensure_ascii=False))


if __name__ == "__main__":
    main()
