"""Cross-benchmark figures + LaTeX table from analyze_v4 confirm summaries.

Usage:
  python make_cross_benchmark_figures.py --out figs \
      "Melting Pot v4 (mixed responders)=/path/melting_v4_confirm/analysis/summary.json" \
      "Kuhn v2b (latent family, 4096 hands)=/path/kuhn_v2b_confirm/analysis/summary.json" \
      "Leduc v2b=/path/leduc_v2b_confirm/analysis/summary.json" \
      "Kuhn v2 (256 hands, instrument-limited)=/path/kuhn_confirm/analysis/summary.json"

Produces:
  forest_primary.pdf/png      H2 (long, primary cap) and H3 (interaction) per benchmark, root-bootstrap 95% CI,
                              plus calibration value (no_hf - proxy) and query value (kg - no_hf)
  isr_vs_cost_<label>.pdf/png HF cost vs mean ISR for every method at long, all caps
  table_main.tex              one row per benchmark x method at primary cap (long), ISR / cost / n
  mechanism_table.tex         squared-gap long-short, best-candidate-by-type, info ceiling
Raw regret units are native per environment and are never compared across benchmarks (only signs / CIs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHOD_LABEL = {"pivot_kg": "PIVOT (exact EVSI/cost)", "pivot_kg_stop": "PIVOT + stop", "uniform_v2": "Uniform HF",
                "ivr_v2": "Global-IVR", "lucb_v2": "posterior best/challenger", "no_hf_v2": "calibrated, no HF",
                "proxy_only": "Proxy only", "all_hf_reference": "All-HF (reference)"}


def load(spec: str):
    label, path = spec.split("=", 1)
    return label.strip(), json.loads(Path(path).read_text())


def ci_row(d):
    return (d["mean"], d["lo"], d["hi"], d["n"])


def forest(items, out: Path):
    rows = []
    for label, s in items:
        h = s["hypotheses"]; sec = h["secondary"]
        rows.append((label, "H2: PIVOT − Uniform (long, primary cap)", ci_row(h["H2_long_primary_gain_minus_comparator_cap192"])))
        rows.append((label, "H3: interaction long − short", ci_row(h["H3_interaction_long_minus_short"])))
        rows.append((label, "single query: PIVOT − Uniform (long)", ci_row(sec["long_smallest_cap_single_query_kg_minus_uniform"])))
        rows.append((label, "query value: PIVOT − calibrated no-HF (long)", ci_row(sec["long_kg_minus_no_hf_v2 (query value over calibration)"])))
        rows.append((label, "calibration value: no-HF − proxy (long)", ci_row(sec["long_no_hf_v2_minus_proxy_only (calibration value)"])))
    nb = len(items)
    fig, axes = plt.subplots(1, nb, figsize=(4.2 * nb, 3.6), squeeze=False)
    for ax, (label, _) in zip(axes[0], items):
        sub = [r for r in rows if r[0] == label]
        y = np.arange(len(sub))[::-1]
        for yi, (_, name, (m, lo, hi, n)) in zip(y, sub):
            color = "#1f77b4" if lo > 0 else ("#d62728" if hi < 0 else "#7f7f7f")
            ax.errorbar(m, yi, xerr=[[m - lo], [hi - m]], fmt="o", color=color, capsize=3)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels([r[1] for r in sub], fontsize=7)
        ax.set_title(f"{label}\n(n={sub[0][2][3]} roots; native units)", fontsize=8)
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout(); fig.savefig(out / "forest_primary.pdf"); fig.savefig(out / "forest_primary.png", dpi=180); plt.close(fig)


def isr_vs_cost(label, s, out: Path):
    hl = max(int(r["adaptation"]) for r in s["method_table_cap192"])
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    pts = {}
    for cap, tbl in s["method_tables_by_cap"].items():
        for r in tbl:
            if r["adaptation"] != hl:
                continue
            pts.setdefault(r["method"], []).append((r["mean_hf_episode_cost"], r["mean_isr"]))
    for m, xy in pts.items():
        xy = sorted(set(xy))
        ax.plot([p[0] for p in xy], [p[1] for p in xy], "o-" if len(xy) > 1 else "o", label=METHOD_LABEL.get(m, m), ms=4)
    ax.set_xlabel("charged HF cost (native units)"); ax.set_ylabel("mean ISR (lower is better)")
    ax.set_title(f"{label}: long response", fontsize=9); ax.grid(alpha=0.2); ax.legend(fontsize=6)
    fig.tight_layout(); safe = "".join(c if c.isalnum() else "_" for c in label)
    fig.savefig(out / f"isr_vs_cost_{safe}.pdf"); fig.savefig(out / f"isr_vs_cost_{safe}.png", dpi=180); plt.close(fig)


def table_main(items, out: Path):
    lines = ["\\begin{tabular}{llrrr}", "\\toprule", "Benchmark & Method & ISR (long) & HF cost & $n$ \\\\", "\\midrule"]
    for label, s in items:
        hl = max(int(r["adaptation"]) for r in s["method_table_cap192"])
        rows = [r for r in s["method_table_cap192"] if r["adaptation"] == hl]
        for i, r in enumerate(sorted(rows, key=lambda r: r["mean_isr"])):
            lines.append(f"{label if i == 0 else ''} & {METHOD_LABEL.get(r['method'], r['method'])} & {r['mean_isr']:.4g} & {r['mean_hf_episode_cost']:.0f} & {r['n']} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"; lines.append("\\end{tabular}")
    (out / "table_main.tex").write_text("\n".join(lines) + "\n")


def table_mechanism(items, out: Path):
    lines = ["\\begin{tabular}{lrrrl}", "\\toprule", "Benchmark & gap$^2$ long$-$short [95\\% CI] & info ceiling (long) & pop.-prior value (long) & best update by latent type (long) \\\\", "\\midrule"]
    for label, s in items:
        mech = s["mechanism"]; hl = str(max(int(k) for k in mech if k.isdigit()))
        g = mech["squared_gap_long_minus_short"]
        types = {t: v for t, v in mech[hl].items() if isinstance(v, dict) and "roots" in v}
        best = "; ".join(f"{t}: {np.argmax(v['mean_label_by_candidate'])}" for t, v in types.items())
        lines.append(f"{label} & {g['mean']:.3g} [{g['lo']:.3g}, {g['hi']:.3g}] & {mech[hl]['value_of_world_specific_information_ceiling']:.3g} & {mech[hl]['population_prior_choice_value']:.3g} & {best} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "mechanism_table.tex").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("specs", nargs="+")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    items = [load(s) for s in a.specs]
    forest(items, a.out)
    for label, s in items:
        isr_vs_cost(label, s, a.out)
    table_main(items, a.out); table_mechanism(items, a.out)
    print("written:", sorted(p.name for p in a.out.iterdir()))


if __name__ == "__main__":
    main()
