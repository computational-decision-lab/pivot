"""Replot the saved MetaDrive cohort; no simulation or selection is rerun."""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np

STEM = 'matched_posterior_selected_audit_gain'
METHODS = [
    ('pivot_kg', 'PIVOT-KG', '#0072B2', 'o', -0.15),
    ('uniform_v2_expected_100', 'Uniform', '#D55E00', 's', -0.05),
    ('ivr_v2', 'IVR', '#B2B8BE', 'D', 0.05),
    ('lucb_v2', 'LUCB', '#B2B8BE', '^', 0.15),
    ('no_hf_v2', 'Calibrated, no HF', '#92999F', 'P', -0.10),
    ('proxy_only', 'Proxy only', '#B2B8BE', 'x', 0.10),
]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def bootstrap(values):
    values = np.asarray(values, dtype=float)
    assert values.shape == (30,) and np.isfinite(values).all()
    rng = np.random.default_rng(78131)
    means = values[rng.integers(30, size=(10000, 30))].mean(axis=1)
    lo, hi = np.quantile(means, [.025, .975])
    return dict(mean=float(values.mean()), lo=float(lo), hi=float(hi))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.data_dir / 'summary.json'
    csv_path = args.data_dir / 'seed_results.csv'
    summary = json.loads(summary_path.read_text())
    assert summary['primary_adaptation'] == 12
    assert summary['primary_query_budget'] == 2
    with csv_path.open(newline='') as stream:
        source = list(csv.DictReader(stream))
    groups = {}
    for row in source:
        if int(row['adaptation']) != 12 or row['method'].startswith('author_'):
            continue
        key = (row['method'], int(row['budget_queries']))
        root = int(row['root'])
        assert root not in groups.setdefault(key, {})
        groups[key][root] = row
    roots = list(range(301, 331))
    stats, complete = {}, []
    for key, rows in sorted(groups.items()):
        assert sorted(rows) == roots
        stats[key] = bootstrap([float(rows[r]['selected_audit_gain']) for r in roots])
        archived = next(r for r in summary['method_table'] if
                        (r['method'], r['budget_queries'], r['adaptation']) == (*key, 12))
        assert all(abs(stats[key][k] - archived['selected_audit_gain'][k]) < 1e-11
                   for k in ['mean', 'lo', 'hi']), key
        complete.append(dict(method=key[0], budget_queries=key[1],
                             **stats[key], n_roots=30,
                             mean_hf_episode_cost=archived['mean_hf_episode_cost']))
    assert len(complete) == 24
    contrasts = []
    for budget in [1, 2, 4]:
        values = [float(groups['pivot_kg', budget][r]['selected_audit_gain']) -
                  float(groups['uniform_v2_expected_100', budget][r]['selected_audit_gain'])
                  for r in roots]
        contrasts.append(dict(budget_queries=budget, **bootstrap(values)))
    primary = contrasts[1]
    archived = summary['hypotheses']['primary_PIVOT_KG_minus_Uniform_expected_100_selected_gain']
    assert all(abs(primary[k] - archived[k]) < 1e-11 for k in ['mean', 'lo', 'hi'])
    assert primary['lo'] < 0 < primary['hi']
    for method in ['no_hf_v2', 'proxy_only']:
        for root in roots:
            assert len({groups[method, b][root]['selected_audit_gain'] for b in [1, 2, 4]}) == 1
            assert all(float(groups[method, b][root]['hf_episode_cost']) == 0 for b in [1, 2, 4])

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .65, 'axes.edgecolor': '#747474',
                         'text.color': '#222222', 'axes.labelcolor': '#222222'})
    fig = plt.figure(figsize=(7.1, 3.6), facecolor='white')
    left = fig.add_axes([.090, .345, .405, .56])
    right = fig.add_axes([.645, .345, .330, .56])
    left.set_title('(a) Selected deployment gain', loc='left', fontsize=10, pad=13)
    right.set_title('(b) Paired contrast', loc='left', fontsize=10, pad=13)
    # The categorical x=2 position marks the prespecified two-query comparison.
    left.axvline(2, color='#C8C8C8', linestyle=(0, (3, 3)), linewidth=.7, zorder=.5)
    handles, reference_handles, no_hf_handles = [], [], []
    styles = [
        dict(linewidth=1.45, elinewidth=1.25, markersize=5.8, capthick=1.0, alpha=1.0),
        dict(linewidth=1.05, elinewidth=.95, markersize=5.1, capthick=.8, alpha=.92),
        dict(linewidth=.45, elinewidth=.45, markersize=3.5, capthick=.45, alpha=.58),
        dict(linewidth=.45, elinewidth=.45, markersize=3.5, capthick=.45, alpha=.58),
        dict(linewidth=.65, elinewidth=.65, markersize=4.5, capthick=.6, alpha=.8),
        dict(linewidth=.65, elinewidth=.65, markersize=4.5, capthick=.6, alpha=.8),
    ]
    for i, (method, label, color, marker, offset) in enumerate(METHODS):
        is_primary = i < 2
        style = styles[i]
        budgets = [1, 2, 4] if i < 4 else [1]
        x = np.array([1, 2, 3] if i < 4 else [0], dtype=float) + offset
        rows = [stats[method, b] for b in budgets]
        means = np.array([r['mean'] for r in rows])
        err = np.array([[r['mean'] - r['lo'] for r in rows],
                        [r['hi'] - r['mean'] for r in rows]])
        if i < 4:
            left.plot(x, means, color=color, linewidth=style['linewidth'],
                      alpha=style['alpha'], zorder=2 if is_primary else 1)
        left.errorbar(x, means, yerr=err, fmt=marker, linestyle='none',
                      color=color, ecolor=color, elinewidth=style['elinewidth'],
                      capsize=2.4 if is_primary else 2, capthick=style['capthick'],
                      markersize=style['markersize'],
                      markeredgewidth=.85 if is_primary else .65, alpha=style['alpha'],
                      zorder=5 if is_primary else 3, label=label)
        handle = Line2D([], [], color=color, marker=marker, linestyle='none',
                        markersize=style['markersize'], alpha=style['alpha'], label=label)
        (handles if is_primary else reference_handles if i < 4 else no_hf_handles).append(handle)
    left.set(xlim=(-.42, 3.42), ylim=(-.5, 16),
             xticks=[0, 1, 2, 3], xticklabels=['No-HF\nbaselines', '1', '2', '4'],
             xlabel='HF candidate-query budget B',
             ylabel='Gain (native return units)')
    left.yaxis.set_major_locator(MultipleLocator(5))
    left.set_yticks([0, 5, 10, 15])
    left.grid(axis='y', color='#E4E4E4', linewidth=.55, zorder=0)
    left.axhline(0, color='#999999', linewidth=.65, zorder=1)
    left.get_xticklabels()[2].set_weight('bold')
    left.get_xticklabels()[0].set_fontsize(7.4)
    left.tick_params(axis='x', labelcolor='#222222')
    left.get_xticklabels()[0].set_color('#777777')

    right.axvline(0, color='#888888', linestyle=(0, (3, 3)), linewidth=.8, zorder=1)
    ys = [2.65, 1.40, .15]
    for contrast, y in zip(contrasts, ys):
        focus = contrast['budget_queries'] == 2
        color = '#0072B2' if focus else '#C0C5CA'
        right.errorbar(contrast['mean'], y,
                       xerr=[[contrast['mean'] - contrast['lo']],
                             [contrast['hi'] - contrast['mean']]],
                       fmt='o', color=color, markersize=6 if focus else 4.8,
                       elinewidth=1.6 if focus else .9, capsize=3,
                       capthick=1.1 if focus else .8, zorder=4)
    right.set(ylim=(-.5, 3.3), xlim=(-4, 7),
              yticks=ys, yticklabels=['B=1', 'B=2', 'B=4'],
              xticks=[-4, 0, 4])
    right.set_xlabel('Selected deployment gain difference\n(PIVOT-KG − Uniform)', fontsize=8.7, labelpad=6)
    right.spines['left'].set_visible(False)
    right.tick_params(axis='y', length=0, pad=7)
    right.tick_params(axis='y', labelcolor='#222222')
    for label, budget in zip(right.get_yticklabels(), [1, 2, 4]):
        label.set_color('#0072B2' if budget == 2 else '#92999F')
        if budget == 2:
            label.set_weight('bold')
    primary_text = f"{primary['mean']:+.2f} [{primary['lo']:.2f}, {primary['hi']:.2f}]"
    annotation_background = dict(facecolor='white', edgecolor='none', pad=.8)
    right.text(.5, .565, 'Primary comparison: B=2', transform=right.transAxes,
               ha='center', fontsize=8, color='#444444', bbox=annotation_background)
    right.text(.5, .395, primary_text.replace('-', '−'), transform=right.transAxes,
               ha='center', fontsize=9.7, weight='bold', color='#0072B2',
               bbox=annotation_background)
    right.text(.5, .285, 'unresolved (95% CI crosses 0)', transform=right.transAxes,
               ha='center', fontsize=7.2, color='#666666', bbox=annotation_background)
    legend = fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.29, .10),
                        ncol=2, frameon=False, fontsize=8.7, handletextpad=.4,
                        columnspacing=1.35, labelspacing=.5)
    # Secondary keys are subdued and separate from the two-method main legend.
    fig.legend(handles=reference_handles, loc='lower center', bbox_to_anchor=(.26, .035),
               ncol=2, frameon=False, fontsize=7.2, labelcolor='#777777',
               handletextpad=.3, columnspacing=.9)
    fig.legend(handles=no_hf_handles, loc='lower center', bbox_to_anchor=(.67, .035),
               ncol=2, frameon=False, fontsize=7.2, labelcolor='#777777',
               handletextpad=.3, columnspacing=.9)
    assert len(legend.get_texts()) == 2
    assert all(t.get_text().strip() for t in legend.get_texts())
    assert not left.patches and not right.patches
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    outside = []
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text().strip():
            box = text.get_window_extent(renderer)
            if box.x0 < bounds.x0-1 or box.y0 < bounds.y0-1 or box.x1 > bounds.x1+1 or box.y1 > bounds.y1+1:
                outside.append(text.get_text())
    assert not outside, outside
    for suffix in ['pdf', 'png', 'svg']:
        target = args.output_dir / f'{STEM}.{suffix}'
        metadata = {'Creator': 'PIVOT saved-evidence figure', 'CreationDate': None, 'ModDate': None} if suffix == 'pdf' else None
        fig.savefig(target, dpi=220, metadata=metadata)
    plt.close(fig)
    table = args.output_dir / 'all_eight_methods_long_response.csv'
    with table.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(complete[0]))
        writer.writeheader(); writer.writerows(complete)
    record = dict(inputs={p.name: sha(p) for p in [summary_path, csv_path]},
                  script_sha256=sha(Path(__file__)),
                  root_ids=roots, adaptation_profiles=12, bootstrap_draws=10000,
                  bootstrap_seed=78131, interval='percentile 2.5% and 97.5%; paired roots on right',
                  plotted_methods=[dict(id=m[0], label=m[1]) for m in METHODS],
                  omitted_from_display=['uniform_v2', 'global_voi_heuristic'],
                  omission_reason='One Uniform comparator; four secondary controls remain gray. All eight rules are retained in the adjacent table and original evidence.',
                  zero_hf_baselines_displayed_at_budget=0, marginal_rows_verified=len(complete),
                  contrasts=contrasts, primary_unresolved=True,
                  text_within_figure_bounds=not outside,
                  visual_style=dict(primary_budget_guide='light gray dashed line at categorical B=2',
                                    hierarchy=styles, overall_title=False,
                                    main_legend=['PIVOT-KG', 'Uniform'],
                                    secondary_references='gray; separate subdued keys',
                                    no_hf_axis_label='No-HF baselines',
                                    panel_b_budget_labels='B=2 blue; B=1 and B=4 gray',
                                    panel_b_xlabel=right.get_xlabel()),
                  outputs={p.name: sha(p) for p in args.output_dir.glob(STEM + '.*') if p.suffix in ['.pdf','.png','.svg']})
    (args.output_dir / (STEM + '.audit.json')).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'verified_method_rows':len(complete), 'paired_contrasts':contrasts,
                      'main_legend_entries':2, 'secondary_reference_entries':4,
                      'out_of_bounds_text':outside}, indent=2))

if __name__ == '__main__':
    main()
