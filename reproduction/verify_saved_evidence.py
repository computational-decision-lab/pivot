"""Verify the supplied files and replay saved labels; no simulator is launched."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import numpy as np

def read(path):
    return json.loads(path.read_text())

def bootstrap(values, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()), lo=float(np.quantile(means, .025)),
                hi=float(np.quantile(means, .975)), n=len(values))

def verify(root):
    manifest = read(root / 'manifest.json')
    bad = [row['path'] for row in manifest['files']
           if hashlib.sha256((root / row['path']).read_bytes()).hexdigest() != row['sha256']]
    assert not bad, f'Evidence hash mismatches: {bad}'
    report = {'input_files': len(manifest['files']), 'input_hash_failures': bad}
    disjoint = tied = total = ratio_disagreements = 0
    for environment, horizon in [('leduc', '8'), ('kuhn', '8'), ('melting', '32')]:
        files = sorted((root / 'core' / environment / 'confirm').glob('seed_*/summary.json'))
        assert len(files) == 30
        for path in files:
            row = read(path)
            ids = sorted(row['proxy_deltas'], key=int)
            proxy = np.array([0.] + [row['proxy_deltas'][i] for i in ids])
            truth = np.array([0.] + [row['audit_gains'][horizon][i] for i in ids])
            pbest = np.isclose(proxy, proxy.max(), atol=1e-12, rtol=0)
            tbest = np.isclose(truth, truth.max(), atol=1e-12, rtol=0)
            b = np.flatnonzero(tbest)[np.argmax(proxy[tbest])]
            correction = truth - proxy
            ratio = np.max((correction[b] - correction[~tbest]) / (truth[b] - truth[~tbest]))
            total += 1
            disjoint += not np.any(pbest & tbest)
            tied += int(tbest.sum() > 1)
            ratio_disagreements += int(ratio > 1 + 1e-10)
    assert (total, disjoint, tied, ratio_disagreements) == (90, 51, 28, 51)
    report['core'] = dict(roots=total, disjoint_optimal_sets=int(disjoint),
                          tied_deployment_optima=tied, ratios_above_one=ratio_disagreements)

    roots = {int(p.parent.name[5:]): read(p) for p in (root / 'leduc_v4/confirm').glob('seed_*/summary.json')}
    rows = read(root / 'leduc_v4/analysis/scored_decisions.json')
    primary, stopping = {}, {}
    max_gain_error = max_regret_error = 0.
    for row in rows:
        labels = np.array([roots[row['root']]['audit_gains'][str(row['adaptation'])][str(i)] for i in range(8)] + [0.])
        choices = row.get('selected_distribution', [row['selected']])
        gain = float(labels[choices].mean())
        max_gain_error = max(max_gain_error, abs(gain - row['gain']))
        max_regret_error = max(max_regret_error, abs(labels.max() - gain - row['isr']))
        if row['adaptation'] == 8 and row['cap'] == 16384 and row['calibration'] == 'exact':
            primary.setdefault(row['root'], {})[row['method']] = gain
            if row['method'] in ['pivot_kg_menu', 'pivot_kg_menu_stop']:
                stopping.setdefault(row['root'], {})[row['method']] = row
    contrast = float(np.mean([x['pivot_kg_menu'] - x['uniform_fixed'] for x in primary.values()]))
    expected = read(root / 'leduc_v4/analysis/summary.json')['hypotheses']['H_A_allocation_pivot_minus_expected_uniform']
    changed = sum(x['pivot_kg_menu']['selected'] != x['pivot_kg_menu_stop']['selected'] for x in stopping.values())
    repeated = sum(len(x['pivot_kg_menu']['queries']) > len(set(
        q.get('candidate', q.get('j')) if isinstance(q, dict) else q[0] if isinstance(q, list) else q
        for q in x['pivot_kg_menu']['queries'])) for x in stopping.values()) if all('queries' in x['pivot_kg_menu'] for x in stopping.values()) else None
    assert len(roots) == 30 and len(rows) == 3960 and max_gain_error < 1e-12 and max_regret_error < 1e-12
    assert abs(contrast - expected['mean']) < 1e-12 and changed == 0
    report['leduc_v4'] = dict(roots=len(roots), scored_rows=len(rows), max_gain_error=max_gain_error,
                              max_regret_error=max_regret_error, primary_gain_contrast=contrast,
                              unchanged_stopping_decisions=len(stopping), repeated_query_roots=repeated)

    primary = {}
    max_gain_error = max_regret_error = 0.
    count = 0
    for record in read(root / 'metadrive/confirmation/analysis/result.json')['root_results']:
        labels = np.asarray(record['audit_gain_estimates_incumbent_first'])
        for row in record['scored']:
            gain = float(labels[row.get('selected_distribution', [row['selected_idx']])].mean())
            max_gain_error = max(max_gain_error, abs(gain - row['selected_audit_gain']))
            max_regret_error = max(max_regret_error, abs(labels.max() - gain - row['empirical_oracle_regret']))
            count += 1
            if record['adaptation'] == 12 and row['budget_queries'] == 2:
                primary.setdefault(record['root'], {})[row['method']] = gain
    contrast = bootstrap([primary[r]['pivot_kg'] - primary[r]['uniform_v2_expected_100'] for r in sorted(primary)], 78131)
    expected = read(root / 'metadrive/confirmation/analysis/summary.json')['hypotheses']['primary_PIVOT_KG_minus_Uniform_expected_100_selected_gain']
    assert count == 2160 and max_gain_error < 1e-12 and max_regret_error < 1e-12
    assert all(abs(contrast[k] - expected[k]) < 1e-12 for k in ['mean', 'lo', 'hi'])
    report['metadrive'] = dict(roots=len(primary), scored_rows=count, max_gain_error=max_gain_error,
                               max_regret_error=max_regret_error, primary_gain_contrast=contrast)

    highway = [r for r in read(root / 'highway/redesign/rows.json')['rows'] if r['budget'] == 4]
    highway_mean = {key: float(np.mean([row[key] for row in highway])) for key in ['pivot_ISR', 'uniform_expected_ISR']}
    assert len(highway) == 120 and round(highway_mean['pivot_ISR'], 4) == .0055 and round(highway_mean['uniform_expected_ISR'], 4) == .0435
    report['highway_redesign'] = dict(roots=len(highway), budget=4, **highway_mean)

    full, excluded = [], 0
    with gzip.open(root / 'controlled/e5c/group_metrics.jsonl.gz', 'rt') as handle:
        for line in handle:
            row = json.loads(line)
            if row['method'] == 'all_hf':
                if row['queries'] == row['candidate_count']:
                    full.append(row['CISR'])
                else:
                    excluded += 1
    assert full and max(abs(x) for x in full) < 1e-12
    report['all_hf_reference'] = dict(fully_queried_rows=len(full), excluded_incomplete_rows=excluded,
                                     correct_reference=float(np.mean(full)))
    report['scope'] = 'Saved-label and artifact replay only. No simulator, retraining, independent preregistration timestamp, or human approval is certified.'
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.evidence)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))
