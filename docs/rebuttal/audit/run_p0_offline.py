#!/usr/bin/env python3
"""Run reproducible offline P0 analyses on the frozen Highway redesign rows.

This script never calls HighwayEnv. It uses the saved post-decision truth audit
as a fixed evaluation reference and therefore reports counterfactual selector
replays, not a new physical rollout. Calibration bootstrap resamples only the
40 calibration roots; test proxy and truth rows remain fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _configure_imports(source_root: Path) -> None:
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(source_root / "src"))


def _load_rows(base: Path, name: str) -> list[dict[str, Any]]:
    return list(json.loads((base / name).read_text())['rows'])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap_ci(values: list[float], seed: int, draws: int = 4000) -> list[float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return [float(value) for value in np.percentile(samples, [2.5, 97.5])]


def _replay_selector(
    proxy: list[dict[str, Any]],
    truth: dict[str, float],
    model: Any,
    budget: int,
    method: str,
) -> tuple[str, float, list[str]]:
    from experiments.revision.highway_b4_redesign import exact_kg_next
    from experiments.revision.highway_calibration import GaussianPosterior

    posterior = GaussianPosterior(proxy, model)
    queries: list[str] = []
    if method == 'PIVOT-KG':
        for _ in range(budget):
            candidate, _ = exact_kg_next(posterior)
            queries.append(candidate)
            posterior.observe(candidate, truth[candidate])
    elif method == 'Top-Proxy':
        ordered = sorted(proxy, key=lambda row: (-float(row['delta_proxy']), int(row['candidate_index'])))
        for row in ordered[:budget]:
            candidate = str(row['candidate_id'])
            queries.append(candidate)
            posterior.observe(candidate, truth[candidate])
    elif method == 'IVR-Trace':
        for _ in range(budget):
            available = [candidate for candidate in posterior.candidate_ids if candidate not in posterior.observed]
            scores: dict[str, float] = {}
            for candidate in available:
                index = posterior._index[candidate]
                denominator = float(posterior.covariance[index, index] + posterior.observation_variance[index])
                scores[candidate] = float(np.dot(posterior.covariance[:, index], posterior.covariance[:, index]) / denominator) if denominator > 0 else 0.0
            candidate = max(available, key=lambda value: (scores[value], -posterior._index[value]))
            queries.append(candidate)
            posterior.observe(candidate, truth[candidate])
    else:
        raise ValueError(f'unknown method: {method}')
    selected = posterior.select()
    best = max([0.0, *truth.values()])
    selected_value = 0.0 if selected == 'incumbent' else truth[selected]
    return selected, float(best - selected_value), queries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=200)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'output exists; choose a new directory: {args.output}')
    args.output.mkdir(parents=True)
    _configure_imports(args.source)
    from experiments.revision.highway_b4_redesign import uniform_expected_selection
    from experiments.revision.highway_calibration import CalibrationModel, fit_calibration_model

    base = args.evidence
    calibration_proxy: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_rows(base, 'calibration-proxy-rows.json'):
        calibration_proxy[int(row['seed'])].append(row)
    calibration_truth: dict[int, dict[str, float]] = defaultdict(dict)
    for row in _load_rows(base, 'calibration-truth.json'):
        calibration_truth[int(row['seed'])][str(row['candidate_id'])] = float(row['paired_delta'])
    model = CalibrationModel.from_record(json.loads((base / 'calibration-model.json').read_text()))

    proxy_by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_rows(base, 'proxy-rows.json'):
        proxy_by_seed[int(row['seed'])].append(row)
    truth_by_seed: dict[int, dict[str, float]] = defaultdict(dict)
    for row in _load_rows(base, 'truth-audit.json'):
        if row['candidate_id'] != 'incumbent':
            truth_by_seed[int(row['seed'])][str(row['candidate_id'])] = float(row['paired_delta'])

    methods = ('PIVOT-KG', 'IVR-Trace', 'Top-Proxy')
    root_rows: list[dict[str, Any]] = []
    for seed in sorted(proxy_by_seed):
        proxy = proxy_by_seed[seed]
        truth = truth_by_seed[seed]
        for budget in (2, 4):
            uniform = uniform_expected_selection(proxy, model, truth, budget)
            root_rows.append({
                'seed': seed,
                'budget': budget,
                'method': 'Uniform-Exact',
                'selected_candidate': 'expected_policy',
                'queries': [],
                'ISR': float(uniform['mean_ISR']),
            })
            for method in methods:
                selected, isr, queries = _replay_selector(proxy, truth, model, budget, method)
                root_rows.append({
                    'seed': seed,
                    'budget': budget,
                    'method': method,
                    'selected_candidate': selected,
                    'queries': queries,
                    'ISR': isr,
                })

    base_summary: dict[str, Any] = {
        'analysis_kind': 'offline_counterfactual_replay',
        'physical_simulator_called': False,
        'truth_source': 'saved post-decision truth-audit.json',
        'calibration_source': 'saved calibration rows and calibration-model.json',
        'methods': list(methods) + ['Uniform-Exact'],
        'budgets': [2, 4],
        'roots': len(proxy_by_seed),
        'rows': len(root_rows),
        'comparisons': {},
        'formal_lucb': {
            'candidate_count': len(model.candidate_ids),
            'primary_budget': 4,
            'status': 'initialization_infeasible',
            'reason': 'formal paired LUCB initialization requires one pull per candidate, but B=4 < K=8',
        },
    }
    for budget in (2, 4):
        by_method = {
            method: [row for row in root_rows if row['budget'] == budget and row['method'] == method]
            for method in (*methods, 'Uniform-Exact')
        }
        pivot_isr = {int(row['seed']): float(row['ISR']) for row in by_method['PIVOT-KG']}
        for method in ('IVR-Trace', 'Top-Proxy', 'Uniform-Exact'):
            values = [float(row['ISR']) - pivot_isr[int(row['seed'])] for row in by_method[method]]
            base_summary['comparisons'][f'{method}_minus_PIVOT_B{budget}'] = {
                'mean_ISR_difference': float(np.mean(values)),
                'ci95': _bootstrap_ci(values, seed=20260927 + budget),
                'positive_favors': 'PIVOT-KG' if method != 'Uniform-Exact' else 'PIVOT-KG',
                'n': len(values),
                'positive': int(sum(value > 1e-12 for value in values)),
                'negative': int(sum(value < -1e-12 for value in values)),
                'zero': int(sum(abs(value) <= 1e-12 for value in values)),
            }

    rng = np.random.default_rng(20261109)
    bootstrap_rows: list[dict[str, Any]] = []
    calibration_seeds = np.asarray(sorted(calibration_proxy), dtype=int)
    for replicate in range(args.replicates):
        sampled = rng.choice(calibration_seeds, size=len(calibration_seeds), replace=True)
        proxy_boot = {index: calibration_proxy[int(seed)] for index, seed in enumerate(sampled)}
        truth_boot = {index: calibration_truth[int(seed)] for index, seed in enumerate(sampled)}
        boot_model = fit_calibration_model(proxy_boot, truth_boot, covariance_shrinkage=0.5)
        row: dict[str, Any] = {'replicate': replicate, 'calibration_roots': len(sampled)}
        for budget in (2, 4):
            contrasts: list[float] = []
            flips: list[int] = []
            for seed in sorted(proxy_by_seed):
                proxy = proxy_by_seed[seed]
                truth = truth_by_seed[seed]
                uniform = uniform_expected_selection(proxy, boot_model, truth, budget)
                selected, pivot_isr, _ = _replay_selector(proxy, truth, boot_model, budget, 'PIVOT-KG')
                original = next(item for item in root_rows if item['seed'] == seed and item['budget'] == budget and item['method'] == 'PIVOT-KG')
                contrasts.append(float(uniform['mean_ISR']) - pivot_isr)
                flips.append(int(selected != original['selected_candidate']))
            row[f'B{budget}'] = {
                'mean_contrast': float(np.mean(contrasts)),
                'positive': bool(np.mean(contrasts) > 0),
                'selection_flip_rate_vs_original': float(np.mean(flips)),
            }
        bootstrap_rows.append(row)
    for budget in (2, 4):
        values = [float(row[f'B{budget}']['mean_contrast']) for row in bootstrap_rows]
        flips = [float(row[f'B{budget}']['selection_flip_rate_vs_original']) for row in bootstrap_rows]
        base_summary[f'calibration_bootstrap_B{budget}'] = {
            'replicates': len(values),
            'mean_of_fitted_contrasts': float(np.mean(values)),
            'ci95_over_calibration_fits': [float(value) for value in np.percentile(values, [2.5, 97.5])],
            'sign_stability': float(np.mean(np.asarray(values) > 0)),
            'mean_selection_flip_rate': float(np.mean(flips)),
            'max_selection_flip_rate': float(np.max(flips)),
        }

    config = {
        'replicates': args.replicates,
        'bootstrap_seed': 20261109,
        'covariance_shrinkage': 0.5,
        'source_manifest_sha256': _sha256(args.source.parent / 'source-manifest.json'),
        'input_hashes': {name: _sha256(base / name) for name in (
            'calibration-model.json', 'calibration-proxy-rows.json', 'calibration-truth.json',
            'proxy-rows.json', 'truth-audit.json', 'summary.json',
        )},
        'truth_budget_caveat': 'subset-ledger is not used for this replay; Uniform is recomputed directly from proxy/truth rows',
    }
    (args.output / 'config.json').write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
    (args.output / 'root-rows.json').write_text(json.dumps({'rows': root_rows}, indent=2, sort_keys=True) + '\n')
    (args.output / 'calibration-bootstrap.json').write_text(json.dumps({'rows': bootstrap_rows}, indent=2, sort_keys=True) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(base_summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(base_summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
