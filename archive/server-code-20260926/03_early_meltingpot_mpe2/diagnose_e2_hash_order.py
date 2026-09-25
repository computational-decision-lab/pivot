"""Check E2C bootstrap sensitivity without modifying author source or outcomes."""
import gzip
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from pivot.v9.statistics import bootstrap_mean_ci, improvement_metrics

base = Path(__file__).resolve().parent
with gzip.open(base / 'remaining_v9/e2c_confirmatory/transition_rows.jsonl.gz', 'rt') as f:
    rows = [json.loads(line) for line in f]
groups = defaultdict(list)
for row in rows:
    groups[(str(row['environment_id']), float(row['response_strength']),
            str(row['operator_family']), int(row['seed']), float(row['operator_shift']))].append(row)
prefixes = {(a, b, c, d) for a, b, c, d, _ in groups}
lo = min(float(row['operator_shift']) for row in rows)
hi = max(float(row['operator_shift']) for row in rows)
def evaluate(order):
    effects = [float(improvement_metrics(groups[(*p, hi)])['IDE'] or 0.0)
               - float(improvement_metrics(groups[(*p, lo)])['IDE'] or 0.0) for p in order]
    low, high = bootstrap_mean_ci(effects, seed=202608261, draws=10000)
    return {'effect_count': len(effects), 'effect_mean': float(np.mean(effects)),
            'ci_low': low, 'ci_high': high,
            'sorted_effects_sha256': hashlib.sha256(np.array(sorted(effects), dtype='<f8').tobytes()).hexdigest()}
print(json.dumps({'python_hash_seed': os.environ.get('PYTHONHASHSEED'),
                  'author_set_order': evaluate(prefixes),
                  'sorted_prefix_control': evaluate(sorted(prefixes))}))
