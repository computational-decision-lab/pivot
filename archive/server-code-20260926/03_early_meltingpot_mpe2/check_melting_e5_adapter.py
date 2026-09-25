"""Local contract/equivalence checks; no environments, training, or API calls.

Run with colin_pivot/src, colin_pivot, and colin_pivot_cloud on PYTHONPATH:
    python -B colin_pivot_cloud/check_melting_e5_adapter.py
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.v9 import e5c_efficiency as author_e5
from pivot.acquisition.pivot_voi import BayesianLinearDeltaPosterior

import melting_e5_adapter as adapter


class QueryOnlyCandidate(Mapping):
    """Fail if selection even attempts to inspect a hidden outcome field."""
    def __init__(self, row):
        self.row = row

    def __getitem__(self, key):
        if key not in {"transition_id", "delta_proxy", "features", "hf_query_cost"}:
            raise AssertionError(f"selection accessed forbidden outcome field: {key}")
        return self.row[key]

    def __iter__(self):
        raise AssertionError("candidate wholesale copying would expose hidden outcomes")

    def __len__(self):
        return 4


def fixture(seed=14, count=8):
    rng = np.random.default_rng(seed)
    features = np.c_[np.ones(28), rng.normal(size=(28, 3))]
    targets = features @ np.array([0.2, -0.8, 0.6, 0.3]) + rng.normal(0, 0.5, 28)
    posterior = BayesianLinearDeltaPosterior(noise_variance=0.7).fit(features, targets)
    matrix = np.c_[np.ones(count), rng.normal(size=(count, 3))]
    proxy = rng.normal(size=count)
    # Nontrivial corrections and heterogeneous predeclared costs exercise
    # EVSI-per-cost and all methods' unqueried-candidate promotion rules.
    actual = proxy + rng.normal(0, 2, count)
    rows = [{"transition_id": str(j), "delta_proxy": float(proxy[j]),
             "features": matrix[j].tolist(), "hf_query_cost": float(100 + 23 * j),
             "delta_true": float(actual[j])} for j in range(count)]
    return posterior, rows


class E5AdapterChecks(unittest.TestCase):
    def test_author_equivalence_over_methods_budgets_and_panel_sizes(self):
        cases = 0
        changed_by_query = 0
        for fixture_seed in (7, 19, 43):
            for count in (4, 8):
                posterior, rows = fixture(fixture_seed, count)
                frozen_mean = posterior.mean.copy()
                frozen_covariance = posterior.covariance.copy()
                original_rows = copy.deepcopy(rows)
                for method in adapter.SUPPORTED_METHODS:
                    for budget in adapter.BUDGETS:
                        with self.subTest(seed=fixture_seed, K=count, method=method, budget=budget):
                            config = {"statistics": adapter.DEFAULT_STATISTICS}
                            ids = author_e5._select(method, rows, posterior, min(budget, count), fixture_seed, config)
                            expected_id, full_information_score = author_e5._select_outcome(rows, ids, method, posterior)
                            called = []

                            def query(candidate):
                                identifier = candidate["transition_id"]
                                called.append(identifier)
                                return {"delta": rows[int(identifier)]["delta_true"],
                                        "total_env_steps": candidate["hf_query_cost"]}

                            result = adapter.run_e5_decision(
                                [QueryOnlyCandidate(r) for r in rows], posterior, query,
                                method=method, budget=budget, seed=fixture_seed,
                            )
                            self.assertEqual(result["queried_ids"], ids)
                            self.assertEqual(called, ids)
                            self.assertEqual(result["selected_candidate_id"], expected_id)
                            self.assertEqual(result["query_packages_used"], len(ids))
                            self.assertEqual(result["query_cost"], sum(rows[int(i)]["hf_query_cost"] for i in ids))
                            if expected_id in ids:
                                self.assertEqual(result["selected_query_delta"], full_information_score)
                            else:
                                self.assertIsNone(result["selected_query_delta"])
                            self.assertNotIn("delta_true", result)
                            self.assertNotIn("audit_gain", result)
                            json.dumps(result, allow_nan=False)
                            no_queries_id, _ = author_e5._select_outcome(rows, [], method, posterior)
                            changed_by_query += int(expected_id != no_queries_id)
                            cases += 1
                np.testing.assert_array_equal(posterior.mean, frozen_mean)
                np.testing.assert_array_equal(posterior.covariance, frozen_covariance)
                self.assertEqual(rows, original_rows)
        self.assertEqual(cases, 270)
        self.assertGreater(changed_by_query, 10, "fixtures must exercise substantive query-dependent promotions")

    def test_no_early_stop_even_when_posterior_selection_is_decisive(self):
        posterior = BayesianLinearDeltaPosterior(
            mean=np.array([0.0]), covariance=np.zeros((1, 1)), noise_variance=0.1
        )
        rows = [{"transition_id": str(i), "delta_proxy": 100.0 if i == 0 else -100.0,
                 "features": [1.0], "hf_query_cost": 1.0} for i in range(8)]
        called = []
        result = adapter.run_e5_decision(
            rows, posterior, lambda row: called.append(row["transition_id"]) or 0.0,
            method="pivot_voi", budget=2, seed=123,
        )
        self.assertEqual(len(called), 2)
        self.assertEqual(result["hf_budget"], 2)
        self.assertIsNone(result["stop_reason"])

    def test_zero_budget_random_and_proxy_keep_proxy_but_others_keep_correction(self):
        posterior = BayesianLinearDeltaPosterior(
            mean=np.array([0.0, 10.0]), covariance=np.eye(2), noise_variance=1.0
        )
        rows = [
            {"transition_id": "0", "delta_proxy": 2.0, "features": [1.0, 0.0], "hf_query_cost": 1.0},
            {"transition_id": "1", "delta_proxy": 1.0, "features": [1.0, 1.0], "hf_query_cost": 1.0},
        ]
        def never_query(_):
            self.fail("zero-budget selection queried a response")
        for method in adapter.METHODS:
            result = adapter.run_e5_decision(rows, posterior, never_query, method=method, budget=0, seed=1)
            expected = "0" if method in {"random_hf", "proxy_only"} else "1"
            self.assertEqual(result["selected_candidate_id"], expected)
            self.assertEqual(result["query_cost"], 0)
        result = adapter.run_e5_decision(rows, posterior, never_query, method="proxy_only", budget=8, seed=1)
        self.assertEqual(result["query_packages_used"], 0)

    def test_query_callback_cannot_mutate_posterior_inputs(self):
        posterior, rows = fixture()
        original_rows = copy.deepcopy(rows)
        def query(row):
            row["delta_proxy"] = 999.0
            row["features"][0] = 999.0
            return 0.0
        adapter.run_e5_decision(rows, posterior, query, method="random_hf", budget=4, seed=7)
        self.assertEqual(rows, original_rows)

    def test_calibration_matches_author_fit_and_payload_round_trips(self):
        rng = np.random.default_rng(86)
        raw = rng.normal(size=(24, 8))
        raw[:, 4] = 9.0  # constant footprint must not cause division by zero
        corrections = rng.normal(size=24)
        calibration = [{"features": f.tolist(), "target_correction": float(y), "noise": 999.0}
                       for f, y in zip(raw, corrections)]
        posterior, payload = adapter.fit_external_posterior(calibration)
        x = np.c_[np.ones(24), (raw[:, 1:] - payload["feature_mean"]) / payload["feature_scale"]]
        converted = [{"features": f.tolist(), "delta_proxy": float(p), "delta_true": float(p) + float(y)}
                     for f, p, y in zip(x, raw[:, 0], corrections)]
        expected = author_e5._fit_posterior(converted)
        np.testing.assert_array_equal(posterior.mean, expected.mean)
        np.testing.assert_array_equal(posterior.covariance, expected.covariance)
        self.assertEqual(posterior.noise_variance, expected.noise_variance)
        restored = adapter.posterior_from_payload(json.loads(json.dumps(payload)))
        np.testing.assert_array_equal(restored.mean, posterior.mean)
        np.testing.assert_array_equal(restored.covariance, posterior.covariance)
        old_replay_payload = {"mean": payload["mean"], "covariance": payload["covariance"],
                              "query_mean_noise": 0.321, "training_mean_noise": 0.123,
                              "feature_mean": payload["feature_mean"], "feature_scale": payload["feature_scale"]}
        old = adapter.posterior_from_payload(old_replay_payload)
        self.assertEqual(old.noise_variance, 0.321)
        for count in (4, 8):
            candidates = adapter.make_candidates({"rows": raw[:count].tolist()}, payload, 123.0)
            self.assertEqual(len(candidates), count)
            np.testing.assert_array_equal([r["features"] for r in candidates], x[:count])
            np.testing.assert_array_equal([r["delta_proxy"] for r in candidates], raw[:count, 0])

    def test_file_reader_reads_only_allocated_selection_files_and_preserves_provenance(self):
        posterior, rows = fixture()
        for row in rows:
            row["hf_query_cost"] = 20.0
        queried = author_e5._select("random_hf", rows, posterior, 1, 21, {"statistics": adapter.DEFAULT_STATISTICS})
        candidate = int(queried[0])
        with tempfile.TemporaryDirectory(prefix="melting-e5-check-") as tmp:
            root = Path(tmp)
            # All unqueried response files are deliberately absent.
            for r in range(2):
                target = root / "selection" / f"candidate_{candidate}" / f"replicate_{r}" / "paired.json"
                target.parent.mkdir(parents=True)
                target.write_text(json.dumps({
                    "delta": float(r + 1), "candidate": candidate, "replicate": r,
                    "training_seed": 100 + r, "evaluation_seeds": [200 + r], "total_env_steps": 10,
                    "old": [{"seed": 200 + r, "focal_return": 1.0}],
                    "new": [{"seed": 200 + r, "focal_return": float(2 + r)}],
                }))
            (root / "audit").mkdir()
            (root / "audit" / "poison.json").write_text("not valid JSON")
            (root / "summary.json").write_text("do not read: historical audit-containing summary")
            real_read = Path.read_bytes
            accesses = []
            def guard(path):
                self.assertIn("selection", path.parts)
                self.assertNotIn("audit", path.parts)
                accesses.append(path)
                return real_read(path)
            with patch.object(Path, "read_bytes", guard):
                result = adapter.run_e5_decision(
                    rows, posterior, adapter.selection_query(root, 2),
                    method="random_hf", budget=1, seed=21,
                )
            self.assertEqual(len(accesses), 2)
            self.assertEqual(result["query_ledger"][0]["delta"], 1.5)
            self.assertEqual(result["query_ledger"][0]["training_seeds"], [100, 101])
            self.assertEqual(result["query_ledger"][0]["evaluation_seeds"], [200, 201])
            self.assertEqual(len(result["query_ledger"][0]["input_files"]), 2)

    def test_audit_stream_and_selection_symlink_are_rejected(self):
        posterior, rows = fixture()
        with self.assertRaisesRegex(ValueError, "only selection"):
            adapter.run_e5_decision(rows, posterior, lambda _: {"delta": 1.0, "stream": "audit"},
                                    method="random_hf", budget=1, seed=1)
        with tempfile.TemporaryDirectory(prefix="melting-e5-boundary-") as tmp:
            root = Path(tmp)
            (root / "audit").mkdir()
            (root / "selection").symlink_to(root / "audit", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "escaped"):
                adapter.selection_query(root)({"transition_id": "0"})

    def test_declared_cost_and_pairing_fail_closed(self):
        posterior, rows = fixture()
        with self.assertRaisesRegex(ValueError, "cost differs"):
            adapter.run_e5_decision(rows, posterior, lambda _: {"delta": 1.0, "total_env_steps": 1},
                                    method="random_hf", budget=1, seed=1)
        with tempfile.TemporaryDirectory(prefix="melting-e5-pairing-") as tmp:
            root = Path(tmp)
            path = root / "selection" / "candidate_0" / "replicate_0" / "paired.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"delta": 1.0, "total_env_steps": 20,
                                        "old": [{"seed": 1, "focal_return": 0}],
                                        "new": [{"seed": 2, "focal_return": 1}]}))
            with self.assertRaisesRegex(ValueError, "not paired"):
                adapter.selection_query(root)({"transition_id": "0"})

    def test_invalid_inputs_are_rejected_before_query(self):
        posterior, rows = fixture()
        def never(_):
            self.fail("invalid input caused a query")
        for budget in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                adapter.run_e5_decision(rows, posterior, never, method="random_hf", budget=budget, seed=1)
        for corrupt in ("duplicate", "nonfinite", "dimension"):
            bad = copy.deepcopy(rows)
            if corrupt == "duplicate":
                bad[1]["transition_id"] = bad[0]["transition_id"]
            elif corrupt == "nonfinite":
                bad[0]["delta_proxy"] = float("nan")
            else:
                bad[0]["features"].append(1.0)
            with self.assertRaises(ValueError):
                adapter.run_e5_decision(bad, posterior, never, method="random_hf", budget=1, seed=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
