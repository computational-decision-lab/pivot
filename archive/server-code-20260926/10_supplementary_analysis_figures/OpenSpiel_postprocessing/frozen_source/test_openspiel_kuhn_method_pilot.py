"""Contract tests for the sealed OpenSpiel method pilot."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
AUTHOR_ROOT = REPOSITORY_ROOT / "colin_pivot"
AUTHOR_SRC = AUTHOR_ROOT / "src"
for search_path in (AUTHOR_ROOT, AUTHOR_SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

try:
    import openspiel_kuhn_method_pilot as pilot
except ModuleNotFoundError:
    from colin_pivot_cloud.openspiel_fallback import openspiel_kuhn_method_pilot as pilot


class _AuthorPosteriorStub:
    def __init__(self, predictive_variances: np.ndarray):
        self._variances = np.asarray(predictive_variances, dtype=float)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.zeros(len(features), dtype=float)

    def predictive_variance(
        self, features: np.ndarray, include_observation: bool = True
    ) -> np.ndarray:
        del include_observation
        return np.asarray(features, dtype=float) @ self._variances


class OpenSpielMethodPilotUnitTests(unittest.TestCase):
    def test_root_beta_is_order_independent(self) -> None:
        forward = {root: pilot._root_beta(root) for root in (71000, 71001, 71099)}
        reverse = {root: pilot._root_beta(root) for root in (71099, 71001, 71000)}
        self.assertEqual(forward, reverse)
        self.assertTrue(all(pilot.BETA_MIN <= value <= pilot.BETA_MAX for value in forward.values()))

    def test_author_global_voi_ranking_matches_repository_entry_point(self) -> None:
        from experiments.v9 import e5c_efficiency as author_e5

        proxies = np.asarray([0.40, -0.10, 0.80, 0.05], dtype=float)
        standard_deviations = np.asarray([0.20, 0.35, 0.90, 0.22], dtype=float)
        features = np.eye(len(proxies), dtype=float)
        candidates = [
            {
                "transition_id": f"candidate_{index}",
                "delta_proxy": float(proxies[index]),
                "features": features[index].tolist(),
            }
            for index in range(len(proxies))
        ]
        author = author_e5._select(
            "global_voi",
            candidates,
            _AuthorPosteriorStub(standard_deviations**2),
            budget=len(candidates),
            seed=0,
            config={},
        )
        ours = pilot._author_global_voi_ranking(
            proxy=np.concatenate([[0.0], proxies]),
            covariance=np.diag(np.concatenate([[0.0], standard_deviations**2])),
            noise=np.zeros(len(proxies) + 1),
        )
        ours_ids = [f"candidate_{index - 1}" for index in ours]
        self.assertEqual(ours_ids, author)

    def test_budget_zero_and_query_contracts(self) -> None:
        mean = np.linspace(0.0, 0.20, len(pilot.ALPHAS))
        covariance = np.diag(np.linspace(0.0, 0.4, len(pilot.ALPHAS)))
        noise = np.asarray([0.0] + [0.2] * (len(pilot.ALPHAS) - 1), dtype=float)
        bank = {
            pilot._candidate_id(index): {"observation": -0.1 + 0.05 * index}
            for index in range(1, len(pilot.ALPHAS))
        }
        zero_selected = []
        for method in ("pivot_cg_v2", "uniform_random", "global_ivr", "posterior_lucb"):
            result = pilot._run_active_method(
                method=method,
                mean=mean.copy(),
                covariance=covariance.copy(),
                noise=noise.copy(),
                bank=bank,
                budget=0,
                seed=7,
            )
            zero_selected.append(result["selected_id"])
            self.assertEqual(result["queried_ids"], [])
            self.assertEqual(result["nominal_paired_hands"], 0)
        self.assertEqual(len(set(zero_selected)), 1)

        result = pilot._run_active_method(
            method="uniform_random",
            mean=mean.copy(),
            covariance=covariance.copy(),
            noise=noise.copy(),
            bank=bank,
            budget=128,
            seed=9,
        )
        self.assertEqual(len(result["queried_ids"]), 2)
        self.assertNotIn("incumbent", result["queried_ids"])
        self.assertEqual(len(result["queried_ids"]), len(set(result["queried_ids"])))
        self.assertEqual(result["nominal_paired_hands"], 128)
        self.assertEqual(result["physical_profile_hands"], 256)


class OpenSpielMethodPilotEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory(prefix="openspiel-pilot-test-")
        cls.output = Path(cls._temp.name)
        pilot.calibrate(cls.output, roots=tuple(range(62000, 62008)))
        pilot.freeze(cls.output, profile="smoke")

        # If fresh selection touches the exact deployment evaluator, this test
        # fails immediately.  Proxy exact values remain allowed in _build_world.
        with mock.patch.object(
            pilot,
            "_exact_deployment_values",
            side_effect=AssertionError("fresh selector touched exact audit"),
        ):
            pilot.run_fresh_selection(cls.output)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def test_selection_is_sealed_before_audit(self) -> None:
        self.assertTrue((self.output / "selection_seal.json").exists())
        self.assertFalse((self.output / "audit_results.json").exists())
        packet = json.loads((self.output / "fresh_decisions.json").read_text())
        self.assertFalse(packet["exact_audit_generated"])
        self.assertFalse(packet["audit_labels_used"])
        serialized = json.dumps(packet)
        self.assertNotIn("exact_deployment_improvement", serialized)
        self.assertNotIn("selection_regret", serialized)

    def test_method_grid_costs_prefixes_and_budget_zero_equivalence(self) -> None:
        packet = json.loads((self.output / "fresh_decisions.json").read_text())
        decisions = packet["decisions"]
        self.assertEqual(len(decisions), 248)
        grouped = {}
        for row in decisions:
            key = (
                row["root_seed"],
                row["horizon"],
                row["method"],
                row["allocation_replicate"],
                row["budget"],
            )
            self.assertNotIn(key, grouped)
            grouped[key] = row
            self.assertEqual(
                row["physical_profile_hands"],
                2 * row["nominal_paired_hands"],
            )
            self.assertNotIn("incumbent", row["queried_ids"])
            self.assertEqual(len(row["queried_ids"]), len(set(row["queried_ids"])))

        for root in range(72000, 72004):
            for horizon in pilot.HORIZONS:
                calibrated = grouped[(root, horizon, "calibrated_no_hf", 0, 0)][
                    "selected_id"
                ]
                for method in (
                    "pivot_cg_v2",
                    "global_ivr",
                    "posterior_lucb",
                    "author_global_voi_e5c_adapter",
                ):
                    self.assertEqual(
                        grouped[(root, horizon, method, 0, 0)]["selected_id"],
                        calibrated,
                    )
                for replicate in range(pilot.RANDOM_ALLOCATION_REPLICATES):
                    self.assertEqual(
                        grouped[(root, horizon, "uniform_random", replicate, 0)][
                            "selected_id"
                        ],
                        calibrated,
                    )
                    q64 = grouped[(root, horizon, "uniform_random", replicate, 64)][
                        "queried_ids"
                    ]
                    q128 = grouped[(root, horizon, "uniform_random", replicate, 128)][
                        "queried_ids"
                    ]
                    self.assertEqual(q64, q128[:1])

    def test_tampering_fails_closed_then_audit_scores_exact_regret(self) -> None:
        decision_path = self.output / "fresh_decisions.json"
        original = decision_path.read_bytes()
        decision_path.write_bytes(original + b"\n")
        lock, _ = pilot._verify_lock(self.output)
        with self.assertRaisesRegex(RuntimeError, "sealed artifact changed"):
            pilot._verify_selection_seal(self.output, lock)
        decision_path.write_bytes(original)

        payload = pilot.audit_fresh_selection(self.output)
        self.assertEqual(payload["status"], "AUDITED_AFTER_SELECTION_SEAL")
        self.assertFalse(payload["selection_audit_seed_overlap"])
        self.assertTrue(
            all(row["selection_regret"] >= -1e-12 for row in payload["scored_decisions"])
        )
        self.assertEqual(
            payload["contrasts"]["random_minus_pivot_long_budget_128"]["n_roots"],
            4,
        )


if __name__ == "__main__":
    unittest.main()
