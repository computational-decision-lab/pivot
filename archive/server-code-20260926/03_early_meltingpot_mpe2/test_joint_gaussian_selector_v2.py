"""Numerical/contract tests only; these create no native benchmark evidence."""
import math
import unittest

import numpy as np

try:
    from joint_gaussian_selector_v2 import (
        METHODS,
        VERSION,
        exact_affine_evsi,
        condition_joint_gaussian,
        gauss_hermite_evsi,
        run_joint_gaussian_selector,
    )
except ModuleNotFoundError:  # Allows discovery from the repository root too.
    from colin_pivot_cloud.joint_gaussian_selector_v2 import (
        METHODS,
        VERSION,
        exact_affine_evsi,
        condition_joint_gaussian,
        gauss_hermite_evsi,
        run_joint_gaussian_selector,
    )


class JointGaussianSelectorV2Tests(unittest.TestCase):
    def test_zero_information_evsi_is_zero(self):
        score = gauss_hermite_evsi(
            [0.0, 1.0],
            [[0.0, 0.0], [0.0, 1.0]],
            [1.0, 1.0],
            0,
        )
        self.assertEqual(score, 0.0)

    def test_two_arm_evsi_matches_analytic_value(self):
        candidate_mean = -0.2
        latent_variance = 1.0
        noise_variance = 0.25
        posterior_mean_sd = latent_variance / math.sqrt(latent_variance + noise_variance)
        z = candidate_mean / posterior_mean_sd
        phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        cdf = (1.0 + math.erf(z / math.sqrt(2.0))) / 2.0
        expected = posterior_mean_sd * phi + candidate_mean * cdf
        actual = gauss_hermite_evsi(
            [candidate_mean, 0.0],
            [[latent_variance, 0.0], [0.0, 0.0]],
            [noise_variance, 0.0],
            0,
            quadrature_order=256,
        )
        self.assertAlmostEqual(actual, expected, delta=5e-4)
        exact = exact_affine_evsi(
            [candidate_mean, 0.0],
            [[latent_variance, 0.0], [0.0, 0.0]],
            [noise_variance, 0.0],
            0,
        )
        self.assertAlmostEqual(exact, expected, places=12)

    def test_exact_evsi_matches_high_order_quadrature(self):
        mean = [0.3, -0.1, 0.0]
        covariance = [[1.2, 0.4, 0.0], [0.4, 0.8, 0.0], [0.0, 0.0, 0.0]]
        noise = [0.7, 0.4, 0.0]
        for index in (0, 1):
            self.assertAlmostEqual(
                exact_affine_evsi(mean, covariance, noise, index),
                gauss_hermite_evsi(mean, covariance, noise, index, quadrature_order=320),
                delta=1e-3,
            )

    def test_query_and_final_decision_use_same_conditioning_rule(self):
        calls = []

        def query(request):
            calls.append(request)
            return {"observation": 2.0, "cost": 1.0}

        result = run_joint_gaussian_selector(
            ["candidate", "incumbent"],
            [0.0, 0.2],
            [[1.0, 0.5], [0.5, 1.0]],
            [0.25, 0.0],
            [1.0, 0.0],
            query,
            method="uniform_random",
            budget=1.0,
            queryable=[True, False],
            incumbent_id="incumbent",
        )
        expected_mean, expected_covariance = condition_joint_gaussian(
            [0.0, 0.2],
            [[1.0, 0.5], [0.5, 1.0]],
            [0.25, 0.0],
            0,
            2.0,
        )
        np.testing.assert_allclose(result["posterior_mean"], expected_mean)
        np.testing.assert_allclose(result["posterior_covariance"], expected_covariance)
        self.assertEqual(result["selected_id"], "candidate")
        self.assertEqual(calls[0]["posterior_mean"], 0.0)
        self.assertEqual(result["final_decision_rule"], "argmax_joint_gaussian_posterior_mean")

    def test_all_methods_never_repeat_and_never_query_incumbent(self):
        ids = ["a", "b", "c", "incumbent"]
        covariance = np.eye(4)
        for method in METHODS:
            calls = []

            def query(request):
                calls.append(request["candidate_id"])
                return {"observation": 0.0, "cost": 1.0}

            result = run_joint_gaussian_selector(
                ids,
                [0.0, 0.0, 0.0, -0.1],
                covariance,
                [1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                query,
                method=method,
                budget=10.0,
                seed=7,
                queryable=[True, True, True, False],
                incumbent_id="incumbent",
                stop_on_zero_voi=False,
            )
            self.assertEqual(set(calls), {"a", "b", "c"})
            self.assertEqual(len(calls), len(set(calls)))
            self.assertNotIn("incumbent", calls)
            self.assertEqual(result["spent_cost"], 3.0)

    def test_unqueried_incumbent_can_win(self):
        def query(request):
            self.assertNotEqual(request["candidate_id"], "incumbent")
            return -3.0

        result = run_joint_gaussian_selector(
            ["a", "b", "incumbent"],
            [-1.0, -2.0, 0.0],
            np.diag([0.1, 0.1, 0.0]),
            [0.1, 0.1, 0.0],
            [1.0, 1.0, 0.0],
            query,
            method="uniform_random",
            budget=2.0,
            seed=3,
            incumbent_id="incumbent",
        )
        self.assertEqual(result["selected_id"], "incumbent")
        self.assertNotIn("incumbent", result["queried_ids"])

    def test_budget_and_reported_cost_contract(self):
        for method in METHODS:
            calls = []

            def query(request):
                calls.append(request["candidate_id"])
                return {"observation": 0.0, "cost": request["cost"]}

            result = run_joint_gaussian_selector(
                ["cheap", "expensive", "incumbent"],
                [0.0, 0.0, 0.0],
                np.eye(3),
                [1.0, 1.0, 0.0],
                [2.0, 5.0, 0.0],
                query,
                method=method,
                budget=4.0,
                seed=1,
                incumbent_id="incumbent",
                stop_on_zero_voi=False,
            )
            self.assertEqual(calls, ["cheap"])
            self.assertLessEqual(result["spent_cost"], 4.0)
            self.assertEqual(result["remaining_budget"], 2.0)

        with self.assertRaisesRegex(ValueError, "reported query cost"):
            run_joint_gaussian_selector(
                ["a"],
                [0.0],
                [[1.0]],
                [1.0],
                [2.0],
                lambda _: {"observation": 0.0, "cost": 1.0},
                method="uniform_random",
                budget=2.0,
            )

    def test_metadata_marks_development_method(self):
        result = run_joint_gaussian_selector(
            ["incumbent"],
            [0.0],
            [[0.0]],
            [0.0],
            [0.0],
            lambda _: self.fail("incumbent must not be queried"),
            method="joint_gaussian_voi",
            budget=0.0,
            incumbent_id="incumbent",
        )
        self.assertEqual(result["version"], VERSION)
        self.assertFalse(result["author_original"])
        self.assertTrue(result["development_candidate"])


if __name__ == "__main__":
    unittest.main()
