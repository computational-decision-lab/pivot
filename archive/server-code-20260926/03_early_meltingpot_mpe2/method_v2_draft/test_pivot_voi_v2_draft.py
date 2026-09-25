"""Analytical tests only; these are not native benchmark episodes."""
import math
import unittest

import numpy as np

from pivot_voi_v2_draft import (
    CandidateValuePosterior,
    estimate_exchangeable_discrepancy,
    from_linear_calibration,
    run_v2,
    score_queries,
    stop_decision,
)


class CandidateGaussianTests(unittest.TestCase):
    def test_condition_matches_joint_normal_and_does_not_raw_overwrite(self):
        posterior = CandidateValuePosterior(
            ("candidate", "incumbent"),
            np.array([0.1, 0.0]),
            np.diag([0.01, 0.0]),
            np.array([1.0, 1.0]),
        )
        updated = posterior.condition("candidate", -0.1)
        self.assertAlmostEqual(updated.mean[0], 0.1 + 0.01 / 1.01 * (-0.2))
        self.assertGreater(updated.mean[0], 0.0)
        self.assertEqual(updated.selected_id, "candidate")
        self.assertNotAlmostEqual(updated.mean[0], -0.1)

    def test_exact_evsi_matches_two_action_closed_form(self):
        posterior = CandidateValuePosterior(
            ("a", "b"), np.zeros(2), np.eye(2), np.ones(2)
        )
        expected = 1.0 / (2.0 * math.sqrt(math.pi))
        self.assertAlmostEqual(posterior.exact_evsi("a"), expected, places=12)
        self.assertAlmostEqual(posterior.exact_evsi("b"), expected, places=12)

    def test_common_uncertainty_has_zero_evsi_and_stops_at_eta_zero(self):
        posterior = CandidateValuePosterior(
            ("a", "b"), np.array([1.0, 0.0]), np.ones((2, 2)) * 5.0, np.ones(2)
        )
        scores = score_queries(posterior, {"a": 1.0, "b": 1.0}, ["a", "b"])
        np.testing.assert_allclose([row["evsi"] for row in scores], [0.0, 0.0], atol=1e-12)
        stop, reason, _ = stop_decision(posterior, scores, regret_tolerance=0.0, eta=0.0)
        self.assertTrue(stop)
        self.assertEqual(reason, "regret_certificate")

    def test_simple_regret_bound_is_exact_for_two_candidates(self):
        posterior = CandidateValuePosterior(
            ("best", "other"),
            np.array([0.5, 0.0]),
            np.array([[1.0, 0.2], [0.2, 1.0]]),
            np.ones(2),
        )
        variance = 1.0 + 1.0 - 2.0 * 0.2
        sd = math.sqrt(variance)
        expected = sd * math.exp(-0.5 * (-0.5 / sd) ** 2) / math.sqrt(2.0 * math.pi)
        expected += -0.5 * 0.5 * (1.0 + math.erf((-0.5 / sd) / math.sqrt(2.0)))
        self.assertAlmostEqual(posterior.bayes_simple_regret_upper_bound(), expected, places=12)

    def test_exact_evsi_tie_uses_declared_ivr_not_lexical_id(self):
        posterior = CandidateValuePosterior(
            ("z", "a", "winner"),
            np.array([-100.0, -100.0, 10.0]),
            np.diag([4.0, 1.0, 0.0]),
            np.ones(3),
        )
        scores = score_queries(posterior, {"z": 1.0, "a": 1.0}, ["a", "z"])
        self.assertEqual(scores[0]["transition_id"], "z")
        self.assertEqual(scores[0]["evsi"], 0.0)

    def test_incumbent_is_an_exact_zero_reference(self):
        posterior = from_linear_calibration(
            ("candidate", "incumbent"),
            (0.2, 0.0),
            ((1.0,), (0.0,)),
            (0.1,),
            ((0.3,),),
            ((0.4, 0.0), (0.0, 0.0)),
            (1.0, 1.0),
        )
        self.assertEqual(posterior.mean[1], 0.0)
        np.testing.assert_array_equal(posterior.covariance[1], np.zeros(2))

    def test_cost_budget_and_posterior_decision_are_enforced(self):
        posterior = CandidateValuePosterior(
            ("candidate", "incumbent"),
            np.array([0.1, 0.0]),
            np.diag([0.01, 0.0]),
            np.ones(2),
        )
        result = run_v2(
            posterior,
            lambda _: {"delta": -0.1, "split": "selection", "hf_query_cost": 32.0},
            {"candidate": 32.0},
            episode_budget=36.0,
            initial_setup_cost=4.0,
            adaptive_stop=False,
        )
        self.assertEqual(result["charged_cost"], 36.0)
        self.assertEqual(result["selected_id"], "candidate")
        self.assertGreater(result["selected_estimate"], 0.0)

    def test_exchangeable_discrepancy_is_psd(self):
        residuals = np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, 1.0], [-1.0, -1.0]])
        known = np.zeros((4, 2, 2))
        discrepancy, info = estimate_exchangeable_discrepancy(residuals, known)
        np.testing.assert_allclose(discrepancy, np.ones((2, 2)))
        self.assertGreaterEqual(np.linalg.eigvalsh(discrepancy).min(), -1e-12)
        self.assertEqual(info["contrast_direction_variance"], 0.0)


if __name__ == "__main__":
    unittest.main()
