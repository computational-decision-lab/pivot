"""Unit tests for pivot_v2 (run: python -m unittest test_pivot_v2). No environment needed."""
import math
import unittest

import numpy as np

from pivot_v2 import (JointGaussianDeltaPosterior, expected_max_affine, fit_posterior_v2, knowledge_gradient,
                      make_posterior, run_selector)


class TestExactKG(unittest.TestCase):
    def test_expected_max_matches_monte_carlo(self):
        rng = np.random.default_rng(1)
        for _ in range(6):
            n = int(rng.integers(2, 10)); a = rng.normal(size=n) * 3; b = rng.normal(size=n) * 2
            Z = rng.standard_normal(1_000_000)
            mc = np.mean(np.max(a[:, None] + b[:, None] * Z[None, :], axis=0))
            self.assertAlmostEqual(expected_max_affine(a, b), mc, delta=0.01)

    def test_relu_identity(self):
        self.assertAlmostEqual(expected_max_affine([0, 0], [0, 1]), 1 / math.sqrt(2 * math.pi), places=12)

    def test_kg_matches_fantasy_monte_carlo_with_same_update(self):
        rng = np.random.default_rng(2)
        m = np.array([1., 2., 1.5]); K = np.array([[4, 1, .5], [1, 3, .2], [.5, .2, 2.]]); R = np.array([1., 1., 1.])
        post = JointGaussianDeltaPosterior(m, K, R)
        for j in range(3):
            base = post.means_with_incumbent().max()
            fant = [post.condition(j, s).means_with_incumbent().max() for s in rng.normal(m[j], math.sqrt(K[j, j] + R[j]), 40000)]
            self.assertAlmostEqual(knowledge_gradient(post, j), np.mean(fant) - base, delta=0.02)

    def test_uninformative_query_has_zero_kg(self):
        post = JointGaussianDeltaPosterior([0., 5.], np.diag([1., 1e-12]), [1., 1.])
        # candidate 1 is certainly best by a wide margin; observing candidate 0 cannot change the decision
        self.assertLess(knowledge_gradient(post, 0), 1e-6)


class TestPosterior(unittest.TestCase):
    def test_condition_is_exact_gaussian_update(self):
        post = JointGaussianDeltaPosterior([0., 0.], [[2., 1.], [1., 2.]], [1., 1.])
        new = post.condition(0, 3.0)
        self.assertAlmostEqual(new.mean[0], 3.0 * 2 / 3); self.assertAlmostEqual(new.mean[1], 3.0 * 1 / 3)
        self.assertAlmostEqual(new.covariance[0, 0], 2 - 4 / 3); self.assertAlmostEqual(new.covariance[1, 1], 2 - 1 / 3)

    def test_incumbent_is_admissible(self):
        post = JointGaussianDeltaPosterior([-1., -2.], np.eye(2), [1., 1.])
        self.assertEqual(post.best(), 2)  # index K == keep incumbent

    def test_fit_shapes_and_psd(self):
        rng = np.random.default_rng(3)
        g = rng.normal(size=(12, 8)); ab = rng.normal(size=(12, 8)); sab = np.abs(rng.normal(size=(12, 8)))
        spec = fit_posterior_v2(g, ab, sab, distances=np.abs(np.array([.125, .225, .325, .425, .575, .675, .775, .875]) - .5))
        K = np.array(spec.K_prior); self.assertGreaterEqual(np.linalg.eigvalsh(K).min(), 0)
        self.assertEqual(len(spec.R), 8); self.assertTrue(all(r > 0 for r in spec.R))
        self.assertAlmostEqual(spec.R[0], spec.R[7])  # symmetric pooling


class TestSelectors(unittest.TestCase):
    def setUp(self):
        self.spec_like = type("S", (), {})()
        self.mu = np.zeros(8); self.K = np.eye(8) * 4 + 1.0; self.R = np.ones(8) * 2

    def _post(self, proxies):
        return JointGaussianDeltaPosterior(proxies + self.mu, self.K, self.R)

    def test_all_equal_scores_do_not_prefer_low_index(self):
        proxies = np.zeros(8); picks = []
        for seed in range(200):
            r = run_selector(self._post(proxies), proxies, lambda j: 0.0, method="pivot_kg", budget=1, costs=np.ones(8), seed=seed)
            picks.append(r["queried"][0])
        self.assertGreater(len(set(picks)), 4)

    def test_budget_and_cost_accounting(self):
        proxies = np.linspace(-1, 1, 8)
        r = run_selector(self._post(proxies), proxies, lambda j: proxies[j] + 1, method="uniform_v2", budget=3, costs=np.full(8, 36.), seed=0)
        self.assertEqual(r["hf_queries"], 3); self.assertEqual(r["charged_cost"], 108.)

    def test_stop_rule_stops_when_confident(self):
        proxies = np.array([0, 0, 0, 0, 0, 0, 0, 50.])
        post = JointGaussianDeltaPosterior(proxies, np.eye(8) * 0.01, np.ones(8))
        r = run_selector(post, proxies, lambda j: 0.0, method="pivot_kg_stop", budget=5, costs=np.ones(8), seed=0)
        self.assertEqual(r["hf_queries"], 0); self.assertEqual(r["stop_reason"], "selection_probability")

    def test_selector_signature_has_no_label_channel(self):
        import inspect
        params = inspect.signature(run_selector).parameters
        self.assertNotIn("labels", params); self.assertNotIn("audit", params)


if __name__ == "__main__":
    unittest.main()
