"""Small synthetic contract tests; no benchmark scores used to choose behavior."""
import copy
import unittest

import numpy as np
from pivot.acquisition.pivot_voi import BayesianLinearDeltaPosterior
from sequential_pivot_extension import run
from run_ablation import ROOTS, loo_noise, replace_observation_noise


class NoiseOnlyContract(unittest.TestCase):
    def setUp(self):
        self.original = BayesianLinearDeltaPosterior(noise_variance=2.).fit(np.eye(2), np.array([1., -1.]))

    def test_only_observation_noise_changes(self):
        changed = replace_observation_noise(self.original, 9.)
        self.assertEqual(changed.noise_variance, 9.)
        self.assertEqual(self.original.noise_variance, 2.)
        self.assertEqual(changed.n_observations, self.original.n_observations)
        self.assertEqual(changed.prior_precision, self.original.prior_precision)
        np.testing.assert_array_equal(changed.mean, self.original.mean)
        np.testing.assert_array_equal(changed.covariance, self.original.covariance)
        np.testing.assert_array_equal(changed.predict(np.eye(2)), self.original.predict(np.eye(2)))
        np.testing.assert_array_equal(changed.predictive_variance(np.eye(2)), self.original.predictive_variance(np.eye(2)))
        self.assertFalse(np.array_equal(changed.condition([1., 0.], 5.).mean, self.original.condition([1., 0.], 5.).mean))

    def test_held_world_cannot_enter_loo_estimate(self):
        data = {root: {"records": [{"candidate": str(i), "adaptation": h, "paired_variance": float(root - 44000 + 1)}
                                  for h in [4, 32] for i in range(8)]} for root in ROOTS}
        before = loo_noise(data, 44000, 4)
        data[44000] = {"records": None}  # If inspected this deliberately breaks.
        after = loo_noise(data, 44000, 4)
        self.assertEqual(before, after)
        self.assertNotIn(44000, after[1])
        self.assertEqual(after[0], 5.)

    def test_no_hf_policy_invariant_and_queries_costed(self):
        rows = [{"transition_id": str(i), "delta_proxy": .1, "features": f, "hf_query_cost": 36}
                for i, f in enumerate([[1., 0.], [0., 1.]])]
        def forbidden(_):
            raise AssertionError("No-HF must not query")
        changed = replace_observation_noise(self.original, 7.)
        a = run(rows, self.original, forbidden, method="calibrated_no_hf", budget=1, seed=7, stop=False)
        b = run(rows, changed, forbidden, method="calibrated_no_hf", budget=1, seed=7, stop=False)
        self.assertEqual(a, b)
        values = {"0": 4., "1": -4.}
        query = lambda row: {"delta": values[row["transition_id"]], "split": "selection", "hf_query_cost": 36}
        args = dict(method="pivot_sequential", budget=1, seed=7, stop=False, fantasies=4, posterior_samples=16)
        c = run(rows, self.original, query, **args)
        d = run(rows, copy.deepcopy(self.original), query, **args)
        self.assertEqual(c, d)
        self.assertEqual(c["charged_cost"], 36.)
        self.assertEqual(c["hf_queries"], 1)

    def test_invalid_noise_rejected(self):
        for value in [0., -1., float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                replace_observation_noise(self.original, value)


if __name__ == "__main__":
    unittest.main()
