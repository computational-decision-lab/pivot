import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pivot_v3 as v3  # noqa: E402

try:  # equivalence tests against the frozen v2 module when it is importable
    import pivot_v2 as v2
except Exception:  # noqa: BLE001
    v2 = None


def _random_posterior(rng, K=6):
    A = rng.normal(size=(K, K))
    cov = A @ A.T / K + 0.05 * np.eye(K)
    mean = rng.normal(size=K)
    unit = rng.uniform(0.5, 2.0, size=K)
    return mean, cov, unit


class TestExactKG(unittest.TestCase):
    def test_expected_max_matches_monte_carlo(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=7), rng.normal(size=7)
        z = rng.standard_normal(400000)
        mc = np.max(a[:, None] + b[:, None] * z[None, :], axis=0).mean()
        self.assertAlmostEqual(v3.expected_max_affine(a, b), mc, places=2)

    def test_kg_zero_when_uninformative(self):
        mean = np.array([0.1, 0.2, 0.3])
        cov = np.diag([1.0, 1.0, 1e-12])
        post = v3.MultiFidelityPosterior(mean, cov, np.ones(3))
        self.assertLess(v3.knowledge_gradient(post, 2, 100), 1e-6)

    def test_kg_increases_with_hands(self):
        rng = np.random.default_rng(1)
        mean, cov, unit = _random_posterior(rng)
        post = v3.MultiFidelityPosterior(mean, cov, unit)
        kgs = [v3.knowledge_gradient(post, 0, n) for n in (64, 512, 4096)]
        self.assertTrue(kgs[0] <= kgs[1] + 1e-12 and kgs[1] <= kgs[2] + 1e-12)

    @unittest.skipIf(v2 is None, "pivot_v2 not importable")
    def test_reduces_to_v2_for_fixed_size(self):
        rng = np.random.default_rng(2)
        mean, cov, unit = _random_posterior(rng)
        n = 256
        p3 = v3.MultiFidelityPosterior(mean, cov, unit)
        p2 = v2.JointGaussianDeltaPosterior(mean, cov, unit / n)
        for i in range(len(mean)):
            self.assertAlmostEqual(v3.knowledge_gradient(p3, i, n), v2.knowledge_gradient(p2, i), places=10)
        q3 = p3.condition(2, 0.7, n)
        q2 = p2.condition(2, 0.7)
        np.testing.assert_allclose(q3.mean, q2.mean, atol=1e-12)
        np.testing.assert_allclose(q3.covariance, q2.covariance, atol=1e-12)


class TestBankAndSelectors(unittest.TestCase):
    def _bank(self, rng, K=5, n_max=4096):
        paired = rng.normal(0.1, 1.0, size=(K, n_max))
        new = rng.normal(0.0, 1.5, size=(K, n_max))
        old = rng.normal(0.0, 1.5, size=(K, n_max))
        return v3.SealedBank(paired, new, old)

    def test_bank_consumes_fresh_hands(self):
        rng = np.random.default_rng(3)
        bank = self._bank(rng)
        y1 = bank.query(0, 512)
        y2 = bank.query(0, 512)
        self.assertEqual(bank.remaining(0), 4096 - 1024)
        self.assertAlmostEqual(y1, bank.paired[0, :512].mean())
        self.assertAlmostEqual(y2, bank.paired[0, 512:1024].mean())
        with self.assertRaises(ValueError):
            bank.query(0, 4096)
        yu = bank.query(1, 256, unpaired=True)
        self.assertAlmostEqual(yu, bank.new[1, :256].mean() - bank.old[1, :256].mean())

    def test_budget_accounting_all_methods(self):
        rng = np.random.default_rng(4)
        K = 5
        mean, cov, unit = _random_posterior(rng, K)
        for m in v3.METHODS_V3:
            bank = self._bank(np.random.default_rng(5), K)
            post = v3.MultiFidelityPosterior(mean, cov, unit)
            r = v3.run_selector(post, np.zeros(K), bank, method=m, cap_hands=16384, sizes=[512, 2048, 4096],
                                fixed_size=4096, seed=7, unpaired_unit_variance=2 * unit)
            if m != "all_hf_fixed":
                self.assertLessEqual(r["hands_used"], 16384, m)
            self.assertEqual(r["hands_used"], sum(2 * n for _, n, _ in r["queries"]), m)
            self.assertTrue(0 <= r["selected"] <= K, m)
            for i, n, _ in r["queries"]:
                self.assertIn(n, (512, 2048, 4096))
        bank = self._bank(np.random.default_rng(5), K)
        r = v3.run_selector(v3.MultiFidelityPosterior(mean, cov, unit), np.zeros(K), bank, method="uniform_fixed",
                            cap_hands=16384, sizes=[512, 2048, 4096], fixed_size=4096, seed=7)
        self.assertEqual(r["n_queries"], 2)
        self.assertEqual(len({i for i, _, _ in r["queries"]}), 2)

    def test_no_hf_methods_do_not_query(self):
        rng = np.random.default_rng(6)
        mean, cov, unit = _random_posterior(rng, 4)
        for m in ("no_hf", "proxy_only"):
            r = v3.run_selector(v3.MultiFidelityPosterior(mean, cov, unit), np.array([1, 2, 3, -1.0]), self._bank(rng, 4),
                                method=m, cap_hands=16384, sizes=[512], fixed_size=4096, seed=1)
            self.assertEqual(r["n_queries"], 0)
        self.assertEqual(r["selected"], 2)  # proxy_only picks the proxy argmax

    def test_menu_kg_prefers_cheap_query_when_signal_is_large(self):
        # one candidate whose prior sd dwarfs the observation noise: a small query already resolves the decision
        mean = np.array([0.0, 0.0])
        cov = np.array([[4.0, 0.0], [0.0, 0.01]])
        unit = np.array([1.0, 1.0])
        bank = self._bank(np.random.default_rng(8), 2)
        r = v3.run_selector(v3.MultiFidelityPosterior(mean, cov, unit), np.zeros(2), bank, method="pivot_kg_menu",
                            cap_hands=8192, sizes=[512, 2048, 4096], fixed_size=4096, seed=1)
        self.assertEqual(r["queries"][0][0], 0)
        self.assertEqual(r["queries"][0][1], 512)

    def test_selector_signature_has_no_label_channel(self):
        import inspect
        params = inspect.signature(v3.run_selector).parameters
        self.assertFalse(any("label" in p or "audit" in p or "exact" in p for p in params))


class TestFit(unittest.TestCase):
    def test_fit_shapes_and_psd(self):
        rng = np.random.default_rng(9)
        R, K = 12, 8
        g = rng.normal(size=(R, K))
        spec = v3.fit_prior_v3(g, np.zeros((R, K)), rng.uniform(0.5, 1, size=(R, K)), rng.uniform(1, 2, size=(R, K)),
                               label_source="exact")
        Kp = np.array(spec.K_prior)
        self.assertEqual(Kp.shape, (K, K))
        self.assertTrue(np.all(np.linalg.eigvalsh(Kp) > 0))
        post = v3.make_posterior(spec, np.zeros(K))
        self.assertEqual(post.size, K)


if __name__ == "__main__":
    unittest.main()
