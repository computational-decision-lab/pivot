import unittest

import numpy as np

try:
    from .development_crossfit_replay_v3 import (
        FEATURES, fit_model, posterior_for_proxy, shrink_covariance, shrink_query_noise,
    )
except ImportError:
    from development_crossfit_replay_v3 import (
        FEATURES, fit_model, posterior_for_proxy, shrink_covariance, shrink_query_noise,
    )


def synthetic_root(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    proxy = np.linspace(-2.0, 2.0, 8) + rng.normal(0.0, 0.15, 8)
    output = {"seed": seed, "proxy": proxy, "selection": {},
              "audit_deployment": {}, "audit_proxy": {}}
    beta = np.asarray([0.4, 0.2]) + rng.normal(0.0, 0.1, 2)
    latent_proxy = proxy + rng.normal(0.0, 0.1, 8)
    latent_deployment = latent_proxy + FEATURES @ beta + rng.normal(0.0, 0.05, 8)
    for adaptation in (4, 32):
        output["audit_proxy"][adaptation] = {
            "A": latent_proxy + rng.normal(0.0, 0.05, 8),
            "B": latent_proxy + rng.normal(0.0, 0.05, 8),
        }
        output["audit_deployment"][adaptation] = {
            "A": latent_deployment + rng.normal(0.0, 0.08, 8),
            "B": latent_deployment + rng.normal(0.0, 0.08, 8),
        }
        output["selection"][adaptation] = latent_deployment + rng.normal(0.0, 0.12, 8)
    output["audit"] = output["audit_deployment"]
    return output


class PosteriorV3Tests(unittest.TestCase):
    def test_covariance_shrinkage_is_psd_and_preserves_marginals(self):
        raw = np.asarray([[2.0, 3.0], [3.0, 2.0]])
        projected, _ = shrink_covariance(raw)
        self.assertGreater(np.min(np.linalg.eigvalsh(projected)), 0.0)
        self.assertTrue(np.all(np.diag(projected) > 0.0))

    def test_query_noise_is_heteroscedastic_and_never_shrunk_down(self):
        raw = np.asarray([-2.0, 1.0, 9.0])
        shrunk, info = shrink_query_noise(raw)
        self.assertTrue(np.all(shrunk > 0.0))
        self.assertGreater(shrunk[2], shrunk[1])
        self.assertGreaterEqual(shrunk[2], 9.0)
        self.assertEqual(info["pooling_weight_for_low_estimates"], 0.5)

    def test_model_has_all_required_covariance_components(self):
        roots = [synthetic_root(seed) for seed in range(100, 112)]
        model = fit_model(roots, 32)
        for name in ("k_beta", "k_mean", "k_proxy", "k_mis"):
            self.assertEqual(model[name].shape, (8, 8))
            self.assertGreater(np.min(np.linalg.eigvalsh(model[name])), -1e-9)
        expected = model["k_beta"] + model["k_mean"] + model["k_proxy"] + model["k_mis"]
        np.testing.assert_allclose(model["latent_covariance"], expected, atol=1e-10)
        self.assertGreater(np.std(model["r_hf"]), 0.0)

    def test_runtime_posterior_accepts_only_proxy_vector(self):
        roots = [synthetic_root(seed) for seed in range(200, 212)]
        model = fit_model(roots, 4)
        proxy = roots[0]["proxy"].copy()
        mean_a, covariance_a, noise_a = posterior_for_proxy(proxy, model)
        # Poisoning all audit labels cannot affect the runtime posterior because
        # it is constructed from the public proxy vector and frozen model only.
        roots[0]["audit_deployment"][4]["A"][:] = 1e9
        mean_b, covariance_b, noise_b = posterior_for_proxy(proxy, model)
        np.testing.assert_array_equal(mean_a, mean_b)
        np.testing.assert_array_equal(covariance_a, covariance_b)
        np.testing.assert_array_equal(noise_a, noise_b)
        self.assertEqual(mean_a.shape, (9,))


if __name__ == "__main__":
    unittest.main()
