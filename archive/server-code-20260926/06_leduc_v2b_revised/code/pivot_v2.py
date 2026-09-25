"""PIVOT v2: joint-Gaussian differential posterior over candidate deployment deltas
with exact correlated knowledge-gradient (KG/EVSI) acquisition.

Design goals (each addresses a diagnosed failure of the frozen v1 implementation):
  1. Posterior lives on the K candidate deployment deltas directly (not only on a
     2-dim feature coefficient), so candidate-specific / cross-world variation and
     model mis-specification are represented (K_mis) and the posterior is no longer
     rank-2 over-confident.
  2. HF observation noise R_c is heteroscedastic and estimated from data that are
     independent of any selection decision (E[(S-A)(S-B)] identity).
  3. Fantasy update and real update are literally the same function (`condition`).
  4. EVSI is computed exactly for the Gaussian model (Frazier, Powell & Dayanik 2009,
     correlated KG) instead of nested Monte Carlo; no spurious zeros, no index tie-break.
  5. The incumbent (no-update) is always an admissible final choice with known delta 0.

Nothing here reads audit labels of the root being decided; the harness enforces that.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

FloatArray = np.ndarray


# ----------------------------------------------------------------------------- posterior
@dataclass
class JointGaussianDeltaPosterior:
    """N(m, K) over the K candidate deployment deltas; incumbent has delta 0 exactly."""

    mean: FloatArray                 # (K,)
    covariance: FloatArray           # (K,K) PSD
    observation_variance: FloatArray  # (K,) heteroscedastic HF noise R_c
    n_observations: int = 0
    history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=float).reshape(-1)
        self.covariance = np.asarray(self.covariance, dtype=float)
        self.observation_variance = np.asarray(self.observation_variance, dtype=float).reshape(-1)
        k = self.mean.shape[0]
        if self.covariance.shape != (k, k) or self.observation_variance.shape != (k,):
            raise ValueError("shape mismatch")
        if not (np.all(np.isfinite(self.mean)) and np.all(np.isfinite(self.covariance))
                and np.all(np.isfinite(self.observation_variance))):
            raise ValueError("non-finite posterior")
        if np.any(self.observation_variance <= 0):
            raise ValueError("observation variance must be positive")

    @property
    def size(self) -> int:
        return int(self.mean.shape[0])

    def condition(self, index: int, observed: float) -> "JointGaussianDeltaPosterior":
        """Exact Gaussian update on S_index = D_index + eps, eps ~ N(0, R_index)."""
        if not math.isfinite(float(observed)):
            raise ValueError("observation must be finite")
        k_col = self.covariance[:, index]
        denom = float(self.covariance[index, index] + self.observation_variance[index])
        gain = k_col / denom
        mean = self.mean + gain * (float(observed) - self.mean[index])
        cov = self.covariance - np.outer(gain, k_col)
        cov = 0.5 * (cov + cov.T)
        return JointGaussianDeltaPosterior(mean, cov, self.observation_variance, self.n_observations + 1,
                                           self.history + [(int(index), float(observed))])

    def means_with_incumbent(self) -> FloatArray:
        return np.concatenate([self.mean, [0.0]])

    def best(self) -> int:
        """Index of argmax posterior mean; K means 'keep incumbent'."""
        return int(np.argmax(self.means_with_incumbent()))

    def selection_probability(self, rng: np.random.Generator, samples: int = 4096) -> float:
        draws = rng.multivariate_normal(self.mean, self.covariance, size=samples, method="svd")
        draws = np.concatenate([draws, np.zeros((samples, 1))], axis=1)
        return float(np.mean(np.argmax(draws, axis=1) == self.best()))

    def expected_regret(self, rng: np.random.Generator, samples: int = 4096) -> float:
        draws = rng.multivariate_normal(self.mean, self.covariance, size=samples, method="svd")
        draws = np.concatenate([draws, np.zeros((samples, 1))], axis=1)
        return float(np.mean(np.max(draws, axis=1) - draws[:, self.best()]))


# ----------------------------------------------------------------------------- exact KG
def expected_max_affine(a: FloatArray, b: FloatArray) -> float:
    """E[max_i (a_i + b_i Z)], Z ~ N(0,1), exactly (Frazier et al. 2009, Alg. 1)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    order = np.lexsort((a, b))  # sort by slope, then intercept
    a, b = a[order], b[order]
    # drop dominated duplicates of equal slope: keep largest intercept
    keep = np.ones(len(a), dtype=bool)
    for i in range(len(a) - 1):
        if math.isclose(b[i], b[i + 1], rel_tol=0, abs_tol=1e-12):
            keep[i] = False
    a, b = a[keep], b[keep]
    # upper envelope
    idx = [0]
    breaks = [-math.inf]
    for i in range(1, len(a)):
        while True:
            j = idx[-1]
            z = (a[j] - a[i]) / (b[i] - b[j])  # intersection
            if z <= breaks[-1] and len(idx) > 1:
                idx.pop(); breaks.pop()
                continue
            idx.append(i); breaks.append(z)
            break
    breaks.append(math.inf)
    total = 0.0
    from math import erf, exp, pi, sqrt
    def Phi(x): return 0.5 * (1 + erf(x / sqrt(2))) if math.isfinite(x) else (1.0 if x > 0 else 0.0)
    def phi(x): return exp(-0.5 * x * x) / sqrt(2 * pi) if math.isfinite(x) else 0.0
    for t, i in enumerate(idx):
        lo, hi = breaks[t], breaks[t + 1]
        total += a[i] * (Phi(hi) - Phi(lo)) + b[i] * (phi(lo) - phi(hi))
    return float(total)


def knowledge_gradient(post: JointGaussianDeltaPosterior, index: int) -> float:
    """Exact EVSI of one HF observation of candidate `index` under the joint Gaussian model."""
    k_col = post.covariance[:, index]
    denom = float(post.covariance[index, index] + post.observation_variance[index])
    if denom <= 0:
        return 0.0
    slope = np.concatenate([k_col / math.sqrt(denom), [0.0]])  # incumbent has no slope
    intercept = post.means_with_incumbent()
    value = expected_max_affine(intercept, slope) - float(np.max(intercept))
    return max(0.0, value)


# ----------------------------------------------------------------------------- selectors
METHODS_V2 = (
    "pivot_kg",            # exact-EVSI per cost, sequential, fixed budget
    "pivot_kg_stop",       # same with posterior-confidence stop
    "uniform_v2",          # random queries, same posterior
    "ivr_v2",              # integrated variance reduction per cost, same posterior
    "lucb_v2",             # best/challenger posterior heuristic, same posterior
    "no_hf_v2",            # calibrated no-query
    "proxy_only",
)


def run_selector(post: JointGaussianDeltaPosterior, proxies: FloatArray, query, *, method: str,
                 budget: int, costs: FloatArray, seed: int, delta: float = 0.05, eta: float = 0.0,
                 stop_samples: int = 4096) -> dict:
    """Sequential loop. `query(index) -> observed deployment delta S_index`.

    Returns selected index (K == incumbent), queried indices, charged cost, per-step details.
    """
    if method not in METHODS_V2:
        raise ValueError(method)
    K = post.size
    rng = np.random.default_rng(seed)
    observed: dict[int, float] = {}
    steps = []
    reason = "budget_exhausted"
    if method in ("no_hf_v2", "proxy_only"):
        budget = 0; reason = "no_hf_method"
    for it in range(min(budget, K)):
        available = [i for i in range(K) if i not in observed]
        if method in ("pivot_kg", "pivot_kg_stop"):
            kg = {i: knowledge_gradient(post, i) for i in available}
            acq = {i: kg[i] / costs[i] for i in available}
            j = max(available, key=lambda i: (acq[i], rng.random()))  # random tie-break, never index
            detail = {"kg": kg, "acquisition": acq}
            if method == "pivot_kg_stop":
                p_sel = post.selection_probability(rng, stop_samples)
                if p_sel >= 1 - delta:
                    reason = "selection_probability"; steps.append({"it": it, "stop": reason, "p_sel": p_sel}); break
                if acq[j] < eta:
                    reason = "evsi_per_cost"; steps.append({"it": it, "stop": reason, "p_sel": p_sel}); break
                detail["p_sel"] = p_sel
        elif method == "uniform_v2":
            j = int(rng.choice(available)); detail = {}
        elif method == "ivr_v2":
            scores = {}
            for i in available:
                k_col = post.covariance[:, i]
                scores[i] = float(np.sum(k_col ** 2) / (post.covariance[i, i] + post.observation_variance[i])) / costs[i]
            j = max(available, key=lambda i: (scores[i], rng.random())); detail = {"ivr": scores}
        elif method == "lucb_v2":
            sd = np.sqrt(np.clip(np.diag(post.covariance), 0, None))
            means = post.mean
            best = int(np.argmax(means))
            ucb = means + 1.96 * sd
            challenger = max((i for i in range(K) if i != best), key=lambda i: ucb[i])
            pair = [i for i in (best, challenger) if i in available] or available
            j = max(pair, key=lambda i: (sd[i] / costs[i], rng.random()))
            detail = {"best": best, "challenger": challenger}
        else:
            raise AssertionError(method)
        value = float(query(j))
        post = post.condition(j, value)
        observed[j] = value
        steps.append({"it": it, "query": j, "observed": value, **detail})
    if method == "proxy_only":
        est = np.concatenate([proxies, [0.0]])
    else:
        est = post.means_with_incumbent()
    selected = int(np.argmax(est))
    return {"method": method, "selected": selected, "queried": list(observed), "hf_queries": len(observed),
            "charged_cost": float(sum(costs[i] for i in observed)), "stop_reason": reason,
            "estimates": est.tolist(), "steps": steps}


# ----------------------------------------------------------------------------- fitting
def psd_project(matrix: FloatArray, floor: float = 1e-8) -> FloatArray:
    sym = 0.5 * (matrix + matrix.T)
    w, v = np.linalg.eigh(sym)
    return (v * np.clip(w, floor, None)) @ v.T


def shrink_covariance(sample: FloatArray, n: int, target: FloatArray | None = None, alpha: float | None = None):
    """Ledoit-Wolf-style linear shrinkage toward `target` (default: diag)."""
    if target is None:
        target = np.diag(np.diag(sample))
    if alpha is None:
        # simple analytic shrinkage intensity: trade-off by sample size
        k = sample.shape[0]
        alpha = min(1.0, (k + 2) / (n + k + 2))
    return (1 - alpha) * sample + alpha * target, float(alpha)


@dataclass
class PosteriorV2Spec:
    """Frozen hyper-parameters produced by `fit_posterior_v2`; serialisable."""
    mu: list           # (K,) prior mean of correction G_c = D_c - proxy_c
    K_prior: list      # (K,K) prior covariance of G (includes mean uncertainty)
    R: list            # (K,) HF observation variance
    n_roots: int
    shrinkage_alpha: float
    notes: dict


def fit_posterior_v2(g: FloatArray, ab_diff: FloatArray, s_minus_a_times_s_minus_b: FloatArray, *,
                     r_smoothing: str = "pooled_by_distance", distances: FloatArray | None = None,
                     r_floor: float = 0.25) -> PosteriorV2Spec:
    """
    g:      (R,K) correction targets  = mean(audit_A, audit_B) - proxy_observed, per dev root
    ab_diff:(R,K) audit_A - audit_B (independent 64-episode blocks, same response world)
    s_minus_a_times_s_minus_b: (R,K) (S-A)(S-B), unbiased for Var(S - D) = R_c
    """
    R_, K = g.shape
    mu = g.mean(axis=0)
    sample = np.cov(g, rowvar=False, ddof=1)
    # remove audit-mean noise: Var(mean(A,B) - D) = Var(A-B)/4
    omega_mean = np.diag(np.mean(ab_diff ** 2, axis=0) / 4.0)
    latent = sample - omega_mean
    shrunk, alpha = shrink_covariance(latent, R_)
    K_prior = psd_project(shrunk)
    K_prior = K_prior + K_prior / R_  # mean-estimation uncertainty V_mu
    raw_R = np.mean(s_minus_a_times_s_minus_b, axis=0)
    if r_smoothing == "pooled_by_distance" and distances is not None:
        # pool candidates with equal |p-.5| (symmetric grid) then floor
        R = np.zeros(K)
        for d in np.unique(np.round(distances, 6)):
            mask = np.isclose(distances, d)
            R[mask] = np.mean(raw_R[mask])
        R = np.maximum(R, r_floor)
    else:
        R = np.maximum(raw_R, r_floor)
    return PosteriorV2Spec(mu.tolist(), K_prior.tolist(), R.tolist(), int(R_), alpha,
                           {"raw_R": raw_R.tolist(), "omega_mean_diag": np.diag(omega_mean).tolist(),
                            "latent_eigs_before_psd": np.linalg.eigvalsh(0.5 * (latent + latent.T)).tolist()})


def make_posterior(spec: PosteriorV2Spec, proxies: FloatArray) -> JointGaussianDeltaPosterior:
    return JointGaussianDeltaPosterior(np.asarray(proxies) + np.asarray(spec.mu), np.asarray(spec.K_prior),
                                       np.asarray(spec.R))
