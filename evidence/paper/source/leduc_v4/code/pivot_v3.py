"""PIVOT v3: joint-Gaussian differential posterior with MULTI-FIDELITY paired queries.

Extends pivot_v2 (frozen, untouched) in three ways needed by the Leduc v4 design:
  1. A query is (candidate i, n hands). Observation y = mean of n fresh paired differences,
     y ~ N(D_i, unit_var[i] / n). Cost = 2 n physical hands (candidate + incumbent rollouts).
     Repeated queries of the same candidate consume fresh hands from the sealed bank.
  2. Exact correlated knowledge gradient (Frazier, Powell & Dayanik 2009) evaluated per (i, n),
     acquisition = KG(i, n) / cost(n)  -> the paper's EVSI-per-cost rule with a fidelity menu.
  3. Unpaired variant: y = mean(new_i[:n]) - mean(old_indep[:n]) with unit_var_unpaired[i].

Nothing here reads deployment labels of the root being decided; the harness enforces that.
With a single fixed n and unit_var = R * n, `condition` and `knowledge_gradient` reduce exactly
to pivot_v2 (checked in test_pivot_v3.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

FloatArray = np.ndarray


# ----------------------------------------------------------------------------- posterior
@dataclass
class MultiFidelityPosterior:
    mean: FloatArray            # (K,) posterior mean of deployment deltas
    covariance: FloatArray      # (K,K) PSD
    unit_variance: FloatArray   # (K,) per-hand variance of the observation stream (paired or unpaired)
    n_observations: int = 0
    history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=float).reshape(-1)
        self.covariance = np.asarray(self.covariance, dtype=float)
        self.unit_variance = np.asarray(self.unit_variance, dtype=float).reshape(-1)
        k = self.mean.shape[0]
        if self.covariance.shape != (k, k) or self.unit_variance.shape != (k,):
            raise ValueError("shape mismatch")
        if not (np.all(np.isfinite(self.mean)) and np.all(np.isfinite(self.covariance))
                and np.all(np.isfinite(self.unit_variance))):
            raise ValueError("non-finite posterior")
        if np.any(self.unit_variance <= 0):
            raise ValueError("unit variance must be positive")

    @property
    def size(self) -> int:
        return int(self.mean.shape[0])

    def observation_variance(self, index: int, n_hands: int) -> float:
        return float(self.unit_variance[index] / float(n_hands))

    def condition(self, index: int, observed: float, n_hands: int) -> "MultiFidelityPosterior":
        if not math.isfinite(float(observed)):
            raise ValueError("observation must be finite")
        if n_hands <= 0:
            raise ValueError("n_hands must be positive")
        k_col = self.covariance[:, index]
        denom = float(self.covariance[index, index] + self.observation_variance(index, n_hands))
        gain = k_col / denom
        mean = self.mean + gain * (float(observed) - self.mean[index])
        cov = self.covariance - np.outer(gain, k_col)
        cov = 0.5 * (cov + cov.T)
        return MultiFidelityPosterior(mean, cov, self.unit_variance, self.n_observations + 1,
                                      self.history + [(int(index), int(n_hands), float(observed))])

    def means_with_incumbent(self) -> FloatArray:
        return np.concatenate([self.mean, [0.0]])

    def best(self) -> int:
        return int(np.argmax(self.means_with_incumbent()))

    def _draws(self, rng: np.random.Generator, samples: int) -> FloatArray:
        # eigen-decomposition sampler: deterministic given rng, robust to tiny negative eigenvalues
        w, v = np.linalg.eigh(0.5 * (self.covariance + self.covariance.T))
        root = v * np.sqrt(np.clip(w, 0.0, None))
        z = rng.standard_normal(size=(samples, self.size))
        draws = self.mean + z @ root.T
        return np.concatenate([draws, np.zeros((samples, 1))], axis=1)

    def selection_probability(self, rng: np.random.Generator, samples: int = 4096) -> float:
        draws = self._draws(rng, samples)
        return float(np.mean(np.argmax(draws, axis=1) == self.best()))

    def expected_regret(self, rng: np.random.Generator, samples: int = 4096) -> float:
        draws = self._draws(rng, samples)
        return float(np.mean(np.max(draws, axis=1) - draws[:, self.best()]))


# ----------------------------------------------------------------------------- exact KG
def expected_max_affine(a: FloatArray, b: FloatArray) -> float:
    """E[max_i (a_i + b_i Z)], Z ~ N(0,1), exactly (Frazier et al. 2009, Alg. 1)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    order = np.lexsort((a, b))
    a, b = a[order], b[order]
    keep = np.ones(len(a), dtype=bool)
    for i in range(len(a) - 1):
        if math.isclose(b[i], b[i + 1], rel_tol=0, abs_tol=1e-12):
            keep[i] = False
    a, b = a[keep], b[keep]
    idx = [0]
    breaks = [-math.inf]
    for i in range(1, len(a)):
        while True:
            j = idx[-1]
            z = (a[j] - a[i]) / (b[i] - b[j])
            if z <= breaks[-1] and len(idx) > 1:
                idx.pop(); breaks.pop()
                continue
            idx.append(i); breaks.append(z)
            break
    breaks.append(math.inf)
    from math import erf, exp, pi, sqrt

    def Phi(x):
        return 0.5 * (1 + erf(x / sqrt(2))) if math.isfinite(x) else (1.0 if x > 0 else 0.0)

    def phi(x):
        return exp(-0.5 * x * x) / sqrt(2 * pi) if math.isfinite(x) else 0.0

    total = 0.0
    for t, i in enumerate(idx):
        lo, hi = breaks[t], breaks[t + 1]
        total += a[i] * (Phi(hi) - Phi(lo)) + b[i] * (phi(lo) - phi(hi))
    return float(total)


def knowledge_gradient(post: MultiFidelityPosterior, index: int, n_hands: int) -> float:
    """Exact EVSI (in gain units) of observing candidate `index` with n_hands paired hands."""
    k_col = post.covariance[:, index]
    denom = float(post.covariance[index, index] + post.observation_variance(index, n_hands))
    if denom <= 0:
        return 0.0
    slope = np.concatenate([k_col / math.sqrt(denom), [0.0]])
    intercept = post.means_with_incumbent()
    return max(0.0, expected_max_affine(intercept, slope) - float(np.max(intercept)))


def ivr_score(post: MultiFidelityPosterior, index: int, n_hands: int) -> float:
    """Integrated variance reduction of the K posterior marginals from one observation."""
    k_col = post.covariance[:, index]
    denom = float(post.covariance[index, index] + post.observation_variance(index, n_hands))
    return float(np.sum(k_col ** 2) / denom) if denom > 0 else 0.0


# ----------------------------------------------------------------------------- sealed bank
class SealedBank:
    """Per-root bank of per-hand outcomes. Queries consume fresh hands sequentially.

    paired_diff[i]  : (n_max,) per-hand (candidate - incumbent) under common random numbers
    new_returns[i]  : (n_max,) per-hand candidate returns (same hands as paired_diff)
    old_indep[i]    : (n_max,) per-hand incumbent returns from an INDEPENDENT stream
    """

    def __init__(self, paired_diff: FloatArray, new_returns: FloatArray, old_indep: FloatArray):
        self.paired = np.asarray(paired_diff, dtype=float)
        self.new = np.asarray(new_returns, dtype=float)
        self.old = np.asarray(old_indep, dtype=float)
        if not (self.paired.shape == self.new.shape == self.old.shape) or self.paired.ndim != 2:
            raise ValueError("bank arrays must share shape (K, n_max)")
        self.K, self.n_max = self.paired.shape
        self.used = np.zeros(self.K, dtype=int)

    def remaining(self, index: int) -> int:
        return int(self.n_max - self.used[index])

    def query(self, index: int, n_hands: int, *, unpaired: bool = False) -> float:
        if n_hands <= 0 or self.remaining(index) < n_hands:
            raise ValueError("not enough fresh hands in the bank")
        lo, hi = self.used[index], self.used[index] + n_hands
        self.used[index] = hi
        if unpaired:
            return float(self.new[index, lo:hi].mean() - self.old[index, lo:hi].mean())
        return float(self.paired[index, lo:hi].mean())


# ----------------------------------------------------------------------------- selectors
METHODS_V3 = (
    "pivot_kg_menu",            # exact EVSI per cost over the (candidate, size) menu           [primary]
    "pivot_kg_menu_stop",       # + posterior-confidence stop
    "pivot_kg_fixed",           # exact EVSI per cost, fixed query size                         (menu ablation)
    "pivot_kg_menu_unpaired",   # menu, but observations are unpaired                          (pairing ablation)
    "ivr_menu",                 # integrated variance reduction per cost over the menu         [acquisition comparator]
    "ivr_fixed",
    "lucb_fixed",               # best / challenger by posterior, fixed size
    "top_proxy_fixed",          # query candidates in descending proxy order, fixed size
    "uniform_fixed",            # random candidate without replacement, fixed size            [registered comparator]
    "uniform_small",            # random candidate with replacement, smallest size
    "all_hf_fixed",             # every candidate once at the fixed size (reference, ignores cap)
    "no_hf",                    # calibrated prior only
    "proxy_only",
)


def _menu(bank: SealedBank, sizes, remaining_hands: int):
    out = []
    for i in range(bank.K):
        for n in sizes:
            if 2 * n <= remaining_hands and bank.remaining(i) >= n:
                out.append((i, int(n)))
    return out


def run_selector(post: MultiFidelityPosterior, proxies: FloatArray, bank: SealedBank, *, method: str,
                 cap_hands: int, sizes, fixed_size: int, seed: int, delta: float = 0.05,
                 stop_samples: int = 4096, unpaired_unit_variance: FloatArray | None = None) -> dict:
    """Sequential budgeted loop. Returns the final choice (K == keep incumbent) and the audit trail."""
    if method not in METHODS_V3:
        raise ValueError(method)
    K = post.size
    rng = np.random.default_rng(seed)
    sizes = sorted(int(s) for s in sizes)
    unpaired = method == "pivot_kg_menu_unpaired"
    if unpaired:
        if unpaired_unit_variance is None:
            raise ValueError("unpaired variance required")
        post = MultiFidelityPosterior(post.mean, post.covariance, unpaired_unit_variance)
    remaining = int(cap_hands)
    queries = []
    steps = []
    reason = "budget_exhausted"
    queried_once: set[int] = set()

    if method in ("no_hf", "proxy_only"):
        reason = "no_hf_method"
        remaining = 0
    if method == "all_hf_fixed":
        for i in range(K):
            y = bank.query(i, fixed_size)
            post = post.condition(i, y, fixed_size)
            queries.append((i, fixed_size, y))
        reason = "all"
        remaining = 0

    while remaining > 0:
        if method in ("pivot_kg_menu", "pivot_kg_menu_stop", "pivot_kg_menu_unpaired", "ivr_menu", "uniform_small"):
            menu = _menu(bank, sizes if method != "uniform_small" else [sizes[0]], remaining)
        else:
            menu = _menu(bank, [fixed_size], remaining)
            if method in ("uniform_fixed", "lucb_fixed", "top_proxy_fixed", "pivot_kg_fixed", "ivr_fixed"):
                fresh = [m for m in menu if m[0] not in queried_once]
                menu = fresh if fresh else ([] if method == "uniform_fixed" else menu)
        if not menu:
            break
        detail: dict = {}
        if method in ("pivot_kg_menu", "pivot_kg_menu_stop", "pivot_kg_menu_unpaired", "pivot_kg_fixed"):
            acq = {m: knowledge_gradient(post, m[0], m[1]) / (2.0 * m[1]) for m in menu}
            if method == "pivot_kg_menu_stop":
                p_sel = post.selection_probability(rng, stop_samples)
                detail["p_sel"] = p_sel
                if p_sel >= 1 - delta:
                    reason = "selection_probability"
                    steps.append({"stop": reason, "p_sel": p_sel})
                    break
            j = max(menu, key=lambda m: (acq[m], rng.random()))
            detail["acquisition"] = {f"{m[0]}@{m[1]}": v for m, v in acq.items()}
        elif method in ("ivr_menu", "ivr_fixed"):
            acq = {m: ivr_score(post, m[0], m[1]) / (2.0 * m[1]) for m in menu}
            j = max(menu, key=lambda m: (acq[m], rng.random()))
            detail["acquisition"] = {f"{m[0]}@{m[1]}": v for m, v in acq.items()}
        elif method == "lucb_fixed":
            sd = np.sqrt(np.clip(np.diag(post.covariance), 0, None))
            best = int(np.argmax(post.mean))
            ucb = post.mean + 1.96 * sd
            challenger = max((i for i in range(K) if i != best), key=lambda i: ucb[i])
            cands = [m for m in menu if m[0] in (best, challenger)] or menu
            j = max(cands, key=lambda m: (sd[m[0]], rng.random()))
            detail.update(best=best, challenger=challenger)
        elif method == "top_proxy_fixed":
            j = max(menu, key=lambda m: (float(proxies[m[0]]), rng.random()))
        elif method in ("uniform_fixed", "uniform_small"):
            j = menu[int(rng.integers(len(menu)))]
        else:
            raise AssertionError(method)
        i, n = j
        y = bank.query(i, n, unpaired=unpaired)
        post = post.condition(i, y, n)
        queries.append((i, n, y))
        queried_once.add(i)
        remaining -= 2 * n
        steps.append({"query": i, "hands": n, "observed": y, **detail})

    est = np.concatenate([np.asarray(proxies, dtype=float), [0.0]]) if method == "proxy_only" else post.means_with_incumbent()
    return {"method": method, "selected": int(np.argmax(est)), "queries": [(int(i), int(n), float(y)) for i, n, y in queries],
            "n_queries": len(queries), "hands_used": int(sum(2 * n for _, n, _ in queries)),
            "stop_reason": reason, "estimates": est.tolist(), "steps": steps}


# ----------------------------------------------------------------------------- fitting
def psd_project(matrix: FloatArray, floor: float = 1e-8) -> FloatArray:
    sym = 0.5 * (matrix + matrix.T)
    w, v = np.linalg.eigh(sym)
    return (v * np.clip(w, floor, None)) @ v.T


def shrink_covariance(sample: FloatArray, n: int):
    target = np.diag(np.diag(sample))
    k = sample.shape[0]
    alpha = min(1.0, (k + 2) / (n + k + 2))
    return (1 - alpha) * sample + alpha * target, float(alpha)


@dataclass
class PriorSpecV3:
    mu: list                    # (K,) prior mean of correction G_c = D_c - proxy_c
    K_prior: list               # (K,K)
    unit_variance_paired: list  # (K,) per-hand variance of paired differences
    unit_variance_unpaired: list  # (K,) per-hand variance of new - old_indep differences (sum of the two)
    n_roots: int
    shrinkage_alpha: float
    label_source: str           # "exact" or "noisy_audit"
    notes: dict


def fit_prior_v3(g: FloatArray, ab_diff: FloatArray, unit_var_paired: FloatArray, unit_var_unpaired: FloatArray,
                 *, label_source: str, var_floor: float = 1e-10) -> PriorSpecV3:
    """
    g               (R,K) correction targets = label - proxy, per calibration root
    ab_diff         (R,K) audit_A - audit_B when labels are noisy audit means (zeros when exact)
    unit_var_*      (R,K) per-hand variance estimates per root (averaged across roots here)
    """
    R_, K = g.shape
    mu = g.mean(axis=0)
    sample = np.cov(g, rowvar=False, ddof=1) if R_ > 1 else np.zeros((K, K))
    omega_mean = np.diag(np.mean(ab_diff ** 2, axis=0) / 4.0)
    latent = sample - omega_mean
    shrunk, alpha = shrink_covariance(latent, R_)
    K_prior = psd_project(shrunk)
    K_prior = K_prior + K_prior / R_
    uvp = np.maximum(np.mean(unit_var_paired, axis=0), var_floor)
    uvu = np.maximum(np.mean(unit_var_unpaired, axis=0), var_floor)
    return PriorSpecV3(mu.tolist(), K_prior.tolist(), uvp.tolist(), uvu.tolist(), int(R_), alpha, label_source,
                       {"omega_mean_diag": np.diag(omega_mean).tolist(),
                        "latent_eigs_before_psd": np.linalg.eigvalsh(0.5 * (latent + latent.T)).tolist()})


def make_posterior(spec: PriorSpecV3, proxies: FloatArray) -> MultiFidelityPosterior:
    return MultiFidelityPosterior(np.asarray(proxies, dtype=float) + np.asarray(spec.mu), np.asarray(spec.K_prior),
                                  np.asarray(spec.unit_variance_paired))
