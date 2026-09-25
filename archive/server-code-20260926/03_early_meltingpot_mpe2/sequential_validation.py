"""Gaussian sequential transition validation, separate from unchanged author code.

EVSI = E[max posterior mean after one observation] - max current mean.
This follows from the tower property for expected simple regret. Each candidate
observation gives a scalar normal innovation, integrated with Gauss-Hermite.
"""
from __future__ import annotations
import numpy as np
from numpy.polynomial.hermite import hermgauss


def condition(mean, covariance, candidate, observed, noise):
    mean = np.asarray(mean, float); covariance = np.asarray(covariance, float)
    if not np.isfinite(observed) or not np.isfinite(noise) or noise <= 0:
        raise ValueError('Observation and strictly positive noise must be finite')
    c = covariance[:, candidate]
    denom = covariance[candidate, candidate] + noise
    updated_mean = mean + c * (observed - mean[candidate]) / denom
    updated_cov = covariance - np.outer(c, c) / denom
    return updated_mean, (updated_cov + updated_cov.T) / 2


def evsi(mean, covariance, noise, nodes=64):
    mean = np.asarray(mean, float); covariance = np.asarray(covariance, float)
    noise = np.broadcast_to(noise, len(mean)).astype(float)
    if np.any(noise <= 0):
        raise ValueError('Positive query noise is required')
    z, weights = hermgauss(nodes); z *= np.sqrt(2); weights /= np.sqrt(np.pi)
    scores = []
    for j in range(len(mean)):
        shifts = covariance[:, j] / np.sqrt(covariance[j, j] + noise[j])
        after = mean[None, :] + z[:, None] * shifts[None, :]
        scores.append(max(0., float(weights @ after.max(1) - mean.max())))
    return np.asarray(scores)


def run(mean, covariance, noise, query, costs, budget, seed, strategy='voi',
        stop=True, confidence_samples=4096, quadrature_nodes=64):
    mean = np.asarray(mean, float).copy(); covariance = np.asarray(covariance, float).copy()
    noise = np.broadcast_to(noise, len(mean)).astype(float); costs = np.asarray(costs, float)
    if budget < 0 or np.any(costs <= 0): raise ValueError('Invalid query budget or costs')
    rng = np.random.default_rng(seed); counts = np.zeros(len(mean), dtype=int)
    queries = []; trace = []; reason = 'budget'
    for step in range(budget + 1):
        winner = int(np.argmax(mean))
        values, vectors = np.linalg.eigh(covariance)
        if values.min() < -1e-7: raise ValueError('Posterior covariance is not PSD')
        factor = vectors * np.sqrt(np.maximum(values, 0))
        draws = mean + rng.normal(size=(confidence_samples, len(mean))) @ factor.T
        confidence = float(np.mean(draws.argmax(1) == winner))
        scores = evsi(mean, covariance, noise, quadrature_nodes)
        trace.append({'step':step, 'mean':mean.tolist(), 'covariance':covariance.tolist(),
                      'best':winner, 'confidence':confidence, 'evsi':scores.tolist()})
        if step == budget: break
        if stop and confidence >= .95:
            reason = 'posterior_confidence'; break
        if strategy == 'voi': j = int(np.argmax(scores / costs))
        elif strategy == 'random': j = int(rng.integers(len(mean)))
        elif strategy == 'uniform': j = int(np.argmin(counts))
        else: raise ValueError(strategy)
        r = int(counts[j]); result = query(j, r)
        observed = float(result['delta']); cost = float(result['total_env_steps'])
        if cost != costs[j]: raise ValueError('Recorded query cost differs from requested cost')
        queries.append({'candidate':j,'replicate':r,'delta':observed,'cost':cost})
        mean, covariance = condition(mean, covariance, j, observed, float(noise[j]))
        counts[j] += 1
    return {'selected':int(np.argmax(mean)), 'estimates':mean.tolist(),
            'queries':queries,'query_cost':sum(q['cost'] for q in queries),
            'stop_reason':reason,'trace':trace}


def exchangeable_discrepancy(residual_vectors, known_covariances):
    errors=np.asarray(residual_vectors, float); k=errors.shape[1]
    raw=errors.T@errors/len(errors)-np.mean(known_covariances,axis=0)
    u=np.ones(k)/np.sqrt(k); common=np.outer(u,u); contrast=np.eye(k)-common
    common_v=max(0.,float(u@raw@u)); contrast_v=max(0.,float(np.trace(contrast@raw)/(k-1)))
    return common_v*common+contrast_v*contrast, {'raw_second_moment':raw.tolist(),
            'common_direction_variance':common_v,'contrast_direction_variance':contrast_v}
