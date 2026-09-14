"""Samplers under test.

All of these draw indices from Categorical(p) where p = softmax(scores), so the
attention-output estimator is the unweighted mean of the gathered value rows:

    AV_hat = (1/S) * sum_s V[idx_s]

Everything is vectorized on the CDF. The ``while cdf[idx] < thresh`` scan in the
reference implementation is O(nk) per draw; ``searchsorted`` is O(log nk), which
matters when nk is 4096 and you are running thousands of trials.
"""

from __future__ import annotations

import numpy as np


def _cdf(probs: np.ndarray) -> tuple[np.ndarray, float]:
    c = np.cumsum(probs, dtype=np.float64)
    return c, float(c[-1])


def _pick(cdf: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.searchsorted(cdf, thresholds, side="right").clip(max=len(cdf) - 1)


def sample_iid(probs, S, rng):
    """SANTA: S independent draws from Categorical(p)."""
    cdf, Z = _cdf(probs)
    return _pick(cdf, rng.random(S) * Z)


def sample_systematic(probs, S, rng):
    """One shared uniform offset, S evenly spaced CDF thresholds.

    Draws are maximally spread but fully correlated: the whole grid shifts
    together, so there is exactly one source of randomness regardless of S.
    """
    cdf, Z = _cdf(probs)
    delta = Z / S
    return _pick(cdf, (rng.random() + np.arange(S)) * delta)


def sample_stratified(probs, S, rng):
    """One *independent* uniform per stratum.

    This is the part that differs from systematic. Each of the S equal-mass
    strata gets its own u_i ~ U[0,1), so draws stay independent across strata
    while still guaranteeing one sample per stratum.
    """
    cdf, Z = _cdf(probs)
    delta = Z / S
    return _pick(cdf, (np.arange(S) + rng.random(S)) * delta)


def sample_mcmc_nn(probs, S, rng, burn_in: int = 0, update: str = "glauber"):
    """Nearest-neighbour random walk with Glauber or Metropolis acceptance.

    Included for completeness, with a warning: on a ring of nk = 4096 states a
    nearest-neighbour walk is diffusive, so its mixing time scales like nk^2.
    At any S a sampler budget would tolerate, the chain has explored O(sqrt(S))
    positions and is nowhere near stationary. Expect it to lose badly, and read
    that as a statement about the proposal, not about MCMC in general.
    """
    logp = np.log(np.clip(probs, 1e-300, None))
    n = len(probs)
    cur = int(rng.integers(n))
    out = np.empty(S, dtype=np.int64)
    steps = burn_in + S
    offs = rng.integers(0, 2, size=steps) * 2 - 1
    unis = rng.random(steps)
    for t in range(steps):
        cand = (cur + int(offs[t])) % n
        d = logp[cand] - logp[cur]
        if update == "glauber":
            acc = 1.0 / (1.0 + np.exp(-d))
        else:
            acc = 1.0 if d >= 0 else np.exp(d)
        if unis[t] < acc:
            cur = cand
        if t >= burn_in:
            out[t - burn_in] = cur
    return out


def sample_mcmc_uniform(probs, S, rng, burn_in: int = 0, update: str = "glauber"):
    """Independence-style proposal: jump to a uniformly random other state."""
    logp = np.log(np.clip(probs, 1e-300, None))
    n = len(probs)
    cur = int(rng.integers(n))
    out = np.empty(S, dtype=np.int64)
    steps = burn_in + S
    cands = rng.integers(1, n, size=steps)
    unis = rng.random(steps)
    for t in range(steps):
        cand = int((cur + cands[t]) % n)
        d = logp[cand] - logp[cur]
        if update == "glauber":
            acc = 1.0 / (1.0 + np.exp(-d))
        else:
            acc = 1.0 if d >= 0 else np.exp(d)
        if unis[t] < acc:
            cur = cand
        if t >= burn_in:
            out[t - burn_in] = cur
    return out


SAMPLERS = {
    "iid": sample_iid,
    "systematic": sample_systematic,
    "stratified": sample_stratified,
    "mcmc_nn_glauber": lambda p, S, r: sample_mcmc_nn(p, S, r, burn_in=0),
    "mcmc_uni_glauber": lambda p, S, r: sample_mcmc_uniform(p, S, r, burn_in=0),
}

# The three that share a common cost model (S value-row fetches, no chain state).
CORE_SAMPLERS = ["iid", "systematic", "stratified"]
