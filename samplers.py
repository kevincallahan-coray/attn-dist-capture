"""Samplers under test.

All draw indices from Categorical(p), p = softmax(scores), so the attention
output estimator is the unweighted mean of the gathered value rows:

    AV_hat = (1/S) * sum_s V[idx_s]

Everything exposes a *batched* interface returning shape ``(T, S)`` for T
independent trials. For the MCMC samplers this is what makes the sweep
tractable: T chains advance together as one length-T vector per step, instead
of T separate Python loops.

Cost model note
---------------
i.i.d., systematic and stratified all need the normalizing constant and a CDF,
which is O(nk). MCMC needs neither -- a Metropolis or Glauber step only ever
compares two unnormalized scores. That is the real argument for MCMC here, and
it means a step-for-step comparison understates it. Use ``--mcmc-extra-budgets``
to hand MCMC the step count its cheaper steps would actually buy.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------- CDF samplers


def _cdf(probs: np.ndarray) -> tuple[np.ndarray, float]:
    c = np.cumsum(probs, dtype=np.float64)
    return c, float(c[-1])


def _pick(cdf: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.searchsorted(cdf, thresholds, side="right").clip(max=len(cdf) - 1)


def batch_iid(probs, S, rng, T):
    """SANTA: S independent draws from Categorical(p), T times."""
    cdf, Z = _cdf(probs)
    return _pick(cdf, rng.random((T, S)) * Z)


def batch_systematic(probs, S, rng, T):
    """One shared uniform offset per trial, S evenly spaced CDF thresholds.

    Draws are maximally spread but fully correlated: the entire grid shifts
    together, so there is one source of randomness regardless of S.
    """
    cdf, Z = _cdf(probs)
    delta = Z / S
    thr = (rng.random((T, 1)) + np.arange(S)[None, :]) * delta
    return _pick(cdf, thr)


def batch_stratified(probs, S, rng, T):
    """One *independent* uniform per stratum.

    This is what distinguishes stratified from systematic. Each of the S
    equal-mass strata gets its own u ~ U[0,1), so draws stay independent across
    strata while still guaranteeing one sample per stratum. Drawing the offset
    once outside the loop collapses this into systematic.
    """
    cdf, Z = _cdf(probs)
    delta = Z / S
    thr = (np.arange(S)[None, :] + rng.random((T, S))) * delta
    return _pick(cdf, thr)


# -------------------------------------------------------------- MCMC samplers


def batch_mcmc(probs, S, rng, T, proposal="nn", update="glauber", burn_in=0):
    """Random-walk MCMC targeting Categorical(p), T chains in parallel.

    proposal
        ``nn``      -- +-1 on a ring of nk states (symmetric)
        ``uniform`` -- jump to a uniformly chosen different state (symmetric)

    update
        ``metropolis`` -- accept w.p. min(1, p_cand / p_curr)
        ``glauber``    -- accept w.p. sigmoid(log p_cand - log p_curr)

    Both proposals are symmetric, so the Hastings ratio drops out and both
    updates leave Categorical(p) invariant. Invariance is not in question; the
    question is how many steps it takes to get there.

    Returns ``(idx, accept_rate)`` where idx has shape (T, S).
    """
    logp = np.log(np.clip(np.asarray(probs, dtype=np.float64), 1e-300, None))
    n = len(logp)
    steps = burn_in + S
    cur = rng.integers(0, n, size=T)
    out = np.empty((T, S), dtype=np.int64)
    accepted = 0
    total = 0

    if proposal == "nn":
        offs = rng.integers(0, 2, size=(steps, T)) * 2 - 1
    else:
        offs = rng.integers(1, n, size=(steps, T))
    unis = rng.random((steps, T))

    for t in range(steps):
        cand = (cur + offs[t]) % n
        d = logp[cand] - logp[cur]
        if update == "glauber":
            acc = 1.0 / (1.0 + np.exp(-np.clip(d, -700, 700)))
        else:
            acc = np.minimum(1.0, np.exp(np.clip(d, -700, 0)))
        take = unis[t] < acc
        cur = np.where(take, cand, cur)
        if t >= burn_in:
            out[:, t - burn_in] = cur
            accepted += int(take.sum())
            total += T

    return out, (accepted / total if total else float("nan"))


def _mk_mcmc(proposal, update):
    def f(probs, S, rng, T, burn_in=0):
        return batch_mcmc(probs, S, rng, T, proposal=proposal,
                          update=update, burn_in=burn_in)
    return f


BATCH_SAMPLERS = {
    "iid": batch_iid,
    "systematic": batch_systematic,
    "stratified": batch_stratified,
    "mcmc_nn_glauber": _mk_mcmc("nn", "glauber"),
    "mcmc_nn_metropolis": _mk_mcmc("nn", "metropolis"),
    "mcmc_uni_glauber": _mk_mcmc("uniform", "glauber"),
    "mcmc_uni_metropolis": _mk_mcmc("uniform", "metropolis"),
}

MCMC_SAMPLERS = [k for k in BATCH_SAMPLERS if k.startswith("mcmc_")]
CDF_SAMPLERS = ["iid", "systematic", "stratified"]
ALL_SAMPLERS = CDF_SAMPLERS + MCMC_SAMPLERS


def draw(name, probs, S, rng, T, burn_in=0):
    """Uniform entry point. Returns (idx[T, S], accept_rate_or_nan)."""
    fn = BATCH_SAMPLERS[name]
    if name in MCMC_SAMPLERS:
        return fn(probs, S, rng, T, burn_in=burn_in)
    return fn(probs, S, rng, T), float("nan")
