"""Worked example: score two samplers on a dump.

This is not the point of the repo -- it exists to prove the dump format is
actually sufficient for estimator evaluation, and to show the stratified
reporting the data calls for. Replace the two samplers with your own.

Error metric is relative L2 on the attention output, ||AV_hat - AV|| / ||AV||,
which is what actually propagates into the model, rather than a distance
between index histograms.
"""

from __future__ import annotations

import argparse

import numpy as np

from attn_store import AttnDump


def sample_iid(probs, S, rng):
    """SANTA: S i.i.d. draws from Categorical(probs)."""
    cdf = np.cumsum(probs, dtype=np.float64)
    u = rng.random(S) * cdf[-1]
    return np.searchsorted(cdf, u, side="right").clip(max=len(probs) - 1)


def sample_systematic(probs, S, rng):
    """S2ANTA-sys: one uniform offset, S evenly spaced CDF thresholds."""
    cdf = np.cumsum(probs, dtype=np.float64)
    z = cdf[-1]
    u = rng.random()
    thresholds = (np.arange(S) + u) * (z / S)
    return np.searchsorted(cdf, thresholds, side="right").clip(max=len(probs) - 1)


SAMPLERS = {"iid": sample_iid, "systematic": sample_systematic}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dump_dir")
    p.add_argument("--budgets", default="8,16,32,64,128")
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--max-records", type=int, default=200)
    args = p.parse_args()

    d = AttnDump(args.dump_dir)
    recs = [r for r in d.index if r.get("v_key")][: args.max_records]
    if not recs:
        raise SystemExit("no records with V stored; recapture with --save-v")
    budgets = [int(x) for x in args.budgets.split(",")]
    rng = np.random.default_rng(0)

    # Stratify: attention sinks make peaked and diffuse distributions behave
    # completely differently under sampling.
    def stratum(rec):
        return "peaked" if rec["top32_mass"] > 0.95 else (
            "diffuse" if rec["top32_mass"] < 0.5 else "mid")

    results: dict[tuple, list[float]] = {}
    for rec in recs:
        probs = d.probs(rec).astype(np.float64)
        V = d.values(rec).astype(np.float64)
        av = probs @ V
        norm = max(1e-12, np.linalg.norm(av))
        st = stratum(rec)
        for name, fn in SAMPLERS.items():
            for S in budgets:
                errs = [np.linalg.norm(V[fn(probs, S, rng)].mean(axis=0) - av) / norm
                        for _ in range(args.trials)]
                results.setdefault((st, name, S), []).extend(errs)

    from collections import Counter
    counts = Counter(stratum(r) for r in recs)
    print(f"records: {len(recs)}  strata: {dict(counts)}\n")
    print(f"{'stratum':>9} {'sampler':>11} " +
          " ".join(f"{f'S={S}':>9}" for S in budgets))
    for st in ["peaked", "mid", "diffuse"]:
        if st not in counts:
            continue
        for name in SAMPLERS:
            row = [np.mean(results[(st, name, S)]) for S in budgets]
            print(f"{st:>9} {name:>11} " + " ".join(f"{v:>9.4f}" for v in row))
    print("\nmean relative L2 error of the attention output, lower is better")


if __name__ == "__main__":
    main()
