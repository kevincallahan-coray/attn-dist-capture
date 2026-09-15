"""Evaluate samplers against a captured dump. Writes per-record raw errors so
aggregation and re-slicing happen later without recomputing.

Error is always on the attention output AV, never on index histograms, since
AV is what propagates into the model:

  abs_err  = ||AV_hat - AV||
  rel_err  = abs_err / ||AV||                 -- collapses on diffuse records
  scaled   = abs_err / (||V||_F / sqrt(nk))   -- scale-free, comparable across
                                                 strata

Extra diagnostics, recorded so a negative MCMC result is explainable rather
than merely assertable:

  accept_rate   fraction of proposed moves accepted
  tv_distance   0.5 * sum |empirical_freq - p|, empirical measure vs target.
                Note this is NOT zero for a perfect sampler at S << nk -- an
                S-point empirical measure cannot match a dense target. It is
                only meaningful compared against the i.i.d. row at the same S,
                which is the achievable floor.
  distinct      number of distinct indices visited. For a diffusive chain this
                is the smoking gun: a nearest-neighbour walk covers O(sqrt(S))
                positions, so most of the "samples" are repeats.
  argmax_hit    whether the top token was captured at all.

The ``order`` dimension controls for positional structure: every configuration
runs on the natural token order and on a random permutation of the same
distribution. i.i.d. should be unaffected, which doubles as a harness check.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from attn_store import AttnDump
from samplers import CDF_SAMPLERS, MCMC_SAMPLERS, draw


def stratum_of(rec: dict) -> str:
    t = rec["top32_mass"]
    return "peaked" if t > 0.95 else ("diffuse" if t < 0.5 else "mid")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dump", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--budgets", default="4,8,16,32,64,128,256,512")
    p.add_argument("--mcmc-extra-budgets", default="",
                   help="additional step counts given ONLY to MCMC, e.g. "
                        "'1024,4096'. MCMC steps avoid the O(nk) CDF the other "
                        "samplers need, so this is the cost-matched comparison")
    p.add_argument("--trials", type=int, default=32)
    p.add_argument("--max-records", type=int, default=400)
    p.add_argument("--samplers", default=",".join(CDF_SAMPLERS + MCMC_SAMPLERS))
    p.add_argument("--orders", default="natural,shuffled")
    p.add_argument("--mcmc-init", choices=["random", "argmax"], default="random",
                   help="argmax starts every chain at the top-probability "
                        "token: the most favourable start available, and an "
                        "alternative to burn-in rather than a complement")
    p.add_argument("--mcmc-burnin-mult", type=float, default=0.0,
                   help="burn-in steps as a multiple of S, discarded before "
                        "collecting. Label is suffixed so runs stay separable")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stratify-balanced", action="store_true")
    args = p.parse_args()

    d = AttnDump(args.dump)
    budgets = [int(x) for x in args.budgets.split(",") if x]
    extra = [int(x) for x in args.mcmc_extra_budgets.split(",") if x]
    names = [s for s in args.samplers.split(",") if s]
    orders = [o for o in args.orders.split(",") if o]
    rng = np.random.default_rng(args.seed)
    T = args.trials
    bmult = args.mcmc_burnin_mult
    suffix = ""
    if args.mcmc_init == "argmax":
        suffix += "+argmax0"
    if bmult > 0:
        suffix += f"+burn{bmult:g}S"

    usable = [r for r in d.index if r.get("v_key")]
    if not usable:
        raise SystemExit(f"{args.dump}: no records with V stored")

    if args.stratify_balanced:
        per = max(1, args.max_records // 3)
        recs = []
        for st in ("peaked", "mid", "diffuse"):
            pool = [r for r in usable if stratum_of(r) == st]
            if not pool:
                continue
            idx = rng.choice(len(pool), min(per, len(pool)), replace=False)
            recs += [pool[i] for i in idx]
    else:
        take = min(args.max_records, len(usable))
        recs = [usable[i] for i in rng.choice(len(usable), take, replace=False)]

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = open(out_path, "w")
    t0 = time.time()
    n_rows = 0

    for n, rec in enumerate(recs):
        probs0 = d.probs(rec).astype(np.float64)
        V0 = d.values(rec).astype(np.float64)
        st = stratum_of(rec)
        nk = len(probs0)
        v_rms = float(np.linalg.norm(V0) / np.sqrt(nk))

        for order in orders:
            if order == "natural":
                probs, V = probs0, V0
            else:
                perm = rng.permutation(nk)
                probs, V = probs0[perm], V0[perm]
            av = probs @ V
            av_norm = float(np.linalg.norm(av))
            argmax = int(np.argmax(probs))

            for name in names:
                is_mcmc = name in MCMC_SAMPLERS
                sweep = budgets + extra if is_mcmc else budgets
                for S in sweep:
                    burn = int(bmult * S) if is_mcmc else 0
                    idx, acc = draw(name, probs, S, rng, T, burn_in=burn,
                                    init=args.mcmc_init)

                    # Go through visit counts rather than V[idx].mean(). The
                    # gather materializes (T, S, d), which at S=4096 is 134 MB
                    # per call; the count form costs one (T, nk) @ (nk, d)
                    # matmul regardless of S, and the counts are needed for the
                    # TV diagnostic anyway.
                    counts = np.stack(
                        [np.bincount(idx[t], minlength=nk) for t in range(T)])
                    freq = counts / float(S)
                    est = freq @ V                         # (T, d)
                    err = np.linalg.norm(est - av[None, :], axis=1)
                    distinct = (counts > 0).sum(axis=1)
                    hit = (counts[:, argmax] > 0).astype(float)
                    tv = 0.5 * np.abs(freq - probs[None, :]).sum(axis=1)

                    fout.write(json.dumps({
                        "dump": str(args.dump),
                        "record_id": rec["record_id"],
                        "layer": rec["layer"], "head": rec["head"],
                        "nk": nk, "stratum": st, "order": order,
                        "top1_mass": rec["top1_mass"],
                        "top32_mass": rec["top32_mass"],
                        "entropy_nats": rec["entropy_nats"],
                        "sampler": name + (suffix if is_mcmc else ""),
                        "family": "mcmc" if is_mcmc else "cdf",
                        "S": S, "burn_in": burn, "trials": T,
                        "mcmc_init": args.mcmc_init if is_mcmc else None,
                        "av_norm": av_norm, "v_rms": v_rms,
                        "abs_mean": float(err.mean()),
                        "abs_p50": float(np.percentile(err, 50)),
                        "abs_p90": float(np.percentile(err, 90)),
                        "abs_p99": float(np.percentile(err, 99)),
                        "abs_max": float(err.max()),
                        "rel_mean": float((err / max(av_norm, 1e-12)).mean()),
                        "scaled_mean": float((err / max(v_rms, 1e-12)).mean()),
                        "scaled_p90": float(
                            np.percentile(err / max(v_rms, 1e-12), 90)),
                        "distinct_mean": float(distinct.mean()),
                        "distinct_frac": float(distinct.mean() / S),
                        "argmax_hit_rate": float(hit.mean()),
                        "accept_rate": float(acc),
                        "tv_distance": float(tv.mean()),
                    }) + "\n")
                    n_rows += 1

        if (n + 1) % 20 == 0:
            print(f"  {n+1}/{len(recs)} records, {n_rows} rows, "
                  f"{time.time()-t0:.0f}s", flush=True)

    fout.close()
    print(f"wrote {n_rows} rows to {out_path} in {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
