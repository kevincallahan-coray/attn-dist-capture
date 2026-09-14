"""Evaluate samplers against a captured dump. Writes per-record raw errors so
aggregation and re-slicing happen later without recomputing.

Error metrics, all on the attention output AV (the thing that propagates into
the model), never on index histograms:

  abs_err  = ||AV_hat - AV||
  rel_err  = abs_err / ||AV||                 -- collapses on diffuse records
  scaled   = abs_err / (||V||_F / sqrt(nk))   -- scale-free, comparable across
                                                 strata because the denominator
                                                 does not depend on cancellation

Also recorded per trial: number of distinct indices drawn, and whether the
argmax token was captured. On peaked records, missing the argmax is the failure
mode that mean error hides.

The ``order`` dimension is the diagnostic for *why* a sampler wins. Systematic
sampling exploits similarity between adjacent indices; here the index axis is
token position. Running the identical distribution under a random permutation
of the index axis breaks any positional structure while leaving the weight
distribution untouched. If systematic's advantage survives the shuffle it came
from variance reduction alone; if it disappears, it was exploiting position.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from attn_store import AttnDump
from samplers import SAMPLERS, CORE_SAMPLERS


def stratum_of(rec: dict) -> str:
    t = rec["top32_mass"]
    return "peaked" if t > 0.95 else ("diffuse" if t < 0.5 else "mid")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dump", required=True, help="a single task dump directory")
    p.add_argument("--out", required=True, help="output .jsonl path")
    p.add_argument("--budgets", default="4,8,16,32,64,128,256,512")
    p.add_argument("--trials", type=int, default=32)
    p.add_argument("--max-records", type=int, default=400)
    p.add_argument("--samplers", default=",".join(CORE_SAMPLERS))
    p.add_argument("--orders", default="natural,shuffled")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stratify-balanced", action="store_true",
                   help="draw equal record counts per stratum rather than "
                        "whatever the dump happens to contain")
    args = p.parse_args()

    d = AttnDump(args.dump)
    budgets = [int(x) for x in args.budgets.split(",")]
    names = [s for s in args.samplers.split(",") if s]
    orders = [o for o in args.orders.split(",") if o]
    rng = np.random.default_rng(args.seed)

    usable = [r for r in d.index if r.get("v_key")]
    if not usable:
        raise SystemExit(f"{args.dump}: no records with V stored "
                         "(recapture with --save-v)")

    if args.stratify_balanced:
        per = max(1, args.max_records // 3)
        chosen: list[dict] = []
        for st in ("peaked", "mid", "diffuse"):
            pool = [r for r in usable if stratum_of(r) == st]
            if not pool:
                continue
            take = min(per, len(pool))
            idx = rng.choice(len(pool), take, replace=False)
            chosen += [pool[i] for i in idx]
        recs = chosen
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
        # Scale-free denominator: RMS norm of a value row. Independent of how
        # much cancellation happens in the weighted average.
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
                fn = SAMPLERS[name]
                for S in budgets:
                    abs_errs = np.empty(args.trials)
                    distinct = np.empty(args.trials)
                    hit = np.empty(args.trials)
                    for t in range(args.trials):
                        idx = fn(probs, S, rng)
                        est = V[idx].mean(axis=0)
                        abs_errs[t] = np.linalg.norm(est - av)
                        distinct[t] = len(np.unique(idx))
                        hit[t] = float(argmax in idx)
                    fout.write(json.dumps({
                        "dump": str(args.dump),
                        "record_id": rec["record_id"],
                        "layer": rec["layer"], "head": rec["head"],
                        "nk": nk, "stratum": st, "order": order,
                        "top1_mass": rec["top1_mass"],
                        "top32_mass": rec["top32_mass"],
                        "entropy_nats": rec["entropy_nats"],
                        "sampler": name, "S": S, "trials": args.trials,
                        "av_norm": av_norm, "v_rms": v_rms,
                        "abs_mean": float(abs_errs.mean()),
                        "abs_p50": float(np.percentile(abs_errs, 50)),
                        "abs_p90": float(np.percentile(abs_errs, 90)),
                        "abs_p99": float(np.percentile(abs_errs, 99)),
                        "abs_max": float(abs_errs.max()),
                        "rel_mean": float((abs_errs / max(av_norm, 1e-12)).mean()),
                        "scaled_mean": float((abs_errs / max(v_rms, 1e-12)).mean()),
                        "scaled_p90": float(
                            np.percentile(abs_errs / max(v_rms, 1e-12), 90)),
                        "distinct_mean": float(distinct.mean()),
                        "argmax_hit_rate": float(hit.mean()),
                    }) + "\n")
                    n_rows += 1
        if (n + 1) % 25 == 0:
            print(f"  {n+1}/{len(recs)} records, {n_rows} rows, "
                  f"{time.time()-t0:.0f}s", flush=True)

    fout.close()
    print(f"wrote {n_rows} rows to {out_path} in {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
