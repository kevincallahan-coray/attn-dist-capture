"""Merge the per-task eval shards, fit convergence rates, and plot.

The headline number is not the error at one budget but the slope of
log(error) vs log(S). i.i.d. Monte Carlo is pinned at -0.5 by the CLT. A
sampler that only shifts the intercept saves you a constant factor; one that
steepens the slope changes what is achievable at a given budget. Those are very
different claims and a single-S bar chart cannot distinguish them.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import pathlib
from collections import defaultdict

import numpy as np

METRICS = ["scaled_mean", "rel_mean", "abs_mean", "scaled_p90"]


def load(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def fit_slope(S: np.ndarray, err: np.ndarray) -> tuple[float, float]:
    ok = (S > 0) & (err > 0) & np.isfinite(err)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(np.log(S[ok]), np.log(err[ok]), 1)
    return float(slope), float(np.exp(intercept))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True,
                   help="glob for the eval shards, e.g. 'results/*.jsonl'")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--metric", default="scaled_mean", choices=METRICS)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    paths = sorted(glob.glob(args.inputs))
    if not paths:
        raise SystemExit(f"no files matched {args.inputs}")
    rows = load(paths)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"loaded {len(rows)} rows from {len(paths)} shards")

    # average over records within each (stratum, order, sampler, S) cell
    cell = defaultdict(list)
    extra = defaultdict(list)
    for r in rows:
        key = (r["stratum"], r["order"], r["sampler"], r["S"])
        cell[key].append(r[args.metric])
        extra[key].append((r["argmax_hit_rate"], r["distinct_mean"]))

    strata = sorted({k[0] for k in cell}, key=lambda s:
                    {"peaked": 0, "mid": 1, "diffuse": 2}.get(s, 3))
    orders = sorted({k[1] for k in cell})
    samplers = sorted({k[2] for k in cell})
    budgets = sorted({k[3] for k in cell})

    # ---- per-cell CSV
    with open(out / "cells.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stratum", "order", "sampler", "S", "n_records",
                    args.metric, "argmax_hit_rate", "distinct_mean"])
        for k in sorted(cell):
            vals = cell[k]
            hits = np.mean([e[0] for e in extra[k]])
            dis = np.mean([e[1] for e in extra[k]])
            w.writerow([*k, len(vals), f"{np.mean(vals):.6g}",
                        f"{hits:.4f}", f"{dis:.2f}"])

    # ---- rate table
    print(f"\nconvergence of {args.metric}: err ~ C * S^slope  "
          "(i.i.d. Monte Carlo predicts slope = -0.50)\n")
    header = f"{'stratum':>9} {'order':>9} {'sampler':>17} {'slope':>8} {'C':>10}"
    header += "".join(f"{f'S={S}':>10}" for S in budgets)
    print(header)
    rate_rows = []
    for st in strata:
        for od in orders:
            for sm in samplers:
                S = np.array([s for s in budgets if (st, od, sm, s) in cell])
                if len(S) == 0:
                    continue
                e = np.array([np.mean(cell[(st, od, sm, s)]) for s in S])
                slope, C = fit_slope(S, e)
                line = f"{st:>9} {od:>9} {sm:>17} {slope:>8.3f} {C:>10.3g}"
                line += "".join(
                    f"{np.mean(cell[(st, od, sm, s)]):>10.4g}"
                    if (st, od, sm, s) in cell else f"{'':>10}"
                    for s in budgets)
                print(line)
                rate_rows.append({"stratum": st, "order": od, "sampler": sm,
                                  "slope": slope, "C": C})
        print()

    with open(out / "rates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stratum", "order", "sampler",
                                          "slope", "C"])
        w.writeheader()
        w.writerows(rate_rows)

    # ---- ordered vs shuffled: did position structure matter?
    if {"natural", "shuffled"} <= set(orders):
        print("effect of shuffling the index axis "
              "(ratio > 1 means the sampler got worse when position "
              "structure was destroyed, i.e. it was exploiting position)\n")
        print(f"{'stratum':>9} {'sampler':>17} " +
              "".join(f"{f'S={S}':>10}" for S in budgets))
        for st in strata:
            for sm in samplers:
                cells = []
                for S in budgets:
                    a = cell.get((st, "natural", sm, S))
                    b = cell.get((st, "shuffled", sm, S))
                    cells.append(np.mean(b) / np.mean(a)
                                 if a and b and np.mean(a) > 0 else float("nan"))
                print(f"{st:>9} {sm:>17} " +
                      "".join(f"{c:>10.3f}" for c in cells))
            print()

    # ---- plots
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib unavailable, skipping plots")
            return
        for od in orders:
            fig, axs = plt.subplots(1, len(strata), figsize=(5 * len(strata), 4),
                                    sharey=False)
            axs = np.atleast_1d(axs)
            for ax, st in zip(axs, strata):
                for sm in samplers:
                    S = np.array([s for s in budgets if (st, od, sm, s) in cell])
                    if len(S) == 0:
                        continue
                    e = np.array([np.mean(cell[(st, od, sm, s)]) for s in S])
                    ax.loglog(S, e, marker="o", label=sm)
                if len(budgets) >= 2:
                    ref = np.array(budgets, dtype=float)
                    anchor = np.mean(cell[(st, od, samplers[0], budgets[0])])
                    ax.loglog(ref, anchor * (ref / ref[0]) ** -0.5, "k--",
                              lw=1, label="$S^{-1/2}$")
                ax.set_title(f"{st} ({od})")
                ax.set_xlabel("samples S")
                ax.set_ylabel(args.metric)
                ax.grid(True, which="both", alpha=0.3)
            axs[0].legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out / f"error_vs_samples_{od}.png", dpi=200)
            plt.close(fig)
            print(f"wrote {out}/error_vs_samples_{od}.png")


if __name__ == "__main__":
    main()
