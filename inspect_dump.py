"""Validate a dump and report how the captured distributions are distributed.

Run this before you trust the data for sampler evaluation. It checks that
probabilities normalize, that the stored ``av`` really equals ``probs @ V`` when
V is present, and then breaks the population down by layer band -- because
attention sinks make layer 0-2 look nothing like mid-network layers, and a
sampler benchmarked on the pooled average is benchmarked on nothing in
particular.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from attn_store import AttnDump


def quantiles(x, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    a = np.asarray(x, dtype=np.float64)
    return {f"p{int(q*100)}": float(np.quantile(a, q)) for q in qs}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dump_dir")
    p.add_argument("--check", type=int, default=64,
                   help="records to numerically verify")
    p.add_argument("--bands", type=int, default=4, help="layer bands to report")
    args = p.parse_args()

    d = AttnDump(args.dump_dir)
    print(f"records: {len(d)}")
    if d.manifest:
        m = d.manifest
        print(f"model:   {m.get('model')}  ({m.get('num_layers')}L x "
              f"{m.get('num_q_heads')}H, kv={m.get('num_kv_heads')})")
        print(f"capture: {m.get('capture')}")
        print(f"stats:   {m.get('stats')}")
    if not len(d):
        return

    # ---- numerical checks
    rng = np.random.default_rng(0)
    sample = [d.index[i] for i in rng.choice(len(d), min(args.check, len(d)),
                                             replace=False)]
    worst_sum, worst_av, neg = 0.0, 0.0, 0
    for rec in sample:
        probs = d.probs(rec).astype(np.float64)
        worst_sum = max(worst_sum, abs(probs.sum() - 1.0))
        neg += int((probs < 0).sum())
        V = d.values(rec)
        if V is not None:
            ref = probs @ V.astype(np.float64)
            got = d.av(rec).astype(np.float64)
            denom = max(1e-12, np.linalg.norm(ref))
            worst_av = max(worst_av, float(np.linalg.norm(ref - got) / denom))
    print(f"\nchecked {len(sample)} records")
    print(f"  max |sum(probs) - 1| : {worst_sum:.3e}")
    print(f"  negative entries     : {neg}")
    if worst_av:
        print(f"  max rel err av vs probs@V : {worst_av:.3e} "
              "(fp16 V, so ~1e-3 is expected)")
    else:
        print("  V not stored -- cannot verify av; rerun with --save-v to score estimators")

    # ---- population breakdown
    nk = [r["nk"] for r in d.index]
    print(f"\ncontext length nk: min={min(nk)} max={max(nk)} {quantiles(nk)}")

    L = max(r["layer"] for r in d.index) + 1
    edges = np.linspace(0, L, args.bands + 1).astype(int)
    print(f"\n{'layers':>10} {'n':>6} {'top1':>8} {'top32':>8} {'tok0':>8} "
          f"{'entropy':>9} {'eff.supp':>9}")
    for i in range(args.bands):
        lo, hi = edges[i], edges[i + 1] - 1
        recs = [r for r in d.index if lo <= r["layer"] <= hi]
        if not recs:
            continue
        f = lambda k: float(np.mean([r[k] for r in recs]))
        print(f"{f'{lo}-{hi}':>10} {len(recs):>6} {f('top1_mass'):>8.3f} "
              f"{f('top32_mass'):>8.3f} {f('mass_on_token0'):>8.3f} "
              f"{f('entropy_nats'):>9.3f} {math.exp(f('entropy_nats')):>9.1f}")

    top1 = [r["top1_mass"] for r in d.index]
    t32 = [r["top32_mass"] for r in d.index]
    print(f"\noverall top1_mass  {quantiles(top1)}")
    print(f"overall top32_mass {quantiles(t32)}")
    peaked = sum(1 for v in t32 if v > 0.95)
    diffuse = sum(1 for v in t32 if v < 0.5)
    print(f"\npeaked (top32 mass > 0.95): {peaked} ({100*peaked/len(d):.1f}%)")
    print(f"diffuse (top32 mass < 0.50): {diffuse} ({100*diffuse/len(d):.1f}%)")
    print("\nThese two groups behave very differently under sampling. Report "
          "sampler results per group, not pooled.")


if __name__ == "__main__":
    main()
