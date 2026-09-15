# Reading the output tables

## `rates.csv` and the convergence table

One row per (stratum, order, sampler). The table fits

    error(S) ≈ C · S^slope

by least squares on log(error) vs log(S) across all budgets in the sweep.

| field | meaning |
|---|---|
| `stratum` | Which population the record came from. `peaked` = top-32 tokens hold >95% of the mass (attention-sink behaviour), `diffuse` = they hold <50%, `mid` = in between. |
| `order` | `natural` = tokens in position order. `shuffled` = the same distribution with the index axis randomly permuted. |
| `sampler` | Method. A `+burn10S` suffix means 10·S steps were discarded before collecting. |
| `slope` | **How fast error falls as you add samples.** −0.5 is plain Monte Carlo. More negative is better. |
| `C` | The fitted error at S=1 — the constant factor. Two samplers with the same slope but different C differ by a fixed ratio at every budget. |
| `S=…` | Mean error at that budget, in whatever metric was selected. |

### What the slope means

Because the fit is in log space, `slope` is the straight-line gradient on a
log-log plot: **a 10× increase in samples multiplies error by 10^slope.**

| slope | error after 10× more samples | reading |
|---|---|---|
| −0.50 | ×0.32 | i.i.d. Monte Carlo. The CLT floor: error ~ σ/√S. |
| −0.80 | ×0.16 | Beating i.i.d. — variance reduction is working. |
| −0.25 | ×0.56 | Worse than i.i.d. Correlated draws. |
| ≈ 0 | ×1.0 | **Not converging.** More samples buy nothing. |

A slope near zero is the important signal: it means the estimator is not
consistent at the budgets tested. That is qualitatively different from being
merely inefficient, and no amount of extra sampling fixes it.

Slope and C answer different questions. A sampler that only improves `C` saves
a constant factor. One that steepens `slope` changes what is reachable at a
given budget. Comparing at a single S conflates the two.

## Diagnostic columns in `cells.csv`

| field | meaning |
|---|---|
| `n_records` | How many captured distributions went into this cell. |
| `argmax_hit_rate` | Fraction of trials that drew the single highest-probability token **at least once**. On peaked records, missing it means the estimate omits most of the true mass. |
| `distinct_mean` | Average number of distinct token indices drawn. |
| `distinct_frac` | `distinct_mean / S` — the fraction of draws that landed somewhere new. |
| `accept_rate` | MCMC only: fraction of proposed moves accepted. Blank/`nan` for the CDF samplers, which have no accept step. |
| `tv_distance` | Total variation between the empirical draw frequencies and the true distribution: `0.5 · Σ|freq − p|`. Ranges 0 to 1. |

### `distinct_frac` — careful, low is not automatically bad

It measures repeats, and repeats have two very different causes.

*Correct repeats.* On a peaked distribution where one token holds 90% of the
mass, a correct sampler **should** return that token over and over. i.i.d. at
S=8 on the peaked stratum shows `distinct_frac ≈ 0.15`, and that is right.

*Pathological repeats.* A random walk that keeps stepping back and forth
between neighbours also repeats, but because it is stuck, not because the
target says so.

The way to tell them apart is to compare against the **i.i.d. row at the same S
and stratum** — that is what correct repetition looks like. If a sampler has a
much lower `distinct_frac` than i.i.d. in the same cell, it is stuck. For a
nearest-neighbour walk this is the direct measurement of the mixing problem: a
±1 walk on a ring of `nk` states is diffusive, so in S steps it reaches only
O(√S) distinct positions regardless of what the distribution looks like.

### `accept_rate` — diagnostic, not a score

The fraction of proposed moves the chain accepted. It does **not** measure
quality; it explains behaviour.

- **Very low** (say <5%): the chain is frozen, rejecting nearly everything.
- **Very high** (say >90%) with a nearest-neighbour proposal: moves are cheap
  to accept because neighbouring tokens have similar probability, so the chain
  wanders locally without ever crossing to another region. High acceptance and
  terrible mixing coexist happily here.

A healthy acceptance rate with a flat error slope is the most damning
combination available: the chain is running fine mechanically and still not
sampling the target.

### `tv_distance` — compare it, never read it absolutely

`0.5 · Σ|empirical_freq − p|` over all `nk` tokens. 0 means the draw
frequencies match the distribution exactly; 1 means no overlap.

**It is not zero for a perfect sampler.** With S draws you have at most S
non-zero frequencies, and the target spreads mass over `nk` tokens. At S=64 and
nk=4096 even flawless i.i.d. sampling has TV close to 1 purely from the
counting limit. So:

> Always read `tv_distance` against the i.i.d. row at the same S and stratum.
> That row is the achievable floor.

A sampler at TV 0.99 where i.i.d. is at 0.98 is fine. A sampler at TV 1.00
where i.i.d. is at 0.30 has visited an essentially disjoint part of the index
space from where the mass is.

## Error metrics

All are computed on the attention output `AV`, not on index histograms, since
`AV` is what propagates into the model.

| metric | definition | when to use |
|---|---|---|
| `abs_mean` | `‖ÂV − AV‖` | Raw magnitude. Not comparable across records with different scales. |
| `rel_mean` | `‖ÂV − AV‖ / ‖AV‖` | Intuitive, but **collapses on diffuse records**: near-uniform weights average roughly zero-mean value rows toward zero, so `‖AV‖` goes small and relative error blows up for reasons unrelated to the sampler. |
| `scaled_mean` | `‖ÂV − AV‖ / (‖V‖_F/√nk)` | Divided by the RMS norm of a single value row. Independent of cancellation, so it is comparable across strata. **The default for cross-stratum claims.** |
| `abs_p90`, `abs_p99` | 90th/99th percentile over trials | Tail behaviour. A sampler that is usually fine and occasionally catastrophic has a good mean and a bad p99. |

## The shuffle table

Prints `error(shuffled) / error(natural)` per cell.

- **≈ 1.0** — index ordering is irrelevant to this sampler. i.i.d. must land
  here; if it does not, something is wrong with the harness.
- **> 1.0** — the sampler got worse when position structure was destroyed, so
  it was exploiting the fact that nearby tokens have similar value rows.
- **< 1.0** — shuffling helped, which for a random-walk sampler can happen
  because a permuted index axis turns local steps into global jumps.

This matters for generalization. An advantage that vanishes under shuffling is
an advantage that depends on positional structure in this model at this context
length, and may not transfer.
