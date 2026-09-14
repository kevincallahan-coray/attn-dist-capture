# attn-dist-capture

Capture a stratified random sample of **decode-time attention distributions**
from Meta-Llama-3.1-8B-Instruct on NRP Nautilus, so that samplers (SANTA,
S²ANTA-strat, S²ANTA-sys, top-k, and whatever comes next) can be benchmarked
offline against real distributions instead of Gaussian synthetic inputs.

Companion to [adaptive-SANTA](https://github.com/kevincallahan-coray/adaptive-SANTA).
Storage goes under `/work/attn-dist/` on the existing `kevin-workspace` PVC.

---

## What gets saved, and why

| Array | dtype | shape | Why |
|---|---|---|---|
| `probs` | **float32** | `[nk]` | The distribution itself. fp32 because at 4k–32k context a real chunk of the tail sits below 6e-8 and would flush to zero in fp16 — that tail is exactly what distinguishes samplers. |
| `av` | float32 | `[d]` | Exact `probs @ V`. Ground truth for estimator error. |
| `V` | float16 (`--v-dtype fp32` for exact) | `[nk, d]` | Optional (`--save-v`). **Without it you cannot score an estimator** — only the index distribution. Deduped per GQA group (32 q-heads / 8 kv-heads = 4× saving) and subsettable with `--v-fraction`. |
| `scores` | float32 | `[nk]` | Optional (`--save-scores`). Pre-softmax masked scores, if you want to re-do softmax at another precision. |

Each record also carries flat metadata in `index.jsonl`: `layer`, `head`,
`kv_head`, `decode_step`, `nk`, `entropy_nats`, `top1_mass`, `top8_mass`,
`top32_mass`, `mass_on_token0`. You can select a stratum by reading only the
index and then touch just the shards you need.

### Sampling design

Attention distributions are **not one population**. Attention sinks make early
layers and certain heads extremely peaked (most mass on token 0) while
mid-network heads are close to diffuse. A uniform pool of "random
distributions" averages over regimes that behave completely differently under
sampling, and a sampler benchmarked on the pool is benchmarked on nothing in
particular. So:

- decode steps are instrumented with probability `--step-prob`, giving coverage
  across the growing context length `nk`;
- within an instrumented step, `--pairs-per-step` (layer, head) pairs are drawn
  with **layer stratification on by default** — uniform draws cluster badly at
  small budgets;
- one dump per RULER task, since task identity is a real covariate.

Report sampler results per stratum. `inspect_dump.py` prints the peaked/diffuse
split to make the shape of your population explicit.

### Correctness

The real forward pass is always HF SDPA, so **generation is bit-identical to a
normal dense run** — the captured distributions are the ones a dense run
produces, not artifacts of an instrumented kernel. Capture is a side channel
that computes score vectors only for the selected (layer, head) pairs, so the
overhead is `O(pairs * nk)` per instrumented step. Prefill is never
instrumented; materializing `O(L²)` scores at long context is a memory hazard
and isn't what SANTA targets anyway.

---

## Quick start on Nautilus

Prerequisites are the same as adaptive-SANTA: `kubectl` context set to your
namespace, an `hf-token` secret, and the `kevin-workspace` PVC bound. The jobs
point `HF_HOME` at `/work/santa-adaptive-z/hf` to reuse the Llama weights
already cached there.

```bash
# 1. Smoke test: synthetic 4k prompts, no dataset needed on the PVC
kubectl apply -f k8s/job-capture-smoke.yaml
kubectl logs -f job/attn-capture-smoke

# 2. Real capture over the four RULER tasks at 4k
kubectl apply -f k8s/job-capture-ruler-4k.yaml
kubectl logs -f job/attn-capture-ruler-4k
```

From Windows: `scripts\run-capture-smoke.bat`, `scripts\run-capture-ruler-4k.bat`.

The RULER job reads `/shared/ruler/data/4096_100/<task>/validation.jsonl` on the
`kevin-ruler-shared` CephFS volume, which `adaptive-SANTA`'s
`job-ruler-prepare-4k-100.yaml` already generates. For 32k, generate
`32768_100` data the same way, then use `k8s/job-capture-ruler-32k.yaml`.

### PVC constraint

`kevin-workspace` is `rook-ceph-block`, ReadWriteOnce. **One pod at a time.**
A capture Job, an inspect Job, and a SANTA benchmark Job cannot all mount it at
once. Run them serially. If you later want concurrent captures, write output to
the CephFS `kevin-ruler-shared` volume instead.

### PVC permissions

The `prp/jupyter-stack` image runs as uid 1000, but the PVC root is owned by
root, so a non-root process cannot create a top-level directory on it —
`mkdir: cannot create directory '/work/attn-dist': Permission denied`. Every
Job therefore runs a root `prepare-dirs` initContainer that creates the output
tree, `chgrp`s it to 1000, and sets the setgid bit so later files inherit the
group. This mirrors what `adaptive-SANTA`'s RULER prepare job does.

If you still hit permission errors, check ownership from inside the inspect
job with `ls -ld /work /work/attn-dist /work/santa-adaptive-z/hf`. A cache
directory written by a job running under a different uid is the next most
likely culprit.

### No idle pods

NRP prohibits pods that sit idle (`sleep`, `tail -f`, interactive shells parked
overnight). Every Job here does real work and exits, and carries
`activeDeadlineSeconds` as a backstop. To look at a dump, run
`k8s/job-inspect-dump.yaml` rather than opening a shell; to get data off the
PVC, run `k8s/job-export-dump.yaml` to produce a tarball and pull it during a
job that is already running, or push it to the NRP S3 store.

### Tearing a run down

```bash
kubectl delete job attn-capture-ruler-4k     # job + its pods
kubectl get pods                             # check nothing lingers
kubectl delete pod <name> --force --grace-period=0
```

Windows: `scripts\delete-run.bat` removes every job this repo creates. PVC data
is untouched; delete that from inside a job with `rm -rf /work/attn-dist/<dir>`.

---

## Output layout

```
/work/attn-dist/ruler_4096/qa_1/
    manifest.json        model, capture config, GPU, torch version, run stats
    index.jsonl          one row per record, no arrays
    shard_00000.npz      arrays for that shard's records
    generations.jsonl    what the model actually produced (sanity check)
```

Shards are self-contained: a record's `V` always lives in the same `.npz` as
its `probs`, so you can ship or delete individual shards.

## Sizing

At `nk`, per record: `4·nk` bytes of `probs` (16 KB at 4k, 128 KB at 32k) and,
if stored, `2·nk·128` bytes of `V` (1 MB at 4k, **8.4 MB at 32k**). `V`
dominates. 3000 records at 4k with full V ≈ 800 MB; the same at 32k would be
~10 GB, which is why the 32k job uses `--v-fraction 0.1`. The runner prints a
projected size before it starts.

## Consuming a dump

```python
from attn_store import AttnDump

d = AttnDump("dumps/ruler_4096/qa_1")

# diffuse mid-network distributions at long context
recs = d.select(layer=(8, 23), nk=(3000, 4096), top32_mass=lambda v: v < 0.5)

for rec in recs:
    probs = d.probs(rec)   # float32 [nk], sums to 1
    V     = d.values(rec)  # float16 [nk, 128] or None
    av    = d.av(rec)      # float32 [128], exact output
```

Validate and profile a dump:

```bash
python inspect_dump.py dumps/ruler_4096/qa_1 --check 128
```

It verifies normalization, checks `av == probs @ V`, and reports the
peaked/diffuse breakdown by layer band. On the cluster, run the same thing via
`kubectl apply -f k8s/job-inspect-dump.yaml`.

A worked estimator comparison (i.i.d. vs systematic, swept over `S`):

```bash
python example_sampler_eval.py dumps/ruler_4096/qa_1 --budgets 8,32,128
```

### One caveat on the error metric

`example_sampler_eval.py` reports relative L2, `‖ÂV − AV‖ / ‖AV‖`. In the
diffuse stratum `‖AV‖` gets small — near-uniform weights average roughly
zero-mean value rows toward zero — so relative error there is large by
construction and not directly comparable across strata. Absolute error, or
error relative to `‖V‖`/`tr(Σ)`, is the fairer cross-stratum comparison. Pick
the metric deliberately before drawing conclusions about which sampler wins
where.

## Files

```
attn_capture.py           capture backend + stratified selection controller
attn_store.py             ShardWriter / AttnDump read-write format
dump_attention.py         main runner
inspect_dump.py           validation + population stats
example_sampler_eval.py   worked estimator comparison
k8s/                      Nautilus Jobs (capture, inspect, export)
scripts/                  Windows helpers matching the adaptive-SANTA workflow
```
