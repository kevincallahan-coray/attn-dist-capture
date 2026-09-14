"""Run Llama-3.1-8B-Instruct on long-context prompts and dump a stratified
random sample of decode-time attention distributions.

Generation itself is exactly dense (HF SDPA); capture is a side channel, so the
distributions are the ones a normal dense run would produce.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attn_capture import CONTROLLER, CaptureConfig, register_backend
from attn_store import ShardWriter

FILLER = (
    "The archivist unlocked the reading room before dawn and began the slow work of "
    "sorting the previous week's accessions. Each folder carried a provenance note, a "
    "date, and a short description of the correspondence it held. She worked steadily, "
    "moving from the earliest boxes to the most recent, pausing only to record the "
    "identifiers in the ledger beside the window. "
)


def load_prompts(args, tok) -> list[dict]:
    if args.dataset:
        rows = []
        with open(args.dataset) as f:
            for i, line in enumerate(f):
                if len(rows) >= args.max_examples:
                    break
                if not line.strip():
                    continue
                rec = json.loads(line)
                text = rec.get("input") or rec.get("prompt")
                if text is None:
                    raise ValueError("dataset rows need an 'input' or 'prompt' field")
                rows.append({"source": "dataset", "index": rec.get("index", i),
                             "prompt": text})
        return rows

    # Synthetic fallback so a smoke test needs no data on the PVC.
    target = args.synthetic_tokens
    unit = len(tok(FILLER, add_special_tokens=False)["input_ids"])
    reps = max(1, target // max(1, unit) + 2)
    rows = []
    for i in range(args.max_examples):
        body = FILLER * reps
        prompt = (f"Document {i}:\n{body}\n\n"
                  "Summarize the document above in two sentences.")
        rows.append({"source": "synthetic", "index": i, "prompt": prompt})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dataset", default=None,
                   help="JSONL with 'input' or 'prompt' (e.g. RULER validation.jsonl)")
    p.add_argument("--synthetic-tokens", type=int, default=4096,
                   help="approx prompt length when --dataset is omitted")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--max-examples", type=int, default=8)
    p.add_argument("--max-input-tokens", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=64)
    # capture knobs
    p.add_argument("--step-prob", type=float, default=0.25)
    p.add_argument("--pairs-per-step", type=int, default=8)
    p.add_argument("--no-stratify", action="store_true")
    p.add_argument("--save-v", action="store_true",
                   help="also store V rows (fp16) needed to score an estimator")
    p.add_argument("--v-fraction", type=float, default=1.0,
                   help="fraction of captured records that store V; V dominates "
                        "dump size at long context")
    p.add_argument("--save-scores", action="store_true",
                   help="also store pre-softmax masked scores")
    p.add_argument("--max-records", type=int, default=100_000)
    p.add_argument("--shard-records", type=int, default=512)
    p.add_argument("--no-compress", action="store_true")
    p.add_argument("--seed", type=int, default=1690)
    args = p.parse_args()

    register_backend()
    token = os.getenv("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(args.model, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, token=token, dtype=torch.bfloat16,
        attn_implementation="capture_dense", low_cpu_mem_usage=True,
    ).cuda().eval()

    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_heads = cfg.num_attention_heads
    print(f"model: {num_layers} layers x {num_heads} q-heads, "
          f"{cfg.num_key_value_heads} kv-heads, head_dim={cfg.hidden_size // num_heads}",
          flush=True)

    out_dir = pathlib.Path(args.out_dir)
    sink = ShardWriter(out_dir, shard_records=args.shard_records,
                       compress=not args.no_compress)
    cap_cfg = CaptureConfig(
        step_prob=args.step_prob,
        pairs_per_step=args.pairs_per_step,
        stratify_layers=not args.no_stratify,
        save_v=args.save_v,
        v_fraction=args.v_fraction,
        save_scores=args.save_scores,
        seed=args.seed,
        max_records=args.max_records,
    )
    CONTROLLER.configure(cap_cfg, num_layers=num_layers, num_heads=num_heads, sink=sink)

    # Up-front size projection: V is the term that explodes at long context.
    nk = args.max_input_tokens
    d_head = cfg.hidden_size // num_heads
    per_rec = 4 * nk + 4 * d_head + (4 * nk if args.save_scores else 0)
    per_v = (2 * nk * d_head) if args.save_v else 0
    # ~1/groups of records share a V, and only v_fraction of those store it.
    groups = num_heads // cfg.num_key_value_heads
    proj = args.max_records * (per_rec + per_v * args.v_fraction / max(1, groups))
    print(f"projected uncompressed dump size at max_records: {proj/1e9:.2f} GB "
          f"(probs {args.max_records*per_rec/1e9:.2f} GB)", flush=True)

    prompts = load_prompts(args, tok)
    print(f"{len(prompts)} prompts from {prompts[0]['source'] if prompts else 'n/a'}",
          flush=True)

    gen_log = out_dir / "generations.jsonl"
    t_start = time.time()
    for i, row in enumerate(prompts):
        enc = tok(row["prompt"], return_tensors="pt", truncation=True,
                  max_length=args.max_input_tokens, add_special_tokens=True)
        enc = {k: v.cuda() for k, v in enc.items()}
        n_tok = enc["input_ids"].shape[-1]

        CONTROLLER.begin_example(row["index"])
        torch.cuda.synchronize()
        t0 = time.time()
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                             do_sample=False, use_cache=True,
                             pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        dt = time.time() - t0
        CONTROLLER.end_example()

        text = tok.decode(out[0, n_tok:], skip_special_tokens=True).strip()
        with open(gen_log, "a") as g:
            g.write(json.dumps({
                "index": row["index"], "source": row["source"],
                "prompt_tokens": n_tok, "seconds": dt, "generation": text,
            }) + "\n")
        s = CONTROLLER.summary()
        print(f"[{i+1}/{len(prompts)}] tokens={n_tok} sec={dt:.2f} "
              f"records={s['records_written']} answer={text[:80]!r}", flush=True)

        if CONTROLLER.records_written >= args.max_records:
            print("record cap reached, stopping early", flush=True)
            break

    manifest = {
        "model": args.model,
        "num_layers": num_layers,
        "num_q_heads": num_heads,
        "num_kv_heads": cfg.num_key_value_heads,
        "head_dim": cfg.hidden_size // num_heads,
        "dtype": "bfloat16",
        "attn_implementation": "capture_dense (SDPA forward + fp32 capture)",
        "probs_dtype": "float32",
        "dataset": args.dataset,
        "synthetic_tokens": None if args.dataset else args.synthetic_tokens,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "capture": vars(cap_cfg) if hasattr(cap_cfg, "__dict__") else cap_cfg.__dict__,
        "stats": CONTROLLER.summary(),
        "elapsed_seconds": time.time() - t_start,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
    }
    sink.close(manifest)
    print("done:", json.dumps(manifest["stats"]), flush=True)
    print(f"wrote {sink.bytes_written/1e6:.1f} MB to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
