"""Attention-distribution capture backend for Llama-3.1 on transformers.

Design notes
------------
* The real forward pass is *always* delegated to HF SDPA, so generation is
  bit-identical to a normal dense run. Capture is a side channel.
* Capture only happens when ``q_len == 1`` (autoregressive decode). Prefill is
  never instrumented: materializing O(L^2) scores at 4k-32k context is both
  pointless here and a memory hazard.
* Only the *selected* (layer, head) pairs have their score vector computed, so
  per captured decode step the extra work is O(selected_heads * nk), not
  O(32 * 32 * nk).
* Probabilities are computed and stored in fp32. fp16 would flush a large part
  of the distribution tail to exactly zero at long context, which is precisely
  the region sampler quality depends on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class CaptureConfig:
    # Fraction of decode steps that get instrumented at all.
    step_prob: float = 0.25
    # Number of (layer, head) pairs captured per instrumented decode step.
    pairs_per_step: int = 8
    # Stratify the pair draw so layers are covered evenly rather than
    # uniformly at random (uniform draws cluster badly at small budgets).
    stratify_layers: bool = True
    # Save the V rows (nk x head_dim, fp16) needed to score an estimator.
    save_v: bool = False
    # Fraction of captured records that get V stored. V dominates dump size at
    # long context (32k x 128 fp16 = 8.4 MB per unique layer/kv-head/step), so
    # keep every distribution but only enough V to measure estimator error.
    v_fraction: float = 1.0
    # Storage dtype for V. fp16 halves the dump; fp32 is exact but doubles it.
    v_dtype: str = "fp16"
    # Save the pre-softmax masked+scaled scores alongside the probabilities.
    save_scores: bool = False
    seed: int = 0
    max_records: int = 100_000


class CaptureController:
    """Holds capture state and the per-example decode-step plan."""

    def __init__(self) -> None:
        self.config = CaptureConfig()
        self.active = False
        self.num_layers: int | None = None
        self.num_heads: int | None = None
        self._rng = np.random.default_rng(0)
        self._v_rng = np.random.default_rng(1)
        self._step = -1
        self._plan: dict[int, list[int]] = {}
        self.example_id: int = -1
        self.sink: "ShardWriter | None" = None
        self.records_written = 0
        self.steps_seen = 0
        self.steps_captured = 0

    # ---------------------------------------------------------------- config

    def configure(self, cfg: CaptureConfig, *, num_layers: int, num_heads: int,
                  sink: "ShardWriter") -> None:
        self.config = cfg
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.sink = sink
        self._rng = np.random.default_rng(cfg.seed)
        # Separate stream so changing v_fraction does not perturb which
        # (layer, head, step) records get captured.
        self._v_rng = np.random.default_rng(cfg.seed + 1_000_003)
        self.records_written = 0
        self.steps_seen = 0
        self.steps_captured = 0

    def begin_example(self, example_id: int) -> None:
        self.example_id = example_id
        self._step = -1
        self._plan = {}
        self.active = True

    def end_example(self) -> None:
        self.active = False
        self._plan = {}

    # ------------------------------------------------------------ step plan

    def _draw_plan(self) -> dict[int, list[int]]:
        cfg = self.config
        L, H = self.num_layers, self.num_heads
        n = min(cfg.pairs_per_step, L * H)
        if cfg.stratify_layers:
            # Spread the budget across evenly spaced layer strata, then pick a
            # random layer inside each stratum and a random head within it.
            edges = np.linspace(0, L, n + 1)
            layers = [
                int(self._rng.integers(int(math.floor(edges[i])),
                                       max(int(math.floor(edges[i])) + 1,
                                           int(math.ceil(edges[i + 1])))))
                for i in range(n)
            ]
            layers = [min(x, L - 1) for x in layers]
        else:
            layers = list(self._rng.integers(0, L, size=n))
        heads = self._rng.integers(0, H, size=n)
        plan: dict[int, list[int]] = {}
        for layer, head in zip(layers, heads):
            plan.setdefault(int(layer), []).append(int(head))
        return {k: sorted(set(v)) for k, v in plan.items()}

    def note_layer(self, layer_idx: int) -> None:
        """Called on every decode-time attention call, before capture."""
        if layer_idx != 0:
            return
        self._step += 1
        self.steps_seen += 1
        if self.records_written >= self.config.max_records:
            self._plan = {}
            return
        if self._rng.random() < self.config.step_prob:
            self._plan = self._draw_plan()
            self.steps_captured += 1
        else:
            self._plan = {}

    def roll_v(self) -> bool:
        """Independent coin for whether this record stores its V rows."""
        return bool(self._v_rng.random() < self.config.v_fraction)

    def heads_for(self, layer_idx: int) -> list[int]:
        return self._plan.get(layer_idx, [])

    @property
    def step(self) -> int:
        return self._step

    def summary(self) -> dict:
        return {
            "records_written": self.records_written,
            "decode_steps_seen": self.steps_seen,
            "decode_steps_captured": self.steps_captured,
        }


CONTROLLER = CaptureController()


def _kv_head(head: int, num_key_value_groups: int) -> int:
    return head // max(1, num_key_value_groups)


_V_OVERFLOW_WARNED = False


def _v_to_numpy(v_mat: "torch.Tensor", dtype: str = "fp16") -> np.ndarray:
    """Convert V to numpy. Always goes through fp32 first: the model tensor is
    bfloat16 and numpy has no bf16 dtype, so a direct .numpy() raises
    ``TypeError: Got unsupported ScalarType BFloat16``.

    bf16 carries a far wider exponent range than fp16 (~3.4e38 vs 65504), so
    the narrowing is checked once rather than assumed safe.
    """
    global _V_OVERFLOW_WARNED
    arr = v_mat.detach().float().cpu().numpy()
    if dtype == "fp32":
        return arr
    out = arr.astype(np.float16)
    finite_in = np.isfinite(arr)
    if not _V_OVERFLOW_WARNED and not np.isfinite(out[finite_in]).all():
        _V_OVERFLOW_WARNED = True
        print("WARNING: V rows overflowed fp16 and were clipped to inf. "
              "Rerun with --v-dtype fp32.", flush=True)
    return out


def _describe(probs32: torch.Tensor) -> dict:
    """Cheap on-GPU summary stats, so the index is sliceable without loading arrays."""
    p = probs32
    nk = p.numel()
    k = min(32, nk)
    top = torch.topk(p, k)
    logp = torch.log(p.clamp_min(1e-30))
    ent = float(-(p * logp).sum().item())
    return {
        "nk": int(nk),
        "entropy_nats": ent,
        "perplexity": float(math.exp(min(ent, 700.0))),
        "top1_mass": float(top.values[0].item()),
        "top1_index": int(top.indices[0].item()),
        "top8_mass": float(top.values[: min(8, k)].sum().item()),
        "top32_mass": float(top.values.sum().item()),
        "mass_on_token0": float(p[0].item()),
        "nnz_above_1e6": int((p > 1e-6).sum().item()),
    }


def capture_attention_forward(module, query, key, value, attention_mask, scaling,
                              dropout=0.0, **kwargs):
    """SDPA forward with a decode-time attention-distribution side channel."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    out = sdpa_attention_forward(
        module, query, key, value, attention_mask,
        scaling=scaling, dropout=dropout, **kwargs
    )

    if query.shape[-2] != 1:
        return out  # prefill: never instrumented

    layer_idx = int(getattr(module, "layer_idx", 0))
    CONTROLLER.note_layer(layer_idx)
    if not CONTROLLER.active:
        return out
    heads = CONTROLLER.heads_for(layer_idx)
    if not heads:
        return out

    cfg = CONTROLLER.config
    groups = int(getattr(module, "num_key_value_groups", 1))
    b = query.shape[0]
    if b != 1:
        raise RuntimeError("capture requires batch size 1")

    with torch.no_grad():
        for head in heads:
            kvh = _kv_head(head, groups)
            q = query[0, head, 0, :]                       # [d]
            k_mat = key[0, kvh]                            # [nk, d]
            scores = (k_mat.float() @ q.float()) * float(scaling)
            if attention_mask is not None:
                m = attention_mask
                m = m[0, head % m.shape[1], 0] if m.dim() == 4 else m.reshape(-1)
                if m.dtype == torch.bool:
                    scores = scores.masked_fill(~m.reshape(-1), float("-inf"))
                else:
                    scores = scores + m.reshape(-1).float()
            probs = torch.softmax(scores, dim=-1)
            v_mat = value[0, kvh]                          # [nk, d]
            av = (probs.unsqueeze(0) @ v_mat.float()).reshape(-1)

            meta = {
                "example_id": CONTROLLER.example_id,
                "decode_step": CONTROLLER.step,
                "layer": layer_idx,
                "head": head,
                "kv_head": kvh,
                "num_key_value_groups": groups,
                "head_dim": int(q.numel()),
                "scaling": float(scaling),
                **_describe(probs),
            }
            arrays = {
                "probs": probs.float().cpu().numpy(),
                "av": av.float().cpu().numpy(),
            }
            if cfg.save_scores:
                arrays["scores"] = scores.float().cpu().numpy()
            v_key = None
            if cfg.save_v:
                # nk is in the key so a shape mismatch can never alias two
                # different V matrices onto one shared entry.
                key = (f"v/{CONTROLLER.example_id}/{CONTROLLER.step}"
                       f"/{layer_idx}/{kvh}/{v_mat.shape[0]}")
                # If a sibling head in this GQA group already stored this V,
                # point at it for free; otherwise roll for it.
                if CONTROLLER.sink.has_v(key):
                    v_key = key
                elif cfg.v_fraction >= 1.0 or CONTROLLER.roll_v():
                    v_key = key
                    arrays["__v_shared__"] = (
                        key, _v_to_numpy(v_mat, cfg.v_dtype))
            meta["v_key"] = v_key
            CONTROLLER.sink.add(meta, arrays)
            CONTROLLER.records_written += 1


    return out


def register_backend() -> None:
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import sdpa_mask

    AttentionInterface.register("capture_dense", capture_attention_forward)
    AttentionMaskInterface.register("capture_dense", sdpa_mask)
