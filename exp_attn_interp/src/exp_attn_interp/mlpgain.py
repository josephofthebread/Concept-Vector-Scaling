from __future__ import annotations

import collections
import json
import logging
import time
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file

from exp_attn_interp.common import load_model, story_span
from exp_attn_interp.config import GENDATA_ROOT, INSTRUCT_MODEL, vector_tag_for
from exp_attn_interp.decomp import POOL_MIN, POOL_SKIP
from exp_attn_interp.vectors import load_diff_vectors
from exp_attn_interp.write import story_ids

log = logging.getLogger("mlpgain")
TRIGGER = Path("../exp_trigger_prompts")
EPS = ((0.1, 1.0), (0.25, 1.0), (1.0, 1.0))
TYPES = ("pole0", "pole1", "neutral")
DIRS = ("v_k(b)", "v_k(last)", "random", "v_other(b)", "e_gap", "u_gap", "mean_resid", "shared", "specific")


class PreMlpCatcher:
    """The residual entering each block's MLP (input of post_attention_layernorm)."""

    def __init__(self, model) -> None:
        n = len(model.layers)
        self.x: list[torch.Tensor | None] = [None] * n
        for i, blk in enumerate(model.layers):
            blk.post_attention_layernorm.register_forward_pre_hook(self._pre(i))

    def _pre(self, i):
        def hook(_m, inputs):
            self.x[i] = inputs[0].detach()

        return hook


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def dsilu(x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(x)
    return s * (1 + x * (1 - s))


@torch.inference_mode()
def block_response(blk, x: torch.Tensor, dirs: torch.Tensor, eps_list, want_neuron: bool):
    """x [S,hid] fp32 residual before the MLP; dirs [D,hid] unit. Returns gain [D, n_eps], resp [D, hid]
    (eps_list[0]), gain_up/gain_gate [D] (analytic linear split), neuron [I] for dirs[0], Q_up/Q_gate [D]."""
    ln = blk.post_attention_layernorm
    gamma = ln.weight.to(torch.float32)
    eps_n = float(ln.variance_epsilon) if hasattr(ln, "variance_epsilon") else float(ln.eps)
    Wg = blk.mlp.gate_proj.weight.to(torch.float32)
    Wu = blk.mlp.up_proj.weight.to(torch.float32)
    Wd = blk.mlp.down_proj.weight.to(torch.float32)

    def mlp(z):
        zn = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps_n) * gamma
        g, u = zn @ Wg.T, zn @ Wu.T
        a = silu(g) * u
        return a @ Wd.T, a, g, u

    base, a0, g0, u0 = mlp(x)
    scale = float(x.norm(dim=-1).median())
    D = dirs.shape[0]
    gain = torch.zeros(D, len(eps_list), device=x.device)
    resp = torch.zeros(D, x.shape[1], device=x.device)
    neuron = None
    wd = dirs @ Wd
    for j in range(D):
        d = dirs[j]
        for e, (eps, sign) in enumerate(eps_list):
            y, a1, _, _ = mlp(x + (sign * eps * scale) * d)
            r = (y - base) / (sign * eps * scale)
            gain[j, e] = (r @ d).mean()
            if e == 0:
                resp[j] = r.mean(0)
                if j == 0 and want_neuron:
                    neuron = (((a1 - a0) / (sign * eps * scale)).mean(0)) * wd[0]
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps_n)
    dt = (gamma * dirs)[:, None, :] * rms[None]
    pg = torch.einsum("dsh,ih->dsi", dt, Wg)
    pu = torch.einsum("dsh,ih->dsi", dt, Wu)
    up_path = (silu(g0)[None] * pu * wd[:, None, :]).sum(-1).mean(-1)
    gate_path = ((dsilu(g0) * u0)[None] * pg * wd[:, None, :]).sum(-1).mean(-1)
    q_up = ((gamma * dirs) @ Wu.T * wd).sum(-1)
    q_gate = ((gamma * dirs) @ Wg.T * wd).sum(-1)
    return gain, resp, up_path, gate_path, neuron, q_up, q_gate


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--per-pole", type=int, default=4)
    parser.add_argument("--neutral", type=int, default=2)
    parser.add_argument("--every", type=int, default=4, help="every n-th concept")
    parser.add_argument("--out", type=Path, default=Path("work/mlpgain"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--markers", type=Path, default=Path("data/markers_presence.json"))
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(offline=args.offline, model_name=args.model)
    pre = PreMlpCatcher(model.model)
    vectors = load_diff_vectors(device=str(model.device), tag=vector_tag_for(args.model))
    K = vectors.shape[1]
    B = len(model.model.layers)
    mk = json.loads(args.markers.read_text())
    hid = model.config.hidden_size
    dev = model.device
    U = model.lm_head.weight.to(torch.float32) * model.model.norm.weight.to(torch.float32)[None, :]
    Emb = model.model.embed_tokens.weight.to(torch.float32)
    gaps: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for kk in range(K):
        ic, ia = mk["concept"].get(str(kk), []), mk["antagonist"].get(str(kk), [])
        if len(ic) >= 2 and len(ia) >= 2:
            gaps[kk] = (
                torch.nn.functional.normalize(Emb[ic].mean(0) - Emb[ia].mean(0), dim=0),
                torch.nn.functional.normalize(U[ic].mean(0) - U[ia].mean(0), dim=0),
            )
    del U, Emb
    torch.cuda.empty_cache()
    shared = torch.nn.functional.normalize(vectors.mean(1), dim=-1)
    tbl = pq.read_table(
        GENDATA_ROOT / "data" / "semantic.parquet",
        columns=["pair_id", "concept_text", "antagonist_text"],
        filters=[("language_id", "==", 0)],
    )
    by_pair: dict[int, list[tuple[str, str]]] = {}
    for p, c, a in zip(tbl["pair_id"].to_pylist(), tbl["concept_text"].to_pylist(), tbl["antagonist_text"].to_pylist()):
        by_pair.setdefault(int(p), []).append((c, a))
    neutral: dict[int, list[str]] = collections.defaultdict(list)
    with (TRIGGER / "work" / "prompts.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            if r["kind"] == "neutral" and r.get("ok") and r.get("prompt"):
                neutral[int(r["pair_id"])].append(r["prompt"])
    kw = {"enable_thinking": False} if "Qwen3" in args.model else {}
    pairs = sorted(by_pair)[:: args.every]
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    started = time.monotonic()

    for n, k in enumerate(pairs):
        path = args.out / f"pair-{k:04d}.safetensors"
        if path.exists():
            continue
        other = (k + K // 2) % K
        rnd = torch.randn(hid, generator=gen).to(dev)
        rnd = rnd / rnd.norm()
        e_gap, u_gap = gaps.get(k, (rnd, rnd))
        nd = len(DIRS)
        acc_gain = torch.zeros(3, B, nd, len(EPS), device=dev)
        acc_resp = torch.zeros(2, B, 2, hid, device=dev)
        acc_path = torch.zeros(3, B, nd, 2, device=dev)
        acc_neur = torch.zeros(2, B, model.config.intermediate_size, device=dev)
        q_up = torch.zeros(B, nd, device=dev)
        q_gate = torch.zeros(B, nd, device=dev)
        counts = torch.zeros(3, device=dev)
        texts: list[tuple[int, list[int], tuple[int, int]]] = []
        for c, a in by_pair[k][: args.per_pole]:
            for side, text in ((0, c), (1, a)):
                ids = story_ids(tokenizer, text)
                s0, s1 = story_span(tokenizer, ids, text)
                span = (s0 + POOL_SKIP, s1) if (s1 - s0 - POOL_SKIP) >= POOL_MIN else (s0, s1)
                texts.append((side, ids, span))
        for p in neutral.get(k, [])[: args.neutral]:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, **kw
            )
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            try:
                s0, s1 = story_span(tokenizer, ids, p)
            except RuntimeError:
                continue
            texts.append((2, ids, (s0, s1)))
        for side, ids, (s0, s1) in texts:
            model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
            typ = side
            rt = 0 if side < 2 else 1
            for b in range(B):
                xb = pre.x[b]
                assert xb is not None
                x = xb[0, s0:s1].to(torch.float32)
                xm = torch.nn.functional.normalize(x.mean(0), dim=0)
                spec = vectors[b, k] - (vectors[b, k] @ shared[b]) * shared[b]
                dirs = torch.stack(
                    [
                        vectors[b, k],
                        vectors[B - 1, k],
                        rnd,
                        vectors[b, other],
                        e_gap,
                        u_gap,
                        xm,
                        shared[b],
                        torch.nn.functional.normalize(spec, dim=0),
                    ]
                )
                gain, resp, up, gate, neur, qu, qg = block_response(model.model.layers[b], x, dirs, EPS, True)
                acc_gain[typ, b] += gain
                acc_resp[rt, b] += resp[:2]
                acc_path[typ, b, :, 0] += up
                acc_path[typ, b, :, 1] += gate
                assert neur is not None
                acc_neur[rt, b] += neur
                if counts.sum() == 0:
                    q_up[b], q_gate[b] = qu, qg
            counts[typ] += 1
        cs = counts.clamp_min(1)
        acc_gain /= cs[:, None, None, None]
        acc_path /= cs[:, None, None, None]
        rc = torch.stack([counts[0] + counts[1], counts[2]]).clamp_min(1)
        acc_resp /= rc[:, None, None, None]
        acc_neur /= rc[:, None, None]
        save_file(
            {
                "gain": acc_gain.cpu(),
                "resp": acc_resp.to(torch.bfloat16).cpu(),
                "path": acc_path.cpu(),
                "neuron": acc_neur.to(torch.float16).cpu(),
                "q_up": q_up.cpu(),
                "q_gate": q_gate.cpu(),
                "counts": counts.cpu(),
                "pair_id": torch.tensor(k),
                "other": torch.tensor(other),
            },
            str(path),
            metadata={"dirs": ",".join(DIRS), "eps": "0.1,0.25,1.0", "types": ",".join(TYPES)},
        )
        if n < 3 or n % 20 == 0:
            g = acc_gain[:2].mean(0)[:, 0, 0].tolist()
            log.info(
                "%d/%d pair %d: gain along v_k(b) on stories by block: %s | linear split up/gate at b=12: %.3f/%.3f; %.1f min",
                n,
                len(pairs),
                k,
                [round(v, 3) for v in g[::3]],
                float(acc_path[:2, 12, 0, 0].mean()),
                float(acc_path[:2, 12, 0, 1].mean()),
                (time.monotonic() - started) / 60,
            )
    log.info("done %d pairs, %.1f min", len(pairs), (time.monotonic() - started) / 60)


if __name__ == "__main__":
    main()
