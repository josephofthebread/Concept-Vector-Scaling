from __future__ import annotations

import json
import logging
import time
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors import safe_open

from exp_attn_interp.common import load_model, story_span
from exp_attn_interp.config import GENDATA_ROOT, INSTRUCT_MODEL, MARKERS_PRESENCE, vector_tag_for
from exp_attn_interp.decomp import POOL_MIN, POOL_SKIP
from exp_attn_interp.vectors import load_diff_vectors
from exp_attn_interp.write import story_ids

log = logging.getLogger("slowgain")
BLOCKS = (6, 10, 14, 18, 22, 26, 30)
EPS = (0.1, 0.25)


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


class PreMlpCatcher:
    def __init__(self, model) -> None:
        n = len(model.layers)
        self.x: list[torch.Tensor | None] = [None] * n
        for i, blk in enumerate(model.layers):
            blk.post_attention_layernorm.register_forward_pre_hook(self._pre(i))

    def _pre(self, i):
        def hook(_m, inputs):
            self.x[i] = inputs[0].detach()

        return hook


def mlp_response(blk, x: torch.Tensor, dirs: torch.Tensor, eps: float, xnorm: float, chunk: int = 8) -> torch.Tensor:
    """gain per direction: mean over tokens of <MLP(x + eps*xnorm*d) - MLP(x), d> / (eps*xnorm). x [T, hid] fp32, dirs [D, hid]."""
    ln, mlp = blk.post_attention_layernorm, blk.mlp
    w = ln.weight.to(torch.float32)
    var_eps = ln.variance_epsilon

    def f(z: torch.Tensor) -> torch.Tensor:
        n = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + var_eps) * w
        return mlp.down_proj(
            torch.nn.functional.silu(mlp.gate_proj(n.to(torch.bfloat16))) * mlp.up_proj(n.to(torch.bfloat16))
        ).to(torch.float32)

    base = f(x)
    out = torch.empty(dirs.shape[0], device=x.device)
    for c0 in range(0, dirs.shape[0], chunk):
        d = dirs[c0 : c0 + chunk]
        z = x[None] + eps * xnorm * d[:, None]
        r = f(z.reshape(-1, x.shape[1])).reshape(d.shape[0], x.shape[0], -1) - base[None]
        out[c0 : c0 + chunk] = ((r * d[:, None]).sum(-1) / (eps * xnorm)).mean(1)
    return out


@torch.inference_mode()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--n-dir-pairs", type=int, default=64)
    parser.add_argument("--n-eval", type=int, default=32, help="evaluation stories (from held-out pairs)")
    parser.add_argument(
        "--n-pca", type=int, default=192, help="stories for the between/within-story PCs (held-out pairs)"
    )
    parser.add_argument("--n-pcs", type=int, default=24)
    parser.add_argument("--out", type=Path, default=Path("work/slowgain.npz"))
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.monotonic()
    model, tokenizer = load_model(offline=args.offline, model_name=args.model)
    dev = model.device
    catcher = PreMlpCatcher(model.model)
    V = load_diff_vectors(device=str(dev), tag=vector_tag_for(args.model))
    hid = V.shape[-1]
    E = model.model.embed_tokens.weight.to(torch.float32)
    U = model.lm_head.weight.to(torch.float32) * model.model.norm.weight.to(torch.float32)[None, :]
    markers = json.loads(MARKERS_PRESENCE.read_text())
    tbl = pq.read_table(
        GENDATA_ROOT / "data" / "semantic.parquet",
        columns=["pair_id", "concept_text", "antagonist_text"],
        filters=[("language_id", "==", 0)],
    )
    by_pair: dict[int, list[tuple[str, str]]] = {}
    for p, c, a in zip(tbl["pair_id"].to_pylist(), tbl["concept_text"].to_pylist(), tbl["antagonist_text"].to_pylist()):
        by_pair.setdefault(int(p), []).append((c, a))
    all_pairs = sorted(by_pair)
    g = torch.Generator().manual_seed(20260907)
    perm = torch.randperm(len(all_pairs), generator=g).tolist()
    dir_pairs = sorted(all_pairs[i] for i in perm[: args.n_dir_pairs])
    eval_pairs = [all_pairs[i] for i in perm[args.n_dir_pairs : args.n_dir_pairs + args.n_eval // 2]]
    pca_pairs = [
        all_pairs[i]
        for i in perm[args.n_dir_pairs + args.n_eval // 2 : args.n_dir_pairs + args.n_eval // 2 + args.n_pca // 2]
    ]
    pop = torch.zeros(37, hid, device=dev)
    pop_pairs = sorted(all_pairs[i] for i in perm[-256:])
    for k in pop_pairs:
        with safe_open(f"work/vdecomp/pair-{k:04d}.safetensors", "pt") as f:
            pop += f.get_tensor("pole_resid").to(dev, torch.float32).mean(0)
    pop /= len(pop_pairs)
    topic = torch.zeros(37, len(dir_pairs), hid, device=dev)
    for i, k in enumerate(dir_pairs):
        acc = torch.zeros(37, hid, device=dev)
        for d in ("work/vdecomp", "work/vdecomp2"):
            with safe_open(f"{d}/pair-{k:04d}.safetensors", "pt") as f:
                acc += f.get_tensor("pole_resid").to(dev, torch.float32).mean(0)
        topic[:, i] = acc / 2 - pop
    topic = unit(topic)

    def run(texts):
        """per story: the pre-MLP residual at each chosen block, story-span tokens (skip rule)."""
        for text in texts:
            ids = story_ids(tokenizer, text)
            s0, s1 = story_span(tokenizer, ids, text)
            span = list(range(s0 + POOL_SKIP, s1)) if (s1 - s0 - POOL_SKIP) >= POOL_MIN else list(range(s0, s1))
            model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
            yield {b: catcher.x[b][0, span].to(torch.float32) for b in BLOCKS}

    eval_texts = [t for k in eval_pairs for t in by_pair[k][0]]
    pca_texts = [t for k in pca_pairs for t in by_pair[k][1]]
    ev = list(run(eval_texts))
    log.info("evaluation corpus: %d stories, %d tokens", len(ev), sum(e[BLOCKS[0]].shape[0] for e in ev))
    pc_between, pc_within, mean_dir = {}, {}, {}
    means = {b: [] for b in BLOCKS}
    within = {b: [] for b in BLOCKS}
    for e in run(pca_texts):
        for b in BLOCKS:
            m = e[b].mean(0)
            means[b].append(m)
            within[b].append(e[b] - m)
    for b in BLOCKS:
        M = torch.stack(means[b])
        mean_dir[b] = unit(M.mean(0))
        _, _, Wt = torch.linalg.svd(M - M.mean(0), full_matrices=False)
        pc_between[b] = Wt[: args.n_pcs]
        Wtok = torch.cat(within[b])
        _, _, Wt2 = torch.linalg.svd(
            Wtok[torch.randperm(Wtok.shape[0], generator=torch.Generator().manual_seed(1))[:20000]], full_matrices=False
        )
        pc_within[b] = Wt2[: args.n_pcs]
    log.info("PCs done, %.1f min", (time.monotonic() - t0) / 60)
    stance_Q = {}
    for b in BLOCKS:
        X = V[b]
        m = X.mean(0)
        _, _, Wt = torch.linalg.svd(X - m, full_matrices=False)
        Q, _ = torch.linalg.qr(torch.cat([unit(m)[None], Wt[:16]], 0).T)
        stance_Q[b] = Q
    idx = torch.tensor(dir_pairs, device=dev)
    egap = torch.stack(
        [
            unit(
                E[torch.tensor(markers["concept"][str(k)], device=dev)].mean(0)
                - E[torch.tensor(markers["antagonist"][str(k)], device=dev)].mean(0)
            )
            for k in dir_pairs
        ]
    )
    ugap = torch.stack(
        [
            unit(
                U[torch.tensor(markers["concept"][str(k)], device=dev)].mean(0)
                - U[torch.tensor(markers["antagonist"][str(k)], device=dev)].mean(0)
            )
            for k in dir_pairs
        ]
    )
    rnd = unit(torch.randn(len(dir_pairs), hid, generator=torch.Generator().manual_seed(7)).to(dev))
    results = {}
    for b in BLOCKS:
        groups = {
            "concept": V[b][idx],
            "topic": topic[b],
            "random": rnd,
            "e_gap": egap,
            "u_gap": ugap,
            "pc_between": pc_between[b],
            "pc_within": pc_within[b],
            "mean_dir": mean_dir[b][None],
        }
        slow, fast = pc_between[b][0], pc_within[b][0]
        fast = unit(fast - (fast @ slow) * slow)
        thetas = torch.linspace(0, np.pi / 2, 9, device=dev)
        groups["mix_slow_fast"] = torch.stack([torch.cos(t) * slow + torch.sin(t) * fast for t in thetas])
        Q = stance_Q[b]
        cv = V[b][idx]
        groups["concept_stance_part"] = unit((cv @ Q) @ Q.T)
        groups["concept_specific_part"] = unit(cv - (cv @ Q) @ Q.T)
        names, dirs = [], []
        for gname, D in groups.items():
            for j in range(D.shape[0]):
                names.append(f"{gname}:{j}")
            dirs.append(D)
        D_all = torch.cat(dirs)
        projs = [e[b] @ D_all.T for e in ev]
        allp = torch.cat(projs)
        tot_var = allp.var(0)
        story_means = torch.stack([p.mean(0) for p in projs])
        between = story_means.var(0)
        icc = between / tot_var.clamp_min(1e-12)
        ac1 = torch.stack(
            [((p[:-1] - p.mean(0)) * (p[1:] - p.mean(0))).mean(0) / p.var(0).clamp_min(1e-12) for p in projs]
        ).mean(0)
        X = torch.cat([e[b] for e in ev])
        xnorm = float(X.norm(dim=-1).median())
        blk = model.model.layers[b]
        gains = {eps: mlp_response(blk, X, D_all, eps, xnorm) for eps in EPS}
        stance_share = ((D_all @ Q) ** 2).sum(-1)
        results[b] = {
            "names": names,
            "icc": icc.cpu().numpy(),
            "ac1": ac1.cpu().numpy(),
            "stance_share": stance_share.cpu().numpy(),
            **{f"gain_{eps}": gains[eps].cpu().numpy() for eps in EPS},
            "xnorm": xnorm,
            "n_tokens": int(X.shape[0]),
        }
        log.info(
            "block %d: %d directions, %d tokens, %.1f min", b, len(names), X.shape[0], (time.monotonic() - t0) / 60
        )
        for gname in groups:
            sel = [i for i, n in enumerate(names) if n.startswith(gname + ":")]
            log.info(
                "  %-22s gain0.1 %+.3f  icc %.3f  ac1 %.3f  stance %.3f",
                gname,
                float(np.median(results[b]["gain_0.1"][sel])),
                float(np.median(results[b]["icc"][sel])),
                float(np.median(results[b]["ac1"][sel])),
                float(np.median(results[b]["stance_share"][sel])),
            )
    np.savez(
        str(args.out),
        blocks=np.array(BLOCKS),
        dir_pairs=np.array(dir_pairs),
        eval_pairs=np.array(eval_pairs),
        **{
            f"b{b}_{k}": (np.array(v) if not isinstance(v, (int, float)) else np.array(v))
            for b, r in results.items()
            for k, v in r.items()
        },
    )
    log.info("done %.1f min", (time.monotonic() - t0) / 60)


if __name__ == "__main__":
    main()
