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
from exp_attn_interp.config import GENVECTORS_ROOT, INSTRUCT_MODEL, MARKERS_PRESENCE, vector_tag_for
from exp_attn_interp.decomp import POOL_MIN, POOL_SKIP
from exp_attn_interp.slowgain import PreMlpCatcher, mlp_response, unit
from exp_attn_interp.vectors import load_diff_vectors
from exp_attn_interp.write import story_ids

log = logging.getLogger("vargain")
BLOCKS = (6, 10, 14, 18, 22, 26, 30)
EPS = (0.1, 0.25)


@torch.inference_mode()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--n-dir-pairs", type=int, default=64)
    parser.add_argument("--n-eval", type=int, default=32)
    parser.add_argument("--n-pca", type=int, default=192)
    parser.add_argument("--n-pcs", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path("work/vargain.npz"))
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
        GENVECTORS_ROOT / "data" / "semantic.parquet",
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
    tok_pcs, tok_eig, pc_between, mean_dir = {}, {}, {}, {}
    means = {b: [] for b in BLOCKS}
    toks = {b: [] for b in BLOCKS}
    for e in run(pca_texts):
        for b in BLOCKS:
            means[b].append(e[b].mean(0))
            toks[b].append(e[b])
    gen = torch.Generator().manual_seed(1)
    for b in BLOCKS:
        M = torch.stack(means[b])
        mean_dir[b] = unit(M.mean(0))
        _, _, Wt = torch.linalg.svd(M - M.mean(0), full_matrices=False)
        pc_between[b] = Wt[:24]
        X = torch.cat(toks[b])
        X = X[torch.randperm(X.shape[0], generator=gen)[:24000]]
        Xc = X - X.mean(0)
        _, Sv, Wt2 = torch.linalg.svd(Xc, full_matrices=False)
        tok_pcs[b] = Wt2[: args.n_pcs]
        tok_eig[b] = Sv[: args.n_pcs] ** 2 / (Xc.shape[0] - 1)
    log.info("PCs done, %.1f min", (time.monotonic() - t0) / 60)
    stance_Q = {}
    for b in BLOCKS:
        Xv = V[b]
        m = Xv.mean(0)
        _, _, Wt = torch.linalg.svd(Xv - m, full_matrices=False)
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
        Q = stance_Q[b]
        cv = V[b][idx]
        groups = {
            "tokpc": tok_pcs[b],
            "concept": cv,
            "topic": topic[b],
            "random": rnd,
            "e_gap": egap,
            "u_gap": ugap,
            "pc_between": pc_between[b],
            "mean_dir": mean_dir[b][None],
            "concept_stance_part": unit((cv @ Q) @ Q.T),
            "concept_specific_part": unit(cv - (cv @ Q) @ Q.T),
        }
        names, dirs, eigidx = [], [], []
        for gname, D in groups.items():
            for j in range(D.shape[0]):
                names.append(f"{gname}:{j}")
                eigidx.append(j if gname == "tokpc" else -1)
            dirs.append(D)
        D_all = torch.cat(dirs)
        projs = [e[b] @ D_all.T for e in ev]
        allp = torch.cat(projs)
        tot_var = allp.var(0)
        story_means = torch.stack([p.mean(0) for p in projs])
        icc = story_means.var(0) / tot_var.clamp_min(1e-12)
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
            "eigidx": np.array(eigidx),
            "eig": tok_eig[b].cpu().numpy(),
            "var": tot_var.cpu().numpy(),
            "icc": icc.cpu().numpy(),
            "ac1": ac1.cpu().numpy(),
            "stance_share": stance_share.cpu().numpy(),
            **{f"gain_{eps}": gains[eps].cpu().numpy() for eps in EPS},
            "xnorm": xnorm,
            "n_tokens": int(X.shape[0]),
        }
        log.info("block %d: %d directions, %.1f min", b, len(names), (time.monotonic() - t0) / 60)
        gn = results[b]["gain_0.1"]
        va = results[b]["var"]
        for gname in groups:
            sel = [i for i, nm in enumerate(names) if nm.startswith(gname + ":")]
            log.info(
                "  %-22s gain0.1 %+.3f  var %.3g  icc %.3f",
                gname,
                float(np.median(gn[sel])),
                float(np.median(va[sel])),
                float(np.median(results[b]["icc"][sel])),
            )
    np.savez(
        str(args.out),
        blocks=np.array(BLOCKS),
        dir_pairs=np.array(dir_pairs),
        **{f"b{b}_{k}": np.array(v) for b, r in results.items() for k, v in r.items()},
    )
    log.info("done %.1f min", (time.monotonic() - t0) / 60)


if __name__ == "__main__":
    main()
