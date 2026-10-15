from __future__ import annotations

import json
import logging
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors import safe_open

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from exp_attn_interp.attnsrc import is_delim  # noqa: E402
from exp_attn_interp.common import ValueCatcher, load_model, story_span  # noqa: E402
from exp_attn_interp.config import GENVECTORS_ROOT, INSTRUCT_MODEL, vector_tag_for  # noqa: E402
from exp_attn_interp.decomp import POOL_MIN, POOL_SKIP  # noqa: E402
from exp_attn_interp.vectors import load_diff_vectors  # noqa: E402
from exp_attn_interp.write import story_ids  # noqa: E402

log = logging.getLogger("fieldtest")
CATS = ("template", "marker", "delim", "word", "self")


@torch.inference_mode()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--every", type=int, default=16)
    parser.add_argument("--per-pole", type=int, default=2)
    parser.add_argument("--rows", default="12,22,30")
    parser.add_argument("--blocks", default="22,24,26,28,30")
    parser.add_argument("--topn", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("work/fieldtest.npz"))
    parser.add_argument("--figs", type=Path, default=Path("figures/field"))
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.figs.mkdir(parents=True, exist_ok=True)
    rows = [int(x) for x in args.rows.split(",")]
    blocks = [int(x) for x in args.blocks.split(",")]
    model, tokenizer = load_model(offline=args.offline, model_name=args.model)
    catcher = ValueCatcher(model.model)
    vectors = load_diff_vectors(device=str(model.device), tag=vector_tag_for(args.model))
    B = len(model.model.layers)
    last = B - 1
    heads, kv, hd = catcher.heads, catcher.kv_heads, catcher.head_dim
    group = heads // kv
    dev = model.device
    mk = json.loads(Path("data/markers_presence.json").read_text())
    tbl = pq.read_table(
        GENVECTORS_ROOT / "data" / "semantic.parquet",
        columns=["pair_id", "concept_text", "antagonist_text"],
        filters=[("language_id", "==", 0)],
    )
    by_pair: dict[int, list[tuple[str, str]]] = {}
    for p, c, a in zip(tbl["pair_id"].to_pylist(), tbl["concept_text"].to_pylist(), tbl["antagonist_text"].to_pylist()):
        by_pair.setdefault(int(p), []).append((c, a))
    pairs = sorted(by_pair)[:: args.every]
    ac: dict[int, list[list[float]]] = {L: [] for L in rows}
    catmean: dict[int, list[list[float]]] = {L: [] for L in rows}
    srcshare: list[np.ndarray] = []
    tokstd: dict[int, list[float]] = {L: [] for L in rows}
    storygap: dict[int, list[float]] = {L: [] for L in rows}
    for n, k in enumerate(pairs):
        ids_c, ids_a = set(mk["concept"].get(str(k), [])), set(mk["antagonist"].get(str(k), []))
        for c, a in by_pair[k][: args.per_pole]:
            for side, text in ((0, c), (1, a)):
                ids = story_ids(tokenizer, text)
                T = len(ids)
                s0, s1 = story_span(tokenizer, ids, text)
                if (s1 - s0 - POOL_SKIP) >= POOL_MIN:
                    s0 += POOL_SKIP
                out = model(
                    input_ids=torch.tensor([ids], device=dev),
                    output_attentions=True,
                    output_hidden_states=True,
                    use_cache=False,
                )
                cat = np.full(T, 3)
                for t, tid in enumerate(ids):
                    if t < s0:
                        cat[t] = 0
                    elif tid in ids_c or tid in ids_a:
                        cat[t] = 1
                    elif is_delim(tokenizer.decode([tid])):
                        cat[t] = 2
                sign = 1.0 if side == 0 else -1.0
                for L in rows:
                    u = vectors[L, k]
                    pr = (out.hidden_states[L][0, s0:s1].to(torch.float32) @ u).cpu().numpy() * sign
                    pc = pr - pr.mean()
                    v0 = float((pc * pc).mean()) + 1e-9
                    ac[L].append([float((pc[:-lag] * pc[lag:]).mean()) / v0 for lag in range(1, 17)])
                    cs = cat[s0:s1]
                    catmean[L].append([float(pc[cs == j].mean()) if (cs == j).any() else np.nan for j in (1, 2, 3)])
                    tokstd[L].append(float(pr.std()))
                    storygap[L].append(float(pr.mean()))
                b = 22
                u = vectors[b, k]
                attn = out.attentions[b][0].to(torch.float32)[:, s0:s1, :]
                raw = catcher.values[b]
                assert raw is not None
                Vv = raw.view(T, kv, hd).to(torch.float32)[:, torch.arange(heads, device=dev) // group]
                W_O = model.model.layers[b].self_attn.o_proj.weight.to(torch.float32).view(-1, heads, hd)
                w = torch.einsum("ohd,o->hd", W_O, u)
                P = torch.einsum("thd,hd->ht", Vv, w)
                per_src = (attn * P[:, None, :]).sum(0)
                onehot = torch.nn.functional.one_hot(torch.tensor(cat, device=dev), 4).to(torch.float32)
                bycat = per_src @ onehot
                selfc = torch.diagonal(per_src[:, s0:s1])
                bycat = (
                    bycat
                    - torch.nn.functional.one_hot(torch.tensor(cat[s0:s1], device=dev), 4).to(torch.float32)
                    * selfc[:, None]
                )
                tot = per_src.sum(1)
                share = torch.cat([bycat, selfc[:, None]], 1) / tot[:, None].abs().clamp_min(1e-6)
                srcshare.append(share.mean(0).cpu().numpy())
                del out, attn
        if n % 16 == 0:
            log.info("A: %d/%d pair %d", n, len(pairs), k)
    resA = {f"ac_{L}": np.array(ac[L]) for L in rows}
    resA.update({f"cat_{L}": np.array(catmean[L]) for L in rows})
    resA.update({f"tokstd_{L}": np.array(tokstd[L]) for L in rows})
    resA.update({f"gap_{L}": np.array(storygap[L]) for L in rows})
    resA["srcshare"] = np.array(srcshare)
    for L in rows:
        log.info(
            "A row %d: autocorrelation of the per-token projection at lags 1/2/4/8/16: %s; token std / |story mean| %.2f; category means (marker, delim, word) relative to story: %s",
            L,
            [round(float(np.median(resA[f"ac_{L}"][:, i])), 2) for i in (0, 1, 3, 7, 15)],
            float(np.median(np.array(tokstd[L]) / np.abs(storygap[L]))),
            [round(float(np.nanmedian(resA[f"cat_{L}"][:, j])), 2) for j in range(3)],
        )
    log.info(
        "A block 22: share of a token's attention write along v(22) from template/marker/delim/word/self sources: %s",
        [round(float(np.median(resA["srcshare"][:, j])), 3) for j in range(5)],
    )
    files = sorted(Path("work/mlpgain").glob("pair-*.safetensors"))
    gpairs = [int(f.stem.split("-")[1]) for f in files]
    neur = np.stack([safe_open(str(f), framework="np").get_tensor("neuron")[0] for f in files]).astype(np.float32)
    Kg = torch.tensor(gpairs, device=dev)
    V22 = vectors[22, Kg]
    V35 = vectors[last, Kg]
    gen = torch.Generator(device="cpu").manual_seed(0)
    resB = {}
    for b in blocks:
        blk = model.model.layers[b]
        Wg = blk.mlp.gate_proj.weight.to(torch.float32)
        Wd = blk.mlp.down_proj.weight.to(torch.float32)
        c = torch.tensor(neur[:, b], device=dev)
        top = torch.topk(c, args.topn, dim=1).indices
        amp = torch.unique(top.flatten())
        nA = int(amp.numel())
        rnd = torch.randperm(Wg.shape[0], generator=gen)[:nA].to(dev)

        def energy(vec, idx):
            Q, _ = torch.linalg.qr(Wg[idx].T)
            Qv, _ = torch.linalg.qr(Wd[:, idx])
            return ((vec @ Q) ** 2).sum(-1), ((vec @ Qv) ** 2).sum(-1)

        ek22, ev22 = energy(V22, amp)
        ek35, ev35 = energy(V35, amp)
        rk22, rv22 = energy(V22, rnd)
        rk35, rv35 = energy(V35, rnd)
        loo_k, loo_v = [], []
        for i in range(0, len(gpairs), max(1, len(gpairs) // 16)):
            others = torch.unique(torch.cat([top[:i].flatten(), top[i + 1 :].flatten()]))
            a, v_ = energy(V22[i : i + 1], others)
            loo_k.append(float(a))
            loo_v.append(float(v_))
        resB[b] = dict(
            nA=nA,
            k22=float(ek22.median()),
            v22=float(ev22.median()),
            k35=float(ek35.median()),
            v35=float(ev35.median()),
            rk22=float(rk22.median()),
            rv22=float(rv22.median()),
            rk35=float(rk35.median()),
            rv35=float(rv35.median()),
            loo_k22=float(np.median(loo_k)),
            loo_v22=float(np.median(loo_v)),
        )
        log.info(
            "B block %d: %d amplifier neurons (top-%d of %d concepts). energy of v(22) in their KEY span %.3f (random %.3f; other concepts' amplifiers %.3f); in their VALUE span %.3f (random %.3f; leave-one-out %.3f). v(35): key %.3f (rnd %.3f) value %.3f (rnd %.3f)",
            b,
            nA,
            args.topn,
            len(gpairs),
            resB[b]["k22"],
            resB[b]["rk22"],
            resB[b]["loo_k22"],
            resB[b]["v22"],
            resB[b]["rv22"],
            resB[b]["loo_v22"],
            resB[b]["k35"],
            resB[b]["rk35"],
            resB[b]["v35"],
            resB[b]["rv35"],
        )
    allres: dict = dict(resA)
    allres.update({f"B_{b}_{k}": v for b, d in resB.items() for k, v in d.items()})
    np.savez(str(args.out), **allres)  # type: ignore[arg-type]
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    ax = axes[0]
    for L, col in zip(rows, ("#9ca3af", "#111827", "#dc2626")):
        M = resA[f"ac_{L}"]
        ax.plot(range(1, 17), np.median(M, 0), marker="o", color=col, lw=2, label=f"row {L}")
        ax.fill_between(range(1, 17), np.percentile(M, 25, 0), np.percentile(M, 75, 0), color=col, alpha=0.15)
    ax.set_xlabel("lag (tokens)")
    ax.set_title("autocorrelation of the per-token projection on v along a story")
    ax.legend(fontsize=8)
    ax = axes[1]
    x = np.arange(5)
    ax.bar(
        x,
        [np.median(resA["srcshare"][:, j]) for j in range(5)],
        color=("#9ca3af", "#dc2626", "#f59e0b", "#7c3aed", "#111827"),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(CATS)
    ax.set_title("where a token's block-22 attention write along v(22) comes from (share)")
    ax = axes[2]
    xb = np.arange(len(blocks))
    ax.bar(
        xb - 0.3, [resB[b]["k22"] for b in blocks], width=0.2, color="#dc2626", label="v(22) in amplifiers' key span"
    )
    ax.bar(
        xb - 0.1,
        [resB[b]["rk22"] for b in blocks],
        width=0.2,
        color="#fca5a5",
        label="v(22) in random neurons' key span",
    )
    ax.bar(
        xb + 0.1, [resB[b]["v35"] for b in blocks], width=0.2, color="#2563eb", label="v(35) in amplifiers' value span"
    )
    ax.bar(
        xb + 0.3,
        [resB[b]["rv35"] for b in blocks],
        width=0.2,
        color="#93c5fd",
        label="v(35) in random neurons' value span",
    )
    ax.set_xticks(xb)
    ax.set_xticklabels([str(b) for b in blocks])
    ax.set_xlabel("block")
    ax.set_title("energy of concept vectors in the spans of the amplifier neurons' keys and values")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.figs / "F1-field.png", dpi=170)
    log.info("wrote %s", args.figs)


if __name__ == "__main__":
    main()
