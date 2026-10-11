from __future__ import annotations

import collections
import glob
import json
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from exp_attn_interp.config import vector_tag_for
from exp_attn_interp.vectors import load_diff_vectors

log = logging.getLogger("saedecode")
SAE_REPO = "models--Qwen--SAE-Res-Qwen3-8B-Base-W64K-L0_50"


def sae_path(layer: int) -> str:
    local = Path(f"work/sae/weights/layer{layer}.sae.pt")
    if local.exists():
        return str(local)
    snaps = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/{SAE_REPO}/snapshots/*/layer{layer}.sae.pt"))
    if not snaps:
        raise FileNotFoundError(f"SAE layer {layer} not downloaded")
    return snaps[0]


@torch.inference_mode()
def main() -> None:
    p = ArgumentParser()
    p.add_argument("--model", default="base", choices=["base", "instruct"])
    p.add_argument("--rows", default="4,12,19,22,24,28,35")
    p.add_argument(
        "--every", type=int, default=8, help="instruct: every n-th pair; base: all pairs with a decomposition"
    )
    p.add_argument("--topn", type=int, default=15)
    p.add_argument("--pool", type=int, default=25)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--wanted",
        type=Path,
        default=None,
        help="JSON {row: [feature ids]}: also store every block's projection on these features, per concept",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "base":
        workdir, tag = "work-Qwen3-8B-Base", vector_tag_for("Qwen/Qwen3-8B-Base", "raw")
        pairs = sorted(int(Path(f).stem.split("-")[1]) for f in glob.glob(f"{workdir}/vdecomp-en/pair-*.safetensors"))
        vd = "vdecomp-en"
    else:
        workdir, tag = "work", vector_tag_for("Qwen/Qwen3-8B")
        pairs = [k for k in range(0, 1036, args.every) if Path(f"{workdir}/vdecomp/pair-{k:04d}.safetensors").exists()]
        vd = "vdecomp"
    out_path = args.out or Path(f"work/sae/decode-{args.model}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        Vall = load_diff_vectors(device="cpu", tag=tag)
    except Exception as exc:  # noqa: BLE001
        log.warning("paper vectors for %s unavailable (%s): using the decomposition's own whole vector", tag, exc)
        Vall = None
    parts: dict[int, dict[str, torch.Tensor]] = {}
    for k in pairs:
        if args.model == "base":
            f = Path(f"{workdir}/{vd}/pair-{k:04d}.safetensors")
            t = load_file(str(f))
            parts[k] = {
                "h": (t["pole_head"][0] - t["pole_head"][1]).float().sum(1),
                "m": (t["pole_mlp"][0] - t["pole_mlp"][1]).float(),
                "e": (t["pole_embed"][0] - t["pole_embed"][1]).float(),
                "r": (t["pole_resid"][0] - t["pole_resid"][1]).float(),
            }
        else:
            acc = None
            for d in ("vdecomp", "vdecomp2"):
                f = Path(f"{workdir}/{d}/pair-{k:04d}.safetensors")
                if not f.exists():
                    continue
                t = load_file(str(f))
                cur = {
                    "h": (t["pole_head"][0] - t["pole_head"][1]).float().sum(1),
                    "m": (t["pole_mlp"][0] - t["pole_mlp"][1]).float(),
                    "e": (t["pole_embed"][0] - t["pole_embed"][1]).float(),
                    "r": (t["pole_resid"][0] - t["pole_resid"][1]).float(),
                }
                acc = cur if acc is None else {n: acc[n] + cur[n] for n in cur}
            if acc is not None:
                parts[k] = acc
    pairs = [k for k in pairs if k in parts]
    log.info("%s: %d concepts", args.model, len(parts))
    rows = [int(x) for x in args.rows.split(",")]
    result: dict[str, Any] = {"model": args.model, "pairs": pairs, "rows": {}}
    for r in rows:
        sd = torch.load(sae_path(r - 1), map_location="cpu")
        W = sd["W_dec"].float()
        D = (W / W.norm(dim=0, keepdim=True).clamp_min(1e-8)).T.contiguous()
        D = D.to(dev, torch.float16)
        objects: dict[str, dict[int, torch.Tensor]] = collections.defaultdict(dict)
        for k, P in parts.items():
            v = Vall[r, k] if Vall is not None else P["r"][r]
            objects["v"][k] = v
            objects["A"][k] = P["h"][:r].sum(0)
            objects["M"][k] = P["m"][:r].sum(0)
            objects["E"][k] = P["e"]
            for b in range(r):
                objects[f"attn{b}"][k] = P["h"][b]
                objects[f"mlp{b}"][k] = P["m"][b]
        rowres: dict[str, dict] = {}
        attribution: dict[int, dict[str, float]] = {}
        for name, vecs in objects.items():
            freq: collections.Counter = collections.Counter()
            score: dict[int, float] = collections.defaultdict(float)
            X = torch.stack([vecs[k] for k in pairs]).to(dev, torch.float16)
            Xn = X / X.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            S = (Xn @ D.T).float()
            top = S.abs().topk(args.topn, dim=-1)
            for i in range(S.shape[0]):
                for j, val in zip(top.indices[i].tolist(), top.values[i].tolist()):
                    freq[j] += 1
                    score[j] += float(S[i, j])
            pooled = [(j, freq[j], score[j] / freq[j]) for j, _ in freq.most_common(args.pool)]
            rowres[name] = {
                "top": [{"f": j, "freq": n, "mean_score": round(s, 4)} for j, n, s in pooled],
                "mean_abs_top1": round(float(top.values[:, 0].mean()), 4),
            }
            if name == "v":
                vraw = torch.stack([parts[k]["r"][r] for k in pairs]).to(dev, torch.float16)
                vnorm = vraw.norm(dim=-1).clamp_min(1e-6)
                feats = [j for j, _, _ in pooled]
                Dsub = D[feats].float()
                for comp in ("E",) + tuple(f"attn{b}" for b in range(r)) + tuple(f"mlp{b}" for b in range(r)):
                    Xc = torch.stack([objects[comp][k] for k in pairs]).to(dev).float()
                    contrib = (Xc @ Dsub.T) / vnorm[:, None].float()
                    mean_c = contrib.mean(0).tolist()
                    for jj, j in enumerate(feats):
                        attribution.setdefault(j, {})[comp] = round(mean_c[jj], 4)
        result["rows"][str(r)] = {"objects": rowres, "v_attribution": {str(j): a for j, a in attribution.items()}}
        if args.wanted and args.wanted.exists():
            want = [int(j) for j in json.loads(args.wanted.read_text()).get(str(r), [])]
            if want:
                Dw = D[want].float()
                comps = ["E"] + [f"attn{b}" for b in range(r)] + [f"mlp{b}" for b in range(r)]
                vraw = torch.stack([parts[k]["r"][r] for k in pairs]).to(dev).float()
                vnorm = vraw.norm(dim=-1).clamp_min(1e-6)
                per_comp = {}
                for comp in comps:
                    Xc = torch.stack([objects[comp][k] for k in pairs]).to(dev).float()
                    per_comp[comp] = ((Xc @ Dw.T) / vnorm[:, None]).cpu()
                vproj = ((vraw @ Dw.T) / vnorm[:, None]).cpu()
                result["rows"][str(r)]["wanted"] = {
                    "features": want,
                    "pairs": pairs,
                    "v": vproj.tolist(),
                    "by_comp": {c: t.tolist() for c, t in per_comp.items()},
                }
        log.info("row %d: v top features %s", r, [(d["f"], d["freq"]) for d in rowres["v"]["top"][:8]])
        del D
        torch.cuda.empty_cache()
    out_path.write_text(json.dumps(result))
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
