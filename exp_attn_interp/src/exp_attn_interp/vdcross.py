from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
import torch
from safetensors.torch import load_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from exp_attn_interp.bstar import crossing_cum  # noqa: E402

log = logging.getLogger("vdcross")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--vdecomp", type=Path, default=Path("work/vdecomp"))
    parser.add_argument("--tag", default="qwen")
    parser.add_argument("--band", default="19,22", help="judged steering band to shade")
    parser.add_argument("--out", type=Path, default=Path("work/vdcross.json"))
    parser.add_argument("--figs", type=Path, default=Path("figures/formation"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.figs.mkdir(parents=True, exist_ok=True)
    dev = args.device
    files = sorted(args.vdecomp.glob("pair-*.safetensors"))
    ML, MR, AL, AR, NV, pairs = [], [], [], [], [], []
    for i, f in enumerate(files):
        t = load_file(str(f))
        A = (t["pole_head"][0] - t["pole_head"][1]).to(dev, torch.float32).sum(1)
        M = (t["pole_mlp"][0] - t["pole_mlp"][1]).to(dev, torch.float32)
        De = (t["pole_embed"][0] - t["pole_embed"][1]).to(dev, torch.float32)
        B = M.shape[0]
        V = torch.cat([De[None], De[None] + torch.cumsum(A + M, 0)], 0)
        vl = V[B - 1] / V[B - 1].norm(dim=-1, keepdim=True).clamp_min(1e-8)
        vr = V[:B] / V[:B].norm(dim=-1, keepdim=True).clamp_min(1e-8)
        ML.append((M @ vl).cpu())
        AL.append((A @ vl).cpu())
        MR.append((M * vr).sum(-1).cpu())
        AR.append((A * vr).sum(-1).cpu())
        NV.append(V[B - 1].norm().cpu())
        pairs.append(int(f.stem.split("-")[1]))
        if i % 200 == 0:
            log.info("%d/%d", i, len(files))
    ml, mr, al, ar = (torch.stack(x).numpy() for x in (ML, MR, AL, AR))
    nv = torch.stack(NV).numpy()[:, None]
    K, B = ml.shape
    pop_l, pop_r = np.median(ml / nv, 0), np.median(mr / nv, 0)
    bl = np.array([crossing_cum(ml[i]) for i in range(K)])
    br = np.array([crossing_cum(mr[i]) for i in range(K)])
    res = {
        "tag": args.tag,
        "n_blocks": B,
        "K": K,
        "bstar_final_population": int(crossing_cum(pop_l)),
        "bstar_running_population": int(crossing_cum(pop_r)),
        "bstar_final_per_concept": {
            "median": float(np.median(bl)),
            "p10": float(np.percentile(bl, 10)),
            "p90": float(np.percentile(bl, 90)),
            "hist": np.bincount(bl, minlength=B).tolist(),
        },
        "bstar_running_per_concept": {
            "median": float(np.median(br)),
            "p10": float(np.percentile(br, 10)),
            "p90": float(np.percentile(br, 90)),
            "hist": np.bincount(br, minlength=B).tolist(),
        },
        "mlp_along_final_median": pop_l.tolist(),
        "mlp_along_running_median": pop_r.tolist(),
        "attn_along_final_median": np.median(al / nv, 0).tolist(),
        "attn_along_running_median": np.median(ar / nv, 0).tolist(),
    }
    thirds = ((0, B // 3), (B // 3, 2 * B // 3), (2 * B // 3, B))
    res["shares_of_v_last"] = {
        f"attn {lo}-{hi - 1}": float(np.median(al[:, lo:hi].sum(1) / nv[:, 0])) for lo, hi in thirds
    }
    res["shares_of_v_last"].update(
        {f"mlp {lo}-{hi - 1}": float(np.median(ml[:, lo:hi].sum(1) / nv[:, 0])) for lo, hi in thirds}
    )
    args.out.write_text(json.dumps(res, indent=1))
    log.info(
        "%s: b* final %d (per concept median %.0f, p10 %.0f, p90 %.0f); b* running %d (median %.0f, p10 %.0f, p90 %.0f)",
        args.tag,
        res["bstar_final_population"],
        np.median(bl),
        np.percentile(bl, 10),
        np.percentile(bl, 90),
        res["bstar_running_population"],
        np.median(br),
        np.percentile(br, 10),
        np.percentile(br, 90),
    )
    log.info("MLP along final, median share by block: %s", [round(float(x), 3) for x in pop_l])
    log.info("MLP along running, median share by block: %s", [round(float(x), 3) for x in pop_r])
    log.info("shares of |v(last)|: %s", {k: round(v, 3) for k, v in res["shares_of_v_last"].items()})
    b0, b1 = (int(x) for x in args.band.split(","))
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    x = np.arange(B)
    ax = axes[0]
    ax.plot(
        x,
        pop_l / np.abs(pop_l).max(),
        color="#111827",
        lw=2,
        label=f"MLP along v(last): b* = {res['bstar_final_population']}",
    )
    ax.plot(
        x,
        pop_r / np.abs(pop_r).max(),
        color="#dc2626",
        lw=2,
        label=f"MLP along v(b): b* = {res['bstar_running_population']}",
    )
    ax.plot(
        x,
        np.median(al / nv, 0) / np.abs(np.median(al / nv, 0)).max(),
        color="#2563eb",
        lw=1.5,
        label="attention along v(last)",
    )
    ax.axhline(0, color="k", lw=0.5)
    ax.axvspan(b0, b1, color="#f59e0b", alpha=0.12)
    ax.set_title(f"{args.tag}: per-block writes along the concept direction (median over {K} concepts, normalized)")
    ax.set_xlabel("block b")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.hist(bl, bins=np.arange(-0.5, B + 0.5), color="#111827", alpha=0.7, label="along v(last)")
    ax.hist(br, bins=np.arange(-0.5, B + 0.5), color="#dc2626", alpha=0.5, label="along v(b)")
    ax.axvspan(b0, b1, color="#f59e0b", alpha=0.12)
    ax.set_title("per-concept b* (cumulative rule)")
    ax.set_xlabel("b*_k")
    ax.legend(fontsize=8)
    ax = axes[2]
    order = np.argsort(bl)
    im = ax.imshow((ml / nv)[order], aspect="auto", cmap="RdBu_r", vmin=-0.02, vmax=0.02, interpolation="nearest")
    ax.set_title("per-concept MLP write along v(last), share of |v(last)| (sorted by b*_k)")
    ax.set_xlabel("block b")
    ax.set_ylabel("concept")
    plt.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(args.figs / f"A12-vdcross-{args.tag}.png", dpi=170)
    log.info("wrote %s", args.figs)


if __name__ == "__main__":
    main()
