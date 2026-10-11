import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TAGS = ["qwen", "llama", "gemma"]
NICE = {"qwen": "Qwen3-8B", "llama": "Llama-3.1-8B", "gemma": "Gemma-2-9b"}
LAD = {"tylenol": "#d62728", "syrup": "#ff7f0e", "ibuprofen": "#9467bd", "steps": "#1f77b4"}
RANDOM_COL = 457


def cell(ax, npz, meta, col, li, title, is_ctl=False):
    key = "ctl" if is_ctl else "cos"
    for lad, color in LAD.items():
        doses = meta["ladders"][lad]["doses"]
        a = npz[f"{key}.{lad}"][:, li, col]
        ax.plot(doses, a - a[0], "o-", color=color, ms=4, lw=1.6, label=lad)
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set_xscale("log")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("dose")
    ax.grid(alpha=0.25)


def main(args):
    res = Path(args.results)
    data = {}
    for tag in TAGS:
        npz = np.load(res / f"dose_{tag}.npz", allow_pickle=True)
        meta = json.loads(str(npz["meta"]))
        rank = json.load(open(res / f"ranking_{tag}.json"))
        data[tag] = (npz, meta, rank)

    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharey="row")
    for r, t in enumerate(TAGS):
        npz, meta, rank = data[t]
        idx5 = meta["headline_layer_idx"]
        top = rank["ranking"][:2]
        for cbox, item in enumerate(top):
            li = item["best_layer_idx"]
            cell(
                axes[r][cbox],
                npz,
                meta,
                item["pair_id"],
                li,
                f"#{item['pair_id']} {item['name']}\nz={item['z']} L{item['best_layer']}",
            )
        cell(axes[r][2], npz, meta, RANDOM_COL, idx5, f"random #{RANDOM_COL} (L{meta['layers'][idx5]})", is_ctl=True)
        axes[r][0].set_ylabel(f"{NICE[t]}\nΔcos, assistant token")
    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Concept dose-response at the assistant token — discovered candidates, 3 models "
        "(danger ladders vs steps control)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(res / "dose_cross.png", dpi=140)
    print("wrote", res / "dose_cross.png")

    P = len(data["qwen"][1]["names"])
    zmap = {t: {r["pair_id"]: r["z"] for r in data[t][2]["ranking"]} for t in TAGS}
    minz = np.array([min(zmap[t].get(c, -1e9) for t in TAGS) for c in range(P)])
    cbest = int(minz.argmax())
    name = data["qwen"][1]["names"][cbest]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for r, t in enumerate(TAGS):
        npz, meta, rank = data[t]
        li = next(
            (it["best_layer_idx"] for it in rank["ranking"] if it["pair_id"] == cbest), meta["headline_layer_idx"]
        )
        z_here = zmap[t].get(cbest, float("nan"))
        cell(axes[r], npz, meta, cbest, li, f"{NICE[t]}  (z={z_here:.1f}, L{meta['layers'][li]})")
        if r == 0:
            axes[r].set_ylabel("Δcos, assistant token")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        f"Most cross-model-consistent danger-dose concept: #{cbest} {name} (min z across models {minz[cbest]:.1f})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(res / "dose_common.png", dpi=140)
    print("wrote", res / "dose_common.png", "| concept", cbest, name)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--results", default="results")
    main(ap.parse_args())
