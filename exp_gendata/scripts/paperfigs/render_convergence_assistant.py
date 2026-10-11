import io
import json
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from safetensors import safe_open
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[3]
REVIVE = ROOT / ".bak" / "CONVERGENCE_REVIVE"
ONTOLOGY = ROOT / ".bak" / "ood-study" / "semantic.ontology.json"
OUT = Path(__file__).resolve().parent

MODELS = [
    ("Qwen3-8B", "Qwen_Qwen3-8B--assistant--semantic", "Qwen3-8B", "#2166ac"),
    (
        "Meta-Llama-3.1-8B-Instruct",
        "unsloth_Meta-Llama-3.1-8B-Instruct--assistant--semantic",
        "Llama-3.1-8B",
        "#d87800",
    ),
    ("gemma-2-9b-it", "google_gemma-2-9b-it--assistant--semantic", "gemma-2-9b", "#2e8b57"),
]
CLASSES = [
    ("Uniformity vs Diversity", "uniformity vs diversity (fast)", "#1f77b4"),
    ("Leadership Archetypes", "leadership archetypes (fast)", "#2ca02c"),
    ("Honor, Shame & Facework", "honor, shame & facework (slow)", "#c0392b"),
    ("Classical Moral Dilemmas", "classical moral dilemmas (slow)", "#e08214"),
]
TOL = 0.005
LO, HI = 5.0, 95.0

plt.rcParams.update(
    {
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "pdf.fonttype": 42,
    }
)


def style_axis(ax):
    ax.grid(True, linestyle=":", linewidth=0.45, alpha=0.45)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def pow2_axis(ax, grid):
    ax.set_xscale("log", base=2)
    ax.set_xticks(grid)
    ax.set_xticklabels([rf"$2^{{{int(np.log2(n))}}}$" for n in grid])
    ax.set_xlabel("stories per pole, $N$")


def bucket_c6(stem, layer):
    with safe_open(str(BUCKETS / f"{stem}.safetensors"), framework="np") as h:
        return h.get_tensor("q_q")[layer].mean(0).astype(np.float64)


def logdev(x):
    return np.log10(np.clip(1.0 - x, 1e-6, None))


def band(values):
    return np.median(values, 0), np.percentile(values, LO, 0), np.percentile(values, HI, 0)


def main():
    global CURVES, SUMMARY, BUCKETS, ONTOLOGY, OUT, LO, HI
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--curves",
        type=Path,
        default=REVIVE / "results" / "convcurve" / "raw",
        help="per-model folders of pair-*.npz, or <tag>.tar.gz archives",
    )
    parser.add_argument("--summary", type=Path, default=REVIVE / "results" / "convcurve")
    parser.add_argument("--buckets", type=Path, default=REVIVE / "inputs" / "convergence")
    parser.add_argument("--ontology", type=Path, default=ONTOLOGY)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--numbers", type=Path, default=REVIVE / "results" / "convcurve" / "paper-numbers.json")
    parser.add_argument("--lo", type=float, default=LO)
    parser.add_argument("--hi", type=float, default=HI)
    args = parser.parse_args()
    CURVES, SUMMARY, BUCKETS, ONTOLOGY, OUT, LO, HI = (
        args.curves,
        args.summary,
        args.buckets,
        args.ontology,
        args.out,
        args.lo,
        args.hi,
    )
    classes = {int(p["pair_id"]): p["class_name"] for p in json.loads(ONTOLOGY.read_text())["pairs"]}
    data, numbers = {}, {}
    for tag, stem, label, color in MODELS:
        pids, auroc, conv, grid = [], [], [], None

        def add(name, blob):
            nonlocal grid
            z = np.load(blob)
            pids.append(int(Path(name).stem.split("-")[1]))
            auroc.append(z["auroc"])
            conv.append(z["conv"])
            grid = z["grid"]

        folder = CURVES / tag
        if folder.is_dir():
            for f in sorted(folder.glob("pair-*.npz")):
                add(f.name, f)
        else:
            with tarfile.open(CURVES / f"{tag}.tar.gz") as tf:
                for member in tf.getmembers():
                    if member.name.endswith(".npz"):
                        add(member.name, io.BytesIO(tf.extractfile(member).read()))
        order = np.argsort(pids)
        pids, auroc, conv, grid = (
            np.array(pids)[order],
            np.stack(auroc)[order],
            np.stack(conv)[order],
            [int(n) for n in grid],
        )
        layer = json.loads((SUMMARY / f"convcurve-{tag}.json").read_text())["auroc"]["best_layer"]
        a, c = auroc[:, layer, :], conv[:, layer, :]
        c6 = bucket_c6(stem, layer)
        g8 = grid.index(8)
        plateau = a[:, -1]
        nstar = np.array([grid[next(i for i in range(len(grid)) if row[i] >= row[-1] - TOL)] for row in a])
        fall = np.array([np.max(np.maximum.accumulate(row) - row) for row in a])
        med_a, lo_a, hi_a = band(a)
        med_c, lo_c, hi_c = band(c)
        n_plateau = grid[next(i for i in range(len(grid)) if med_a[i] >= med_a[-1] - TOL)]
        corr = {}
        for n in (8, 16, 32):
            x, y = logdev(c6), logdev(a[:, grid.index(n)])
            corr[n] = {"pearson": pearsonr(x, y)[0], "spearman": spearmanr(x, y)[0]}
        data[label] = dict(
            a=a,
            c=c,
            c6=c6,
            grid=grid,
            color=color,
            layer=layer,
            pids=pids,
            nstar=nstar,
            plateau=plateau,
            fall=fall,
            hard=logdev(a[:, g8]),
        )
        numbers[label] = {
            "best_layer": int(layer),
            "auroc_median": dict(zip(grid, med_a.round(4).tolist())),
            "auroc_lo": dict(zip(grid, lo_a.round(4).tolist())),
            "auroc_hi": dict(zip(grid, hi_a.round(4).tolist())),
            "cos_median": dict(zip(grid, med_c.round(4).tolist())),
            "cos_lo": dict(zip(grid, lo_c.round(4).tolist())),
            "cos_hi": dict(zip(grid, hi_c.round(4).tolist())),
            "median_curve_plateau_N": n_plateau,
            "plateau_auroc_median": round(float(np.median(plateau)), 4),
            "nstar_median": int(np.median(nstar)),
            "nstar_p10": int(np.percentile(nstar, 10)),
            "nstar_p90": int(np.percentile(nstar, 90)),
            "hardness_vs_c6": {n: {k: round(float(v), 3) for k, v in d.items()} for n, d in corr.items()},
            "nstar_vs_c6_spearman": round(float(spearmanr(nstar, c6)[0]), 3),
            "post_peak_fall": {
                "median": round(float(np.median(fall)), 4),
                "p95": round(float(np.percentile(fall, 95)), 4),
                "max": round(float(fall.max()), 4),
            },
        }
        lo, hi = np.percentile(c6, [100 / 3, 200 / 3])
        cls = np.array([classes[int(p)] for p in pids])
        numbers[label]["classes"] = {}
        for prefix, _, _ in CLASSES:
            m = np.array([s.startswith(prefix) for s in cls])
            numbers[label]["classes"][prefix] = {
                "n": int(m.sum()),
                "fast": int((c6[m] >= hi).sum()),
                "slow": int((c6[m] <= lo).sum()),
            }

    labels = [m[2] for m in MODELS]
    cross = {}
    for key in ("c6", "hard", "plateau", "nstar", "fall"):
        vals = {l: data[l][key] if key != "c6" else logdev(data[l]["c6"]) for l in labels}
        cross[key] = {
            f"{x}--{y}": round(float(spearmanr(vals[x], vals[y])[0]), 3)
            for i, x in enumerate(labels)
            for y in labels[i + 1 :]
        }
    numbers["cross_model_spearman"] = cross

    numbers["rho_c6_vs_auroc_by_N"], numbers["frac_auroc_1_by_N"] = {}, {}
    for l in labels:
        d = data[l]
        numbers["rho_c6_vs_auroc_by_N"][l] = {
            n: round(float(spearmanr(d["c6"], d["a"][:, i])[0]), 3) for i, n in enumerate(d["grid"])
        }
        numbers["frac_auroc_1_by_N"][l] = {
            n: round(float((d["a"][:, i] >= 0.9999).mean()), 3) for i, n in enumerate(d["grid"])
        }

    fmts = ("user", "assistant", "paired_assistant")
    numbers["cross_format_spearman_c6"] = {}
    for tag, stem, label, _ in MODELS:
        layer = data[label]["layer"]
        c6f = {f: bucket_c6(stem.replace("--assistant--", f"--{f}--"), layer) for f in fmts}
        numbers["cross_format_spearman_c6"][label] = {
            f"{x}--{y}": round(float(spearmanr(c6f[x], c6f[y])[0]), 3)
            for i, x in enumerate(fmts)
            for y in fmts[i + 1 :]
        }

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.2), sharey=True)
    for ax, (tag, stem, label, _) in zip(axes, MODELS):
        with safe_open(str(BUCKETS / f"{stem}.safetensors"), framework="np") as h:
            qq, hh = h.get_tensor("q_q").mean(axis=1), h.get_tensor("h_h").mean(axis=1)
        layers = np.arange(qq.shape[0])
        for vals, color, name in ((qq, "#d87800", "250"), (hh, "#2166ac", "500")):
            med, lo, hi = band(logdev(vals).T)
            ax.fill_between(layers, lo, hi, color=color, alpha=0.16, linewidth=0)
            ax.plot(layers, med, color=color, lw=1.25)
        ax.axvline(data[label]["layer"], color="grey", lw=0.6, ls=":")
        ax.set_title(f"({chr(65 + [m[2] for m in MODELS].index(label))}) {label}", loc="left")
        ax.set_xlabel("layer")
        style_axis(ax)
    axes[0].set_ylabel(r"$\log_{10}(1-\cos)$")
    handles = [
        Line2D([], [], color="#d87800", lw=1.4, label="disjoint 250-story vectors"),
        Line2D([], [], color="#2166ac", lw=1.4, label="disjoint 500-story vectors"),
        Line2D([], [], color="black", lw=5, alpha=0.12, label=f"{LO:g}th--{HI:g}th percentiles"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(OUT / "convergence-details.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    for label in labels:
        with safe_open(str(BUCKETS / f"{[m for m in MODELS if m[2] == label][0][1]}.safetensors"), framework="np") as h:
            L = data[label]["layer"]
            numbers[label]["bucket_cos_median"] = {
                "250": round(float(np.median(h.get_tensor("q_q")[L].mean(0))), 4),
                "500": round(float(np.median(h.get_tensor("h_h")[L].mean(0))), 4),
            }

    def curves(ax):
        for l in labels:
            d = data[l]
            g = d["grid"]
            med, lo, hi = band(d["a"])
            ax.fill_between(g, lo, hi, color=d["color"], alpha=0.10, linewidth=0)
            ax.plot(g, med, color=d["color"], lw=1.3, marker="o", ms=2.2)
            med, lo, hi = band(d["c"])
            ax.fill_between(g, lo, hi, color=d["color"], alpha=0.10, linewidth=0, hatch="////", edgecolor=d["color"])
            ax.plot(g, med, color=d["color"], lw=1.3, ls="--", marker="s", ms=2.0)
        pow2_axis(ax, data[labels[0]]["grid"])
        ax.set_ylabel("median over concepts")
        ax.set_ylim(-0.12, 1.02)
        style_axis(ax)
        handles = [Line2D([], [], color=data[l]["color"], lw=1.3, label=l) for l in labels]
        handles += [
            Line2D([], [], color="k", lw=1.2, marker="o", ms=2.2, label="held-out AUROC"),
            Line2D([], [], color="k", lw=1.2, ls="--", marker="s", ms=2.0, label="disjoint-sample cosine"),
        ]
        ax.legend(handles=handles, loc="lower right", frameon=False, ncol=1, handlelength=2.2)

    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    curves(ax)
    fig.tight_layout()
    fig.savefig(OUT / "convergence-curves.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.9, 2.6), gridspec_kw={"width_ratios": [1, 1.25]})
    d = data["Qwen3-8B"]
    axa.scatter(logdev(d["c6"]), d["hard"], s=5, alpha=0.3, color="#2166ac", linewidths=0, rasterized=True)
    ticks_c = [0.995, 0.99, 0.98, 0.97]
    axa.set_xticks(logdev(np.array(ticks_c)))
    axa.set_xticklabels([str(t) for t in ticks_c])
    ticks_a = [0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
    axa.set_yticks(logdev(np.array(ticks_a)))
    axa.set_yticklabels([str(t) for t in ticks_a])
    p = numbers["Qwen3-8B"]["hardness_vs_c6"][8]
    axa.text(
        0.04,
        0.96,
        f"Pearson {p['pearson']:.2f}\nSpearman {p['spearman']:.2f}\n$n={len(d['c6'])}$",
        transform=axa.transAxes,
        va="top",
        fontsize=6.5,
    )
    axa.set_xlabel(r"direction convergence $\bar c_6$ (cosine)")
    axa.set_ylabel(r"task hardness: AUROC$_{N=8}$")
    axa.set_title(f"(A) Qwen3-8B, layer {d['layer']}", loc="left")
    style_axis(axa)
    curves(axb)
    axb.set_title("(B)", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "ood-convergence.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    with plt.rc_context(
        {
            "font.size": 5.6,
            "axes.titlesize": 6.2,
            "axes.labelsize": 5.8,
            "xtick.labelsize": 5.2,
            "ytick.labelsize": 5.2,
            "legend.fontsize": 5.0,
            "lines.linewidth": 1.0,
        }
    ):
        fig, (axa, axb) = plt.subplots(1, 2, figsize=(3.75, 1.95))
        for l in labels:
            d = data[l]
            g = d["grid"]
            med, lo, hi = band(d["a"])
            axa.fill_between(g, lo, hi, color=d["color"], alpha=0.10, linewidth=0)
            axa.plot(g, med, color=d["color"], lw=1.0, marker="o", ms=1.6)
            med, lo, hi = band(d["c"])
            axa.fill_between(g, lo, hi, color=d["color"], alpha=0.10, linewidth=0)
            axa.plot(g, med, color=d["color"], lw=1.0, ls="--", marker="s", ms=1.4)
        g = data[labels[0]]["grid"]
        axa.set_xscale("log", base=2)
        axa.set_xticks(g)
        axa.set_xticklabels([rf"$2^{{{int(np.log2(n))}}}$" for n in g])
        axa.set_xlabel("stories per pole, $N$", labelpad=1)
        axa.set_ylabel("median over concepts", labelpad=1)
        axa.set_ylim(-0.12, 1.02)
        axa.set_title("(A) learning curves", loc="left")
        style_axis(axa)
        handles = [Line2D([], [], color=data[l]["color"], lw=1.0, label=l) for l in labels]
        handles += [
            Line2D([], [], color="k", lw=0.9, marker="o", ms=1.6, label="AUROC"),
            Line2D([], [], color="k", lw=0.9, ls="--", marker="s", ms=1.4, label="cosine"),
        ]
        axa.legend(
            handles=handles, loc="lower right", frameon=False, handlelength=1.8, labelspacing=0.25, borderaxespad=0.2
        )

        d = data["Qwen3-8B"]
        floor = 1.0 / (2 * 25 * 25)
        x = logdev(d["c6"])
        for n, color in {2: "#e41a1c", 32: "#17becf", 8: "#000000"}.items():
            y = np.log10(np.clip(1.0 - d["a"][:, d["grid"].index(n)], floor, None))
            axb.scatter(
                x,
                y,
                s=2.4,
                alpha=0.55,
                color=color,
                linewidths=0,
                rasterized=False,
                label=rf"$N={n}$, $\rho={spearmanr(x, y)[0]:.2f}$",
            )
        ticks_c = [0.97, 0.98, 0.99, 0.995]
        axb.set_xticks(logdev(np.array(ticks_c)))
        axb.set_xticklabels([str(t) for t in ticks_c])
        ticks_a = [0.6, 0.8, 0.9, 0.95, 0.99, 0.999]
        axb.set_yticks(np.log10(1 - np.array(ticks_a)))
        axb.set_yticklabels([str(t) for t in ticks_a])
        axb.invert_xaxis()
        axb.invert_yaxis()
        axb.set_xlabel(r"convergence $\bar c_6$", labelpad=1)
        axb.set_ylabel(r"held-out AUROC$_N$", labelpad=1)
        axb.set_title(f"(B) Qwen3-8B, layer {d['layer']}", loc="left")
        hs, ls = axb.get_legend_handles_labels()
        order = sorted(range(len(ls)), key=lambda i: int(ls[i].split("=")[1].split("$")[0]))
        leg = axb.legend(
            [hs[i] for i in order],
            [ls[i] for i in order],
            loc="upper left",
            frameon=False,
            markerscale=2.5,
            handletextpad=0.2,
            labelspacing=0.25,
            borderaxespad=0.2,
        )
        for h in leg.legend_handles:
            h.set_alpha(1)
        style_axis(axb)
        fig.tight_layout(pad=0.2, w_pad=0.6)
        fig.savefig(OUT / "convergence-main.pdf", bbox_inches="tight", pad_inches=0.01, dpi=600)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    d = data["Qwen3-8B"]
    floor = 1.0 / (2 * 25 * 25)
    by_n = {2: "#fdae61", 8: "#d7191c", 32: "#2c7bb6"}
    x = logdev(d["c6"])
    for n, color in by_n.items():
        y = np.log10(np.clip(1.0 - d["a"][:, d["grid"].index(n)], floor, None))
        rho = spearmanr(x, y)[0]
        ax.scatter(
            x, y, s=4, alpha=0.35, color=color, linewidths=0, rasterized=True, label=rf"$N={n}$ ($\rho={rho:.2f}$)"
        )
    ticks_c = [0.995, 0.99, 0.98, 0.97]
    ax.set_xticks(logdev(np.array(ticks_c)))
    ax.set_xticklabels([str(t) for t in ticks_c])
    ticks_a = [0.6, 0.8, 0.9, 0.95, 0.99, 0.999]
    ax.set_yticks(np.log10(1 - np.array(ticks_a)))
    ax.set_yticklabels([str(t) for t in ticks_a])
    ax.set_xlabel(r"direction convergence $\bar c_6$ (cosine)")
    ax.set_ylabel(r"task hardness: held-out AUROC$_N$")
    ax.set_title(f"Qwen3-8B, layer {d['layer']}", loc="left")
    ax.invert_xaxis()
    ax.invert_yaxis()
    leg = ax.legend(loc="upper left", frameon=False, markerscale=2.5)
    for h in leg.legend_handles:
        h.set_alpha(1)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(OUT / "hardness-byN.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.35), sharey=True)
    rng = np.random.default_rng(0)
    for ax, l in zip(axes, labels):
        d = data[l]
        g = np.array(d["grid"])
        keep = rng.choice(len(d["a"]), 220, replace=False)
        for i, n in enumerate(g):
            ax.scatter(
                n * np.exp(rng.uniform(-0.05, 0.05, len(keep))),
                d["a"][keep, i],
                s=0.6,
                color="#b9bdc5",
                alpha=0.35,
                linewidths=0,
                rasterized=True,
                zorder=1,
            )
        med, lo, hi = band(d["a"])
        ax.fill_between(g, lo, hi, color="grey", alpha=0.15, linewidth=0, zorder=2)
        ax.plot(g, med, color="k", ls="--", lw=1.1, zorder=4)
        cls = np.array([classes[int(p)] for p in d["pids"]])
        for prefix, _, color in CLASSES:
            m = np.array([s.startswith(prefix) for s in cls])
            ax.plot(g, d["a"][m].mean(0), color=color, lw=1.5, marker="o", ms=2.2, zorder=5)
        pow2_axis(ax, list(g))
        ax.set_title(f"({chr(65 + labels.index(l))}) {l}, layer {d['layer']}", loc="left")
        ax.set_ylim(0.45, 1.01)
        style_axis(ax)
    axes[0].set_ylabel("held-out AUROC")
    handles = [Line2D([], [], color=c, lw=1.5, marker="o", ms=2.2, label=lab) for _, lab, c in CLASSES]
    handles += [
        Line2D([], [], color="k", ls="--", lw=1.1, label="median"),
        Line2D([], [], color="grey", lw=5, alpha=0.3, label=f"{LO:g}th--{HI:g}th percentiles"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(OUT / "ood-learning.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    out = args.numbers
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(numbers, indent=1, default=str))
    print(json.dumps(numbers, indent=1, default=str))


if __name__ == "__main__":
    main()
