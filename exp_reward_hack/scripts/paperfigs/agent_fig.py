import collections
import glob
import json
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "lines.linewidth": 1.5,
    }
)

MODELS = {
    "qwen": (
        "meanthink",
        24,
        "Qwen3-8B",
        "layer 24, thinking tokens",
        ["sycophancy", "fear", "honestly admits uncertainty", "confirmation bias"],
    ),
    "llama": (
        "assistant",
        14,
        "Llama-3.1-8B",
        "layer 14, assistant token",
        [
            "decomposing into subproblems",
            "planning the approach before calculating",
            "fear",
            "confirmation bias",
            "shortcut acceptance",
        ],
    ),
    "gemma": (
        "assistant",
        19,
        "gemma-2-9b",
        "layer 19, assistant token",
        ["rational thinking", "claims no feelings", "emotional load", "fear"],
    ),
}
LO, HI, MINN = -4, 8, 8
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
feats = json.load(open("../exp_lmsys_probe/data/features.json"))["features"]
feats.sort(key=lambda f: f["pair_id"])


def resolve(name):
    for f in feats:
        if f["concept"] == name:
            return f["pair_id"], f["concept"], f.get("antagonist", "")
    for f in feats:
        if f["concept"].startswith(name):
            return f["pair_id"], f["concept"], f.get("antagonist", "")
    raise KeyError(name)


W = HI - LO + 1
x = np.arange(LO, HI + 1)
dump = {"lo": LO, "hi": HI, "minn": MINN, "models": {}}
fig, axes = plt.subplots(1, 3, figsize=(6.3, 4.7))
for ax, (tag, (posname, layerval, nice, where, names)) in zip(axes, MODELS.items()):
    files = sorted(glob.glob(f"turnwork5_{tag}/*.npz"))
    e0 = np.load(files[0], allow_pickle=True)
    POS = [str(v) for v in e0["pos"]]
    LAYERS = [int(v) for v in e0["layers"]]
    pi, li = POS.index(posname), LAYERS.index(layerval)
    tgt = [resolve(n) for n in names]
    acc = {pid: {"s": np.zeros(W), "ss": np.zeros(W), "n": np.zeros(W)} for pid, _, _ in tgt}
    labels = [collections.Counter() for _ in range(W)]
    n_ep = 0
    n_wl = collections.Counter()
    for npzf in files:
        base = os.path.basename(npzf)[:-4]
        jf = f"episodes5/{base}.json"
        if not os.path.exists(jf):
            continue
        turns = json.load(open(jf))["turns"]
        tools = [t.get("tool") for t in turns]
        t0 = next((i for i, t in enumerate(tools) if t == "run_tests"), None)
        if t0 is None:
            continue
        n_ep += 1
        n_wl[base.split("-")[0]] += 1
        z = np.load(npzf, allow_pickle=True)["z"].astype(np.float32)
        for i in range(len(tools)):
            a = i - t0
            if LO <= a <= HI and i < z.shape[0]:
                name = turns[i].get("tool")
                args = turns[i].get("arguments") or {}
                path = os.path.basename(str(args.get("path", "")).rstrip("/")) if args.get("path") else ""
                labels[a - LO][
                    {
                        "read_file": "read " + path,
                        "write_file": "write " + path,
                        "list_files": "list files",
                        "run_tests": "run tests",
                        "submit": "submit",
                        "give_up": "give up",
                    }.get(name, name or "no tool call")
                ] += 1
                for pid, _, _ in tgt:
                    v = z[i, pi, li, pid]
                    acc[pid]["s"][a - LO] += v
                    acc[pid]["ss"][a - LO] += v**2
                    acc[pid]["n"][a - LO] += 1
    lab = [c.most_common(1)[0][0] if c else "-" for c in labels]
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(0, color="0.8", lw=0.5)
    rec = {
        "position": posname,
        "layer": layerval,
        "n_episodes": n_ep,
        "n_by_workload": dict(n_wl),
        "labels": lab,
        "concepts": [],
    }
    for k, (pid, cname, anta) in enumerate(tgt):
        a = acc[pid]
        n = a["n"]
        m = np.divide(a["s"], n, out=np.full(W, np.nan), where=n > 0)
        sd = np.sqrt(np.clip(np.divide(a["ss"], n, out=np.zeros(W), where=n > 0) - np.nan_to_num(m) ** 2, 0, None))
        sem = np.divide(sd, np.sqrt(n), out=np.zeros(W), where=n > 0)
        m[n < MINN] = np.nan
        ax.plot(x, m, color=COLORS[k], label=textwrap.fill(f"{cname} vs {anta}", 36))
        ax.fill_between(x, m - sem, m + sem, color=COLORS[k], alpha=0.14, lw=0)
        rec["concepts"].append(
            {
                "concept": cname,
                "antagonist": anta,
                "mean": [None if np.isnan(v) else round(float(v), 3) for v in m],
                "sem": [round(float(v), 3) for v in sem],
                "n": [int(v) for v in n],
            }
        )
    dump["models"][tag] = rec
    ax.set_title(f"{nice}\n{where}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{xi:+d} {lab[i]}" for i, xi in enumerate(x)], rotation=45, ha="right", rotation_mode="anchor")
    ax.grid(axis="x", color="0.93", lw=0.5)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.50),
        frameon=False,
        handlelength=1.2,
        borderaxespad=0.0,
        labelspacing=0.5,
    )
axes[0].set_ylabel("z-scored cosine")
fig.subplots_adjust(left=0.075, right=0.99, top=0.90, bottom=0.50, wspace=0.30)
fig.savefig("xresults/agent-turns.pdf")
fig.savefig("xresults/agent-turns.png", dpi=200)
json.dump(dump, open("xresults/agent-turns.json", "w"), indent=1)
for tag, rec in dump["models"].items():
    print(tag, rec["position"], rec["layer"], "episodes", rec["n_episodes"], rec["n_by_workload"])
    print("  labels:", rec["labels"])
    for c in rec["concepts"]:
        print("  %-42s" % (c["concept"] + " vs " + c["antagonist"]), c["mean"])
print("wrote xresults/agent-turns.{pdf,png,json}")
