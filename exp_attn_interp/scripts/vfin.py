from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ATTN = Path(".")
TRIG = Path("../exp_trigger_prompts/work")
MODELS = {
    "Qwen3-8B": (["judge", "judge-1217"], None),
    "Llama-3.1-8B-Instruct": (["judge-llama"], None),
    "gemma-2-9b-it": (["judge-gemma"], None),
}
ALL_X = [
    "Qwen3-8B",
    "Qwen3-4B",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B",
    "Qwen1.5-0.5B",
    "deepseek-llm-7b-chat",
    "Llama-3.1-8B-Instruct",
    "gemma-2-9b-it",
]
CLAIM_BSTAR = {
    "Qwen3-8B": 19,
    "Qwen3-4B": 21,
    "Qwen2.5-3B-Instruct": 24,
    "Qwen2.5-7B": 18,
    "Qwen1.5-0.5B": 16,
    "deepseek-llm-7b-chat": 17,
    "Llama-3.1-8B-Instruct": 15,
    "gemma-2-9b-it": 23,
}
CLAIM_PROBE = {
    "Qwen3-8B": 23,
    "Qwen3-4B": 23,
    "Qwen2.5-3B-Instruct": 26,
    "Qwen2.5-7B": 15,
    "Qwen1.5-0.5B": 14,
    "deepseek-llm-7b-chat": 15,
    "Llama-3.1-8B-Instruct": 15,
    "gemma-2-9b-it": 24,
}


def ranks(x):
    o = np.argsort(x)
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    return r


def sp(a, b):
    return float(np.corrcoef(ranks(np.asarray(a, float)), ranks(np.asarray(b, float)))[0, 1])


def pe(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else float("nan")


def bstar(m):
    neg = [b for b in range(len(m)) if m[b] < 0]
    last_neg = max(neg)
    return next(b for b in range(last_neg + 1, len(m)) if m[b] > 0)


def E_law(m, a, clip=0.02):
    B = len(m)
    P = np.concatenate([[0.0], np.cumsum(m + a)[:-1]])
    g = m / np.clip(P, clip, None)
    mp = np.clip(m, 0, None)
    E = np.zeros(B)
    for r in range(B):
        seed, e = 1.0, 0.0
        for b in range(r, B):
            e += mp[b] * seed
            seed *= 1 + g[b]
        E[r] = e
    S = np.array([np.prod(1 + np.clip(g[r:], None, 0)) for r in range(B)])
    A = np.array([mp[r:].sum() for r in range(B)])
    return E, S, A


def main():
    print("== §CF: b* and probe optimum (medians over 259 pairs, raw npz)")
    print("model | n | b* recomputed / claimed | probe(SNR argmax) recomputed / claimed")
    xd = {}
    for name in ALL_X:
        files = sorted(glob.glob(str(ATTN / f"work-x/{name}/xdecomp/pair-*.npz")))
        sm, sa, snr = [], [], []
        for f in files:
            z = np.load(f)
            sm.append(z["share_mlp_last"])
            sa.append(z["share_attn_last"])
            snr.append(z["story_snr"])
        m, a, snr, n = np.median(np.stack(sm), 0), np.median(np.stack(sa), 0), np.median(np.stack(snr), 0), len(files)
        xd[name] = (m, a, snr)
        bs = bstar(m)
        pr = int(np.argmax(snr))
        ok = "OK" if bs == CLAIM_BSTAR[name] and pr == CLAIM_PROBE[name] else "DIFF"
        print(f"{name:22s} | {n} | {bs} / {CLAIM_BSTAR[name]} | {pr} / {CLAIM_PROBE[name]}  {ok}")

    print("\n== §CH: the row-efficacy law vs the judged grids")
    print("model | E argmax | judged best (concept level) | Spearman/Pearson E vs (concept, antagonist, sum)")
    for name, (dirs, _) in MODELS.items():
        m, a, snr = xd[name]
        E, S, A = E_law(m, a)
        by = defaultdict(list)
        for d in dirs:
            for f in glob.glob(str(TRIG / d / "concept-shard-*.jsonl")):
                for line in open(f):
                    r = json.loads(line)
                    p, slot, lay, al = r["key"].split(":")
                    by[(int(lay[1:]), float(al[1:]))].append(float(r["score"]))
        rows = sorted({k[0] for k in by})
        lc = np.array([np.mean(by[(L, 0.5)]) for L in rows])
        la = np.array([np.mean(by[(L, -0.5)]) for L in rows])
        rows = np.array(rows)
        Ej = E[rows]
        print(
            f"{name:22s} | argmax {int(np.argmax(E[1 : len(m)])) + 1} | best {rows[int(np.argmax(lc))]} |"
            f" C {sp(Ej, lc):+.2f}/{pe(Ej, lc):+.2f} | A {sp(Ej, la):+.2f}/{pe(Ej, la):+.2f} | Sum {sp(Ej, lc + la):+.2f}/{pe(Ej, lc + la):+.2f}"
        )
        print(
            f"    controls: survival S Spearman-vs-concept {sp(S[rows], lc):+.2f} ; amplification A {sp(A[rows], lc):+.2f}"
        )
    print("\n  unjudged E argmax vs b*:")
    for name in ALL_X:
        if name in MODELS:
            continue
        m, a, snr = xd[name]
        E, _, _ = E_law(m, a)
        print(f"    {name:22s} E argmax {int(np.argmax(E[1 : len(m)])) + 1}, b* {bstar(m)}")

    print("\n== §CI: Spearman(gain, log10 variance) per block, raw vargain.npz")
    z = np.load(str(ATTN / "work/vargain.npz"), allow_pickle=True)
    for b in z["blocks"].tolist():
        names = [str(x) for x in z[f"b{b}_names"]]
        grp = np.array([x.split(":")[0] for x in names])
        lv = np.log10(np.clip(z[f"b{b}_var"], 1e-12, None))
        g1 = z[f"b{b}_gain_0.1"]
        ispc = grp == "tokpc"
        print(
            f"  block {b:2d}: all {sp(g1, lv):+.3f} | tokpc {sp(g1[ispc], lv[ispc]):+.3f} | nonpc {sp(g1[~ispc], lv[~ispc]):+.3f}"
            f" | concept med gain {np.median(g1[grp == 'concept']):+.3f} | random {np.median(g1[grp == 'random']):+.3f}"
        )


if __name__ == "__main__":
    main()
