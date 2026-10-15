from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import numpy as np
from datasets import load_dataset

TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)
MAX_VOCAB = 6000
FOLDS = 5


def tfidf(docs: list[str]) -> np.ndarray:
    tok = []
    for d in docs:
        w = TOKEN.findall(d.lower())
        tok.append(w + [f"{w[i]}_{w[i + 1]}" for i in range(len(w) - 1)])
    df: collections.Counter = collections.Counter()
    for t in tok:
        df.update(set(t))
    vocab = {
        w: i
        for i, (w, _) in enumerate(sorted(((w, c) for w, c in df.items() if c >= 2), key=lambda kv: -kv[1])[:MAX_VOCAB])
    }
    n, v = len(docs), len(vocab)
    x = np.zeros((n, v), np.float32)
    for i, t in enumerate(tok):
        for w, c in collections.Counter(t).items():
            j = vocab.get(w)
            if j is not None:
                x[i, j] = c
    idf = np.zeros(v, np.float32)
    for w, j in vocab.items():
        idf[j] = np.log((n + 1) / (df[w] + 1)) + 1.0
    x *= idf
    x /= np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)
    return x, list(vocab)


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    order = s.argsort()
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main() -> None:
    rng = np.random.default_rng(0)
    ds = load_dataset("AntonKorznikov/feature_stories", split="train")
    groups: dict = collections.defaultdict(lambda: {"pole": [], "ant": []})
    meta: dict = {}
    for r in ds:
        key = (r["concept"], r["antagonist"])
        g = groups[(key, r["language"])]
        g["pole"].append(r["concept_text"])
        g["ant"].append(r["antagonist_text"])
        meta.setdefault(key, r["class_name"])

    pairs = sorted({k for (k, _) in groups})
    langs = sorted({lang for (_, lang) in groups})

    P, L, Y, S = [], [], [], []
    summary = []
    for pi, pair in enumerate(pairs):
        start = len(S)
        by_lang = {}
        words: collections.Counter = collections.Counter()
        for li, lang in enumerate(langs):
            g = groups.get((pair, lang))
            if not g:
                continue
            docs = g["pole"] + g["ant"]
            y = np.array([1] * len(g["pole"]) + [0] * len(g["ant"]))
            x, vocab = tfidf(docs)
            fold = rng.integers(0, FOLDS, len(docs))
            score = np.zeros(len(docs), np.float32)
            for f in range(FOLDS):
                tr, te = fold != f, fold == f
                if not te.any() or y[tr].sum() == 0 or (y[tr] == 0).sum() == 0:
                    continue
                w = x[tr][y[tr] == 1].mean(0) - x[tr][y[tr] == 0].mean(0)
                score[te] = x[te] @ w
            by_lang[lang] = auroc(y, score)
            wfull = x[y == 1].mean(0) - x[y == 0].mean(0)
            for j in np.argsort(-wfull)[:10]:
                words[vocab[j]] += float(wfull[j])
            P.extend([pi] * len(docs))
            L.extend([li] * len(docs))
            Y.extend(y.tolist())
            S.extend(score.tolist())
        allscore = np.array(S[start:], np.float32)
        ally = np.array(Y[start:], np.int8)
        summary.append(
            {
                "pair_idx": pi,
                "concept": pair[0],
                "antagonist": pair[1],
                "class": meta[pair],
                "auroc_pooled": auroc(ally, allscore) if len(ally) else None,
                "auroc_by_lang": {ll: by_lang.get(ll) for ll in langs},
                "shortcut_words": [w for w, _ in words.most_common(10)],
            }
        )
        if pi % 100 == 0:
            print(f"pair {pi}/{len(pairs)} {pair[0]!r} pooled_auroc={summary[-1]['auroc_pooled']}", flush=True)

    out = Path("work/lexbias")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "perstory.npz",
        pair_idx=np.array(P, np.int16),
        lang_idx=np.array(L, np.int8),
        label=np.array(Y, np.int8),
        score=np.array(S, np.float32),
    )
    (out / "summary.json").write_text(
        json.dumps({"pairs": [p for p in pairs], "langs": langs, "summary": summary}, ensure_ascii=False)
    )
    print(f"wrote {len(S)} story scores over {len(pairs)} pairs x {len(langs)} langs")


if __name__ == "__main__":
    main()
