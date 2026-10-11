from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from safetensors import safe_open
from scipy.stats import pearsonr, spearmanr

from .probe import VEC_ROOT, VEC_TAG

QWEN = "Qwen/Qwen3-8B"
REF = 21
SEM = Path("../exp_gendata/data/semantic.parquet")
LEXBIAS = Path("../exp_gendata/work/lexbias/summary.json")
CAP = 300
MAX_VOCAB = 5000
FOLDS = 5
TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)


def tokenize(t: str) -> list[str]:
    w = TOKEN.findall(t.lower())
    return w + [f"{w[i]}_{w[i + 1]}" for i in range(len(w) - 1)]


def oof_auroc(pole: list[str], ant: list[str], rng: np.random.Generator) -> float:
    docs = pole + ant
    if len(pole) < 4 or len(ant) < 4:
        return float("nan")
    tok = [tokenize(d) for d in docs]
    df: collections.Counter = collections.Counter()
    for t in tok:
        df.update(set(t))
    vocab = {
        w: i
        for i, (w, _) in enumerate(sorted(((w, c) for w, c in df.items() if c >= 2), key=lambda kv: -kv[1])[:MAX_VOCAB])
    }
    x = np.zeros((len(docs), len(vocab)), np.float32)
    for i, t in enumerate(tok):
        for word, c in collections.Counter(t).items():
            j = vocab.get(word)
            if j is not None:
                x[i, j] = c
    idf = np.array([np.log((len(docs) + 1) / (df[w] + 1)) + 1.0 for w in vocab], np.float32)
    x *= idf
    x /= np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)
    y = np.array([1] * len(pole) + [0] * len(ant))
    fold = rng.integers(0, FOLDS, len(docs))
    score = np.zeros(len(docs), np.float32)
    for f in range(FOLDS):
        tr, te = fold != f, fold == f
        if not te.any() or y[tr].sum() == 0 or (y[tr] == 0).sum() == 0:
            continue
        w = x[tr][y[tr] == 1].mean(0) - x[tr][y[tr] == 0].mean(0)
        score[te] = x[te] @ w
    order = score.argsort()
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def partial(a: np.ndarray, b: np.ndarray, ctrl: np.ndarray) -> float:
    """Pearson of a,b after linearly regressing ctrl out of both (finite rows only)."""
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(ctrl)
    A, B, C = a[m], b[m], ctrl[m]
    G = np.c_[np.ones_like(C), C]
    ra = A - G @ np.linalg.lstsq(G, A, rcond=None)[0]
    rb = B - G @ np.linalg.lstsq(G, B, rcond=None)[0]
    return float(pearsonr(ra, rb)[0])


def main() -> None:
    rng = np.random.default_rng(0)
    meta = pq.read_table(VEC_ROOT / VEC_TAG[QWEN] / "pairs.parquet").to_pydict()
    pair_id = list(meta["pair_id"])
    concept = list(meta["concept"])
    n = len(pair_id)
    reliab = np.array([float(x) for x in meta[f"L{REF:02d}_directional_agreement"]])
    row_of_pid = {p: i for i, p in enumerate(pair_id)}

    repro = np.load("work/crossrepro.npz", allow_pickle=True)["repro"]

    with safe_open(VEC_ROOT / VEC_TAG[QWEN] / "diff.safetensors", framework="pt") as h:
        diff = h.get_tensor("diff").float().numpy()
    nL = diff.shape[0]
    U = diff / (np.linalg.norm(diff, axis=2, keepdims=True) + 1e-9)
    cos_ref = np.abs(np.einsum("lnh,nh->ln", U, U[REF]))
    conv_auc = cos_ref.mean(0)
    conv_layer = np.array([(np.argmax(cos_ref[:, i] >= 0.9)) / (nL - 1) for i in range(n)])

    lb = {r["pair_idx"]: r for r in json.loads(LEXBIAS.read_text())["summary"]}
    lexauroc = np.full(n, np.nan)
    lang_min = np.full(n, np.nan)
    lang_std = np.full(n, np.nan)
    for pi, r in lb.items():
        i = row_of_pid.get(pi, pi if pi < n else None)
        if i is None:
            continue
        lexauroc[i] = r["auroc_pooled"] if r["auroc_pooled"] is not None else np.nan
        vals = [v for v in r["auroc_by_lang"].values() if v is not None]
        if vals:
            lang_min[i], lang_std[i] = float(np.min(vals)), float(np.std(vals))

    tbl = pq.read_table(SEM, columns=["pair_id", "generator_id", "genre_id", "concept_text", "antagonist_text"])
    tbl = tbl.take(pc.sort_indices(tbl, sort_keys=[("pair_id", "ascending")]))
    pids = tbl["pair_id"].to_numpy()
    starts = np.searchsorted(pids, np.array(sorted(set(pids.tolist()))))
    bounds = list(starts) + [len(pids)]
    uniq_pids = sorted(set(pids.tolist()))

    gen_min, gen_std = np.full(n, np.nan), np.full(n, np.nan)
    genre_min, genre_std = np.full(n, np.nan), np.full(n, np.nan)
    for k, pid in enumerate(uniq_pids):
        i = row_of_pid.get(int(pid))
        if i is None:
            continue
        lo, hi = bounds[k], bounds[k + 1]
        gen = tbl["generator_id"].slice(lo, hi - lo).to_numpy()
        genre = tbl["genre_id"].slice(lo, hi - lo).to_numpy()
        pole_all = tbl["concept_text"].slice(lo, hi - lo).to_pylist()
        ant_all = tbl["antagonist_text"].slice(lo, hi - lo).to_pylist()
        for arr, dst_min, dst_std in ((gen, gen_min, gen_std), (genre, genre_min, genre_std)):
            aur = []
            for lv in np.unique(arr):
                sel = np.where(arr == lv)[0]
                if len(sel) > CAP:
                    sel = rng.choice(sel, CAP, replace=False)
                a = oof_auroc([pole_all[j] for j in sel], [ant_all[j] for j in sel], rng)
                if np.isfinite(a):
                    aur.append(a)
            if aur:
                dst_min[i], dst_std[i] = float(np.min(aur)), float(np.std(aur))
        if k % 100 == 0:
            print(f"  covariate AUROC {k}/{len(uniq_pids)}", flush=True)

    predictor_map = {
        "reliability": reliab,
        "lexAUROC": lexauroc,
        "conv_layer": conv_layer,
        "conv_auc": conv_auc,
        "lang_min": lang_min,
        "lang_std": lang_std,
        "gen_min": gen_min,
        "gen_std": gen_std,
        "genre_min": genre_min,
        "genre_std": genre_std,
    }

    print(f"\ntarget = repro (median {np.median(repro):.3f})   n={n}")
    print(f"{'predictor':12} {'spearman':>9} {'pearson':>8} {'partial|reliab':>15}")
    corr = {}
    for name, x in predictor_map.items():
        m = np.isfinite(x) & np.isfinite(repro)
        sp = float(spearmanr(x[m], repro[m]).statistic)
        pe = float(pearsonr(x[m], repro[m])[0])
        pa = np.nan if name == "reliability" else partial(repro, x, reliab)
        corr[name] = {
            "spearman": round(sp, 3),
            "pearson": round(pe, 3),
            "partial_reliab": None if np.isnan(pa) else round(pa, 3),
            "n": int(m.sum()),
        }
        print(f"{name:12} {sp:9.3f} {pe:8.3f} {('' if np.isnan(pa) else f'{pa:.3f}'):>15}")

    Path("work/reprocorr").mkdir(parents=True, exist_ok=True)
    Path("work/reprocorr/reprocorr.json").write_text(
        json.dumps(
            {
                "concept": concept,
                "repro": [round(float(x), 4) for x in repro],
                "predictors": {
                    k: [None if not np.isfinite(v) else round(float(v), 4) for v in x] for k, x in predictor_map.items()
                },
                "correlations": corr,
            },
            ensure_ascii=False,
        )
    )
    print("\nwrote work/reprocorr/reprocorr.json")


if __name__ == "__main__":
    main()
