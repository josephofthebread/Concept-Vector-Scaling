from __future__ import annotations

import glob
import json
import string
import time
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(".")
H = 32
LAGS = np.arange(-H, H + 1)
PREFIX = [2, 4, 8, 16, 32, 64, 128, 256, 100000]
CLASSES = ["punct", "function", "content", "format", "other"]
FUNCTION_WORDS = set(
    """a an the and or but if while although though because so as of at by for with about against between into through during
before after above below to from up down in out on off over under again further then once here there when where why how all any both each
few more most other some such no nor not only own same than too very s t can will just don should now i me my myself we our ours ourselves you
your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who whom this
that these those am is are was were be been being have has had having do does did doing would could ought might must shall may also even
still yet already ever never often sometimes always usually perhaps maybe quite rather much many well back around away toward towards since
until whether either neither every another whose whereas within without upon among across along behind beyond beside near onto per via
one two three first second last next new old own way thing things something anything nothing everything someone anyone everyone""".split()
)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Return P(pos > neg) over all pairs, ties counted as 0.5."""
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def main() -> None:
    t0 = time.monotonic()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True)
    files = sorted(glob.glob(str(ROOT / "work/tokproj/pair-*.npz")))
    print(f"{len(files)} pair files", flush=True)
    R = 37
    cls_cache: dict[int, int] = {}
    cls_index = {c: i for i, c in enumerate(CLASSES)}
    Sxy = {name: np.zeros((R, len(LAGS))) for name in ("pol_u", "pol_e", "flag")}
    Sxx = {name: np.zeros((R, len(LAGS))) for name in ("pol_u", "pol_e", "flag")}
    Syy = {name: np.zeros((R, len(LAGS))) for name in ("pol_u", "pol_e", "flag")}
    ac_num = np.zeros((R, 8))
    ac_den = np.zeros((R, 8))
    lat_raw = []
    lat_cos = []
    cls_gap = []
    cls_spread = []
    cls_auc = []
    cls_share = np.zeros(len(CLASSES))
    n_tok_total = 0
    snr_all = []
    n_pairs = 0
    for fi, f in enumerate(files):
        d = np.load(f)
        proj = d["proj"].astype(np.float32)
        norm = d["norm"].astype(np.float32)
        pol = {
            "pol_u": d["pol_u"].astype(np.float32),
            "pol_e": d["pol_e"].astype(np.float32),
            "flag": d["flag"].astype(np.float32),
        }
        tokid = d["tok"]
        length = d["length"]
        pole = d["pole"]
        S = proj.shape[0]
        cos = proj / np.where(norm > 0, norm, np.nan)
        for i in range(S):
            L = int(length[i])
            if L < 8:
                continue
            x = proj[i, :L]
            mu = np.nanmean(x, 0, keepdims=True)
            sd = np.nanstd(x, 0, keepdims=True)
            X = (x - mu) / np.where(sd > 1e-6, sd, np.nan)
            X = np.nan_to_num(X)
            for name in pol:
                y = pol[name][i, :L]
                y = y - y.mean()
                if y.std() < 1e-6:
                    continue
                y = y / y.std()
                for li, h in enumerate(LAGS):
                    n_ov = L - abs(int(h))
                    if n_ov < 4:
                        continue
                    if h >= 0:
                        xs, ys = X[:n_ov], y[h : h + n_ov]
                    else:
                        xs, ys = X[-h : -h + n_ov], y[:n_ov]
                    Sxy[name][:, li] += xs.T @ ys
                    Sxx[name][:, li] += (xs**2).sum(0)
                    Syy[name][:, li] += (ys**2).sum()
            for lag in range(1, 9):
                ac_num[:, lag - 1] += (X[:-lag] * X[lag:]).sum(0)
                ac_den[:, lag - 1] += (X**2).sum(0)
        cpos = np.where(pole == 0)[0]
        cneg = np.where(pole == 1)[0]
        rows_raw = np.full((len(PREFIX), R), np.nan)
        rows_cos = np.full((len(PREFIX), R), np.nan)
        for pi, n in enumerate(PREFIX):
            m_raw = np.stack([np.nanmean(proj[i, : min(n, int(length[i]))], 0) for i in range(S)])
            m_cos = np.stack([np.nanmean(cos[i, : min(n, int(length[i]))], 0) for i in range(S)])
            for r in range(R):
                rows_raw[pi, r] = auroc(m_raw[cpos, r], m_raw[cneg, r])
                rows_cos[pi, r] = auroc(m_cos[cpos, r], m_cos[cneg, r])
        lat_raw.append(rows_raw)
        lat_cos.append(rows_cos)
        cl = np.full(tokid.shape, -1, dtype=np.int8)
        for i in range(S):
            for t in range(int(length[i])):
                tid = int(tokid[i, t])
                if tid not in cls_cache:
                    if tid >= 151643:
                        tclass = "format"
                    else:
                        s = tok.decode([tid])
                        st = s.strip()
                        if st == "":
                            tclass = "format"
                        elif all(
                            (not ch.isalnum())
                            and (ch in string.punctuation or unicodedata.category(ch).startswith("P"))
                            for ch in st
                        ):
                            tclass = "punct"
                        else:
                            low = st.lower().strip(string.punctuation)
                            if low.isalpha():
                                tclass = "function" if low in FUNCTION_WORDS else "content"
                            else:
                                tclass = "other"
                    cls_cache[tid] = cls_index[tclass]
                cl[i, t] = cls_cache[tid]
        gap = np.full((len(CLASSES), R), np.nan)
        spread = np.full((len(CLASSES), R), np.nan)
        cauc = np.full((len(CLASSES), R), np.nan)
        for ci in range(len(CLASSES)):
            means = np.full((S, R), np.nan)
            dev2 = np.zeros(R)
            cnt = 0
            for i in range(S):
                L = int(length[i])
                sel = cl[i, :L] == ci
                cls_share[ci] += sel.sum()
                if sel.sum() >= 2:
                    p = proj[i, :L][sel]
                    means[i] = p.mean(0)
                    dev2 += ((p - p.mean(0)) ** 2).sum(0)
                    cnt += sel.sum() - 1
            ok_p = cpos[~np.isnan(means[cpos, 0])]
            ok_n = cneg[~np.isnan(means[cneg, 0])]
            if len(ok_p) >= 2 and len(ok_n) >= 2 and cnt > 10:
                gap[ci] = means[ok_p].mean(0) - means[ok_n].mean(0)
                spread[ci] = np.sqrt(dev2 / cnt)
                for r in range(R):
                    cauc[ci, r] = auroc(means[ok_p, r], means[ok_n, r])
        n_tok_total += int(length.sum())
        cls_gap.append(gap)
        cls_spread.append(spread)
        cls_auc.append(cauc)
        allm = np.stack([np.nanmean(proj[i, : int(length[i])], 0) for i in range(S)])
        dev2 = np.zeros(R)
        cnt = 0
        for i in range(S):
            L = int(length[i])
            p = proj[i, :L]
            dev2 += ((p - p.mean(0)) ** 2).sum(0)
            cnt += L - 1
        snr_all.append((allm[cpos].mean(0) - allm[cneg].mean(0)) / np.sqrt(dev2 / cnt))
        n_pairs += 1
        if fi % 100 == 0:
            print(f"{fi}/{len(files)} {(time.monotonic() - t0) / 60:.1f} min", flush=True)
    out: dict = {
        "n_pairs": n_pairs,
        "n_tokens": n_tok_total,
        "lags": LAGS.tolist(),
        "prefix": PREFIX,
        "classes": CLASSES,
        "class_share": (cls_share / max(1, cls_share.sum())).tolist(),
    }
    win = {}
    for name in Sxy:
        corr = Sxy[name] / np.sqrt(Sxx[name] * Syy[name] + 1e-12)
        pos = np.clip(corr, 0, None)
        tot = pos.sum(1) + 1e-12
        com = (pos * LAGS[None, :]).sum(1) / tot
        fut = pos[:, LAGS > 0].sum(1) / tot
        past = pos[:, LAGS < 0].sum(1) / tot
        peak_lag = LAGS[corr.argmax(1)]
        half = (corr >= 0.5 * corr.max(1, keepdims=True)).sum(1)
        win[name] = {
            "corr": corr.tolist(),
            "centre_of_mass": com.tolist(),
            "future_mass": fut.tolist(),
            "past_mass": past.tolist(),
            "peak_lag": peak_lag.tolist(),
            "half_height_width": half.tolist(),
            "peak_corr": corr.max(1).tolist(),
            "corr_at_0": corr[:, H].tolist(),
            "corr_at_p1": corr[:, H + 1].tolist(),
            "corr_at_m1": corr[:, H - 1].tolist(),
            "corr_sum_past": corr[:, LAGS < 0].sum(1).tolist(),
            "corr_sum_future": corr[:, LAGS > 0].sum(1).tolist(),
        }
    out["window"] = win
    out["autocorr"] = (ac_num / np.maximum(ac_den, 1e-12)).tolist()
    LR = np.stack(lat_raw)
    LC = np.stack(lat_cos)
    out["latency"] = {
        "auroc_raw_mean": np.nanmean(LR, 0).tolist(),
        "auroc_raw_se": (np.nanstd(LR, 0) / np.sqrt(LR.shape[0])).tolist(),
        "auroc_cos_mean": np.nanmean(LC, 0).tolist(),
        "auroc_cos_se": (np.nanstd(LC, 0) / np.sqrt(LC.shape[0])).tolist(),
    }
    need = {}
    for thr in (0.85, 0.9, 0.95):
        row = []
        for r in range(R):
            m = np.nanmean(LR[:, :, r], 0)
            idx = np.where(m >= thr)[0]
            row.append(PREFIX[idx[0]] if len(idx) else None)
        need[str(thr)] = row
    out["latency"]["tokens_needed_raw"] = need
    G = np.stack(cls_gap)
    Sp = np.stack(cls_spread)
    CA = np.stack(cls_auc)
    snr_cls = G / Sp
    out["classes_snr_median"] = np.nanmedian(snr_cls, 0).tolist()
    out["classes_auc_mean"] = np.nanmean(CA, 0).tolist()
    out["snr_all_median"] = np.nanmedian(np.stack(snr_all), 0).tolist()
    snr_med = np.nanmedian(snr_cls, 0)
    out["ratio_function_over_content"] = (snr_med[cls_index["function"]] / snr_med[cls_index["content"]]).tolist()
    out["ratio_punct_over_content"] = (snr_med[cls_index["punct"]] / snr_med[cls_index["content"]]).tolist()
    (ROOT / "work/tmp/tokan.json").write_text(json.dumps(out))
    print(f"\npairs {n_pairs}, tokens {n_tok_total}, class share {dict(zip(CLASSES, np.round(out['class_share'], 3)))}")
    print(
        "\n(a) temporal window (pol_u output-form polarity): row | peak lag | corr@-1 | corr@0 | corr@+1 | past mass | future mass | center | half-width | peak corr"
    )
    w = win["pol_u"]
    for r in range(R):
        print(
            f"{r:3d} | {w['peak_lag'][r]:+3d} | {w['corr_at_m1'][r]:+.3f} | {w['corr_at_0'][r]:+.3f} | {w['corr_at_p1'][r]:+.3f} | {w['past_mass'][r]:.3f} | {w['future_mass'][r]:.3f} | {w['centre_of_mass'][r]:+.2f} | {w['half_height_width'][r]:2d} | {w['peak_corr'][r]:.3f}"
        )
    print("\n(a') same with pol_e (input form): row | peak lag | corr@0 | past mass | future mass | center")
    w = win["pol_e"]
    for r in range(R):
        print(
            f"{r:3d} | {w['peak_lag'][r]:+3d} | {w['corr_at_0'][r]:+.3f} | {w['past_mass'][r]:.3f} | {w['future_mass'][r]:.3f} | {w['centre_of_mass'][r]:+.2f}"
        )
    print("\n(b) prefix AUROC (raw mean projection), rows x prefix:", PREFIX)
    A = np.array(out["latency"]["auroc_raw_mean"])
    for r in range(R):
        print(
            f"{r:3d} | "
            + " ".join(f"{A[pi, r]:.3f}" for pi in range(len(PREFIX)))
            + f" | need0.9 {need['0.9'][r]} need0.95 {need['0.95'][r]}"
        )
    print("\n(c) token-class SNR median, rows x classes:", CLASSES, "| all-token SNR | fn/content | punct/content")
    Sm = np.array(out["classes_snr_median"])
    for r in range(R):
        print(
            f"{r:3d} | "
            + " ".join(f"{Sm[ci, r]:.3f}" for ci in range(len(CLASSES)))
            + f" | {out['snr_all_median'][r]:.3f} | {out['ratio_function_over_content'][r]:.3f} | {out['ratio_punct_over_content'][r]:.3f}"
        )
    print("\n(c') class-only story AUROC mean, rows x classes")
    Am = np.array(out["classes_auc_mean"])
    for r in range(R):
        print(f"{r:3d} | " + " ".join(f"{Am[ci, r]:.3f}" for ci in range(len(CLASSES))))
    print("\n(d) autocorrelation lags 1..8 by row")
    ac = np.array(out["autocorr"])
    for r in range(R):
        print(f"{r:3d} | " + " ".join(f"{ac[r, l]:.3f}" for l in range(8)))
    print(f"\ntotal {(time.monotonic() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
