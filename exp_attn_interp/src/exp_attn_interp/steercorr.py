from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

log = logging.getLogger("steercorr")


def ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x))
    return r


FAMILIES = {
    "qwen": ("work", [(30, 0), (27, 8), (34, 19), (34, 15), (30, 20), (26, 27)], 19),
    "llama": ("work-Meta-Llama-3.1-8B-Instruct", [(25, 18), (17, 20), (14, 24), (29, 31), (21, 19), (28, 20)], 13),
    "gemma": ("work-gemma-2-9b-it", [(40, 3), (26, 9), (31, 3), (23, 14), (29, 11), (40, 6)], 21),
}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--steer", type=Path, default=Path("work/steerability.json"))
    parser.add_argument("--rotation", type=Path, default=Path("work/rotation_per_concept.npy"))
    parser.add_argument("--out", type=Path, default=Path("work/steercorr.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    steer = json.loads(args.steer.read_text())
    rot = np.load(args.rotation)
    out: dict = {}
    for fi, (fam, (work, amps, band)) in enumerate(FAMILIES.items()):
        dd = Path(work) / "decomp"
        if not dd.exists():
            continue
        files = sorted(dd.glob("item-*.safetensors"))
        with safe_open(str(files[0]), framework="np") as h:
            read_layers = [int(x) for x in h.metadata()["read_layers"].split(",")]
        nL = len(read_layers)
        acc: dict[tuple[int, int], dict] = {}
        for f in files:
            with safe_open(str(f), framework="np") as h:
                k, side = int(h.get_tensor("pair_id")), int(h.get_tensor("side"))
                ca = h.get_tensor("c_attn").astype(np.float64)
                cm = h.get_tensor("c_mlp").astype(np.float64)
                tot = h.get_tensor("total").astype(np.float64)
            a = acc.setdefault(
                (k, side), {"ca": np.zeros_like(ca), "cm": np.zeros_like(cm), "tot": np.zeros(nL), "n": 0}
            )
            a["ca"] += ca
            a["cm"] += cm
            a["tot"] += tot
            a["n"] += 1
        feats: dict[int, dict] = {}
        li_last, li_mid = nL - 1, nL // 2
        for k in sorted({k for k, _ in acc}):
            if (k, 0) not in acc or (k, 1) not in acc:
                continue
            c, a = acc[(k, 0)], acc[(k, 1)]
            D = c["ca"] / c["n"] - a["ca"] / a["n"]
            Dm = c["cm"] / c["n"] - a["cm"] / a["n"]
            Dt = c["tot"] / c["n"] - a["tot"] / a["n"]
            attn_share = D.sum((1, 2)) / Dt
            amp_last = sum(D[li_last, b, h] for b, h in amps) / Dt[li_last]
            amp_mid = sum(D[li_mid, b, h] for b, h in amps if b < read_layers[li_mid]) / Dt[li_mid]
            per_block = D[li_last].sum(1) + Dm[li_last]
            cum = np.cumsum(per_block) / Dt[li_last]
            w = np.abs(D[li_last]).ravel()
            pr = float(w.sum() ** 2 / (w**2).sum())
            feats[k] = {
                "attn_share": attn_share.tolist(),
                "amp_last": float(amp_last),
                "amp_mid": float(amp_mid),
                "before_band": float(cum[band - 1]),
                "pr": pr,
                "signal_last": float(Dt[li_last]),
                "signal_mid": float(Dt[li_mid]),
            }
        ks = sorted(k for k in feats if str(k) in steer[fam])
        s = np.array([steer[fam][str(k)] for k in ks])
        r = rot[fi, ks]
        res: dict[str, Any] = {
            "n": len(ks),
            "read_layers": read_layers,
            "rho_rotation": float(np.corrcoef(ranks(r), ranks(s))[0, 1]),
        }
        cols = {
            "amp_last": np.array([feats[k]["amp_last"] for k in ks]),
            "amp_mid": np.array([feats[k]["amp_mid"] for k in ks]),
            "before_band": np.array([feats[k]["before_band"] for k in ks]),
            "pr": np.array([feats[k]["pr"] for k in ks]),
            "signal_last": np.array([feats[k]["signal_last"] for k in ks]),
            "signal_mid": np.array([feats[k]["signal_mid"] for k in ks]),
        }
        for i, L in enumerate(read_layers):
            cols[f"attn_share_L{L}"] = np.array([feats[k]["attn_share"][i] for k in ks])
        for name, x in cols.items():
            rho_xs = float(np.corrcoef(ranks(x), ranks(s))[0, 1])
            ra, rb, rc = ranks(x), ranks(s), ranks(r)
            X = np.stack([np.ones_like(rc), rc], 1)
            ra_r = ra - X @ np.linalg.lstsq(X, ra, rcond=None)[0]
            rb_r = rb - X @ np.linalg.lstsq(X, rb, rcond=None)[0]
            res[name] = {
                "rho": rho_xs,
                "rho_given_rotation": float(np.corrcoef(ra_r, rb_r)[0, 1]),
                "rho_with_rotation": float(np.corrcoef(ranks(x), ranks(r))[0, 1]),
            }
            log.info(
                "%s %-16s rho=%+.3f | rotation removed %+.3f | rho(x, rotation)=%+.3f",
                fam,
                name,
                res[name]["rho"],
                res[name]["rho_given_rotation"],
                res[name]["rho_with_rotation"],
            )
        log.info("%s rotation alone rho=%+.3f (n=%d)", fam, res["rho_rotation"], len(ks))
        out[fam] = res
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
