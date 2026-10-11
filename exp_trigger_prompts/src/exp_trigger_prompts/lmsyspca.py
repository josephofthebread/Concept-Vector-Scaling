from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from .probe import VEC_ROOT, VEC_TAG, locate_span

log = logging.getLogger("lmsyspca")

MODEL = "Qwen/Qwen3-8B"
LAYERS = list(range(12, 25))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--prefills", type=Path, default=Path("work/lmsys-gap/lmsys_prefills.json"))
    parser.add_argument("--features", type=Path, default=Path("data/features.json"))
    parser.add_argument("--out", type=Path, default=Path("work/lmsys-gap/pca.json"))
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    prefills = json.loads(args.prefills.read_text())
    feats = json.loads(args.features.read_text())["features"]
    cname = [f["concept"] for f in feats]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto"
    )
    model.eval()
    device = model.device
    with safe_open(VEC_ROOT / VEC_TAG[MODEL] / "diff.safetensors", framework="pt") as handle:
        diff = handle.get_tensor("diff").float().to(device)
    unit = diff / diff.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    per_layer: dict[int, list[np.ndarray]] = {L: [] for L in LAYERS}
    pieces: list[str] = []
    for entry in prefills:
        messages = entry["conversation"]
        enc = cast(
            "Any",
            tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
            ),
        )  # type: ignore[arg-type]
        ids_t = enc["input_ids"]
        if ids_t.shape[1] > args.max_tokens:
            continue
        ids = cast("list[int]", ids_t[0].tolist())
        p0, p1 = locate_span(ids, messages[0]["content"], tok)
        r0, r1 = locate_span(ids, messages[1]["content"], tok, start=max(p1, 0))
        sel = [i for i in range(len(ids)) if (p0 <= i < p1) or (r0 <= i < r1)]
        if len(sel) < 2:
            continue
        with torch.no_grad():
            out = model(ids_t.to(device), output_hidden_states=True)
        toks = tok.convert_ids_to_tokens(ids)
        pieces.extend(toks[i] for i in sel)
        for L in LAYERS:
            h = out.hidden_states[L][0].float()
            hu = (h / h.norm(dim=-1, keepdim=True).clamp_min(1e-8))[sel]
            per_layer[L].append(hu.cpu().numpy().astype(np.float32))

    V = unit.cpu().numpy()
    results: dict[str, Any] = {"layers": LAYERS, "pieces": pieces}
    for L in LAYERS:
        U = np.concatenate(per_layer[L], 0)
        N = U.shape[0]
        Vc = V[L]
        C = U @ Vc.T
        assert C.min() > -1.01 and C.max() < 1.01, (float(C.min()), float(C.max()))
        mu = C.mean(0)
        dev = np.abs(C - mu)
        dom = dev.argmax(1)
        domfreq = np.bincount(dom, minlength=1036).astype(float)
        projvar = C.var(0)

        Uc = U - U.mean(0, keepdims=True)
        Sigma = (Uc.T @ Uc) / N
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
        order = eigenvalues[::-1]
        trace = float(eigenvalues.sum())
        lam1 = float(order[0])
        pr = float((eigenvalues.sum() ** 2) / (eigenvalues**2).sum())
        E = eigenvectors[:, ::-1][:, :3].T
        cosE = Vc @ E.T
        proj = Uc @ E.T

        extremes = {}
        for k in range(3):
            o = proj[:, k].argsort()
            extremes[f"e{k + 1}"] = {
                "top": [pieces[i] for i in o[-10:][::-1]],
                "bottom": [pieces[i] for i in o[:10]],
            }
        top_projvar = projvar.argsort()[::-1][:8]
        top_cosE1 = np.abs(cosE[:, 0]).argsort()[::-1][:8]
        top_dom = domfreq.argsort()[::-1][:8]
        results[str(L)] = {
            "N": N,
            "lambda1_frac": lam1 / trace,
            "participation_ratio": pr,
            "spearman_domfreq_projvar": spearman(domfreq, projvar),
            "spearman_domfreq_absCosE1": spearman(domfreq, np.abs(cosE[:, 0])),
            "spearman_projvar_cosE1sq": spearman(projvar, cosE[:, 0] ** 2),
            "top_projvar": [[cname[i], float(projvar[i]), float(abs(cosE[i, 0]))] for i in top_projvar],
            "top_cosE1": [[cname[i], float(abs(cosE[i, 0])), float(projvar[i])] for i in top_cosE1],
            "top_dominators": [[cname[i], int(domfreq[i])] for i in top_dom],
            "extremes": extremes,
        }
        log.info(
            "L%d N=%d lam1=%.3f PR=%.1f  rho(dom,projvar)=%.3f rho(dom,cosE1)=%.3f",
            L,
            N,
            lam1 / trace,
            pr,
            results[str(L)]["spearman_domfreq_projvar"],
            results[str(L)]["spearman_domfreq_absCosE1"],
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False))
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
