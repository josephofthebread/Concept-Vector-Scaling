from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from .probe import VEC_ROOT, VEC_TAG

MODEL = "Qwen/Qwen3-8B"
K = 10
REF = 18
BATCH = 128


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto"
    )
    model.eval()
    device = model.device
    nL, H = model.config.num_hidden_layers, model.config.hidden_size
    unemb = model.get_output_embeddings().weight.float()
    gamma = model.model.norm.weight.float().to(device)
    with safe_open(VEC_ROOT / VEC_TAG[MODEL] / "diff.safetensors", framework="pt") as h:
        diff = h.get_tensor("diff").float()
    npairs = diff.shape[1]
    base = float(np.sqrt(K / H))

    Vref = diff[REF].to(device)
    Vref = Vref / Vref.norm(dim=1, keepdim=True).clamp_min(1e-8)
    concept_markers: list[list[str]] = []
    vocab: dict[str, int] = {}
    for pid in range(npairs):
        ids = torch.topk(unemb @ (gamma * Vref[pid]), K * 2).indices.tolist()
        words: list[str] = []
        for tid in ids:
            w = str(tok.decode([tid])).strip().lower()
            if w and "�" not in w and w not in words:
                words.append(w)
                vocab.setdefault(w, len(vocab))
            if len(words) == K:
                break
        concept_markers.append(words)
    uniq = list(vocab)
    print(f"{len(uniq)} unique markers across {npairs} concepts", flush=True)

    cache = torch.zeros(len(uniq), nL + 1, H, device=device)
    for s in range(0, len(uniq), BATCH):
        chunk = uniq[s : s + BATCH]
        texts = [" " + w for w in chunk]
        enc = cast("Any", tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True))
        ids = enc["input_ids"].to(device)
        with torch.no_grad():
            hs = model(ids, attention_mask=enc["attention_mask"].to(device), output_hidden_states=True).hidden_states
        for bi, w in enumerate(chunk):
            offs = enc["offset_mapping"][bi].tolist()
            pos = next((i for i, (a, b) in enumerate(offs) if a <= 1 < b), None)
            if pos is None:
                pos = int(enc["attention_mask"][bi].sum()) - 1
            cache[vocab[w]] = torch.stack([hs[L][bi, pos].float() for L in range(nL + 1)])
        if s % (BATCH * 10) == 0:
            print(f"  markers {s}/{len(uniq)}", flush=True)

    pl: dict[int, dict[str, list[float]]] = {L: {"cos": [], "rank": [], "lexfrac": []} for L in range(nL + 1)}
    for pid in range(npairs):
        idx = [vocab[w] for w in concept_markers[pid]]
        if len(idx) < 3:
            continue
        R = cache[idx].double()
        for L in range(nL + 1):
            M = R[:, L, :]
            k = M.shape[0]
            Mn = M / M.norm(dim=1, keepdim=True).clamp_min(1e-9)
            meancos = float(((Mn @ Mn.T).abs().sum() - k) / (k * (k - 1)))
            s = torch.linalg.svdvals(M)
            pl[L]["rank"].append(float(s.sum() ** 2 / (s**2).sum()))
            pl[L]["cos"].append(meancos)
            q, _ = torch.linalg.qr(M.T)
            x = diff[L, pid].double().to(device)
            pl[L]["lexfrac"].append(float((q @ (q.T @ x)).norm() / x.norm().clamp_min(1e-9)))

    out = {
        "model": MODEL,
        "K": K,
        "ref_layer": REF,
        "H": H,
        "baseline_lexfrac": base,
        "layers": {L: {m: float(np.mean(pl[L][m])) for m in ("cos", "rank", "lexfrac")} for L in range(nL + 1)},
    }
    Path("work/lexcohere").mkdir(parents=True, exist_ok=True)
    Path("work/lexcohere/lexcohere.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\nK={K}, H={H}, random-K lexfrac baseline = {base:.3f}")
    print("layer | mean|cos|  effrank/K  lexfrac  (ratio vs baseline)")
    for L in range(nL + 1):
        c, r, lf = (out["layers"][L][m] for m in ("cos", "rank", "lexfrac"))
        print(f"  {L:2d}  |  {c:.3f}     {r:4.1f}/{K}    {lf:.3f}   ({lf / base:.1f}x)")


if __name__ == "__main__":
    main()
