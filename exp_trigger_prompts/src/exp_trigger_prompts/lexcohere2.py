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
LAYERS = [0, 2, 4, 6, 8, 10, 12, 15, 18, 21, 25, 30, 36]


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

    Vref = diff[REF].to(device)
    Vref = Vref / Vref.norm(dim=1, keepdim=True).clamp_min(1e-8)
    concept_markers, vocab = [], {}
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
    print(f"{len(uniq)} unique markers", flush=True)

    cache = torch.zeros(len(uniq), nL + 1, H, device=device)
    for s in range(0, len(uniq), BATCH):
        chunk = uniq[s : s + BATCH]
        enc = cast("Any", tok([" " + w for w in chunk], return_tensors="pt", padding=True, return_offsets_mapping=True))
        with torch.no_grad():
            hs = model(
                enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device), output_hidden_states=True
            ).hidden_states
        for bi, w in enumerate(chunk):
            offs = enc["offset_mapping"][bi].tolist()
            pos = next((i for i, (a, b) in enumerate(offs) if a <= 1 < b), int(enc["attention_mask"][bi].sum()) - 1)
            cache[vocab[w]] = torch.stack([hs[L][bi, pos].float() for L in range(nL + 1)])

    res = {}
    for L in LAYERS:
        U = torch.zeros(npairs, H, device=device)
        align = []
        ok = []
        for pid in range(npairs):
            idx = [vocab[w] for w in concept_markers[pid]]
            if len(idx) < 3:
                continue
            M = cache[idx][:, L, :].double()
            u = torch.linalg.svd(M, full_matrices=False)[2][0]
            U[pid] = (u / u.norm()).float()
            dv = diff[L, pid].double().to(device)
            align.append(float((u @ dv).abs() / (u.norm() * dv.norm()).clamp_min(1e-9)))
            ok.append(pid)
        Uok = U[ok]
        Uok = Uok / Uok.norm(dim=1, keepdim=True).clamp_min(1e-9)
        C = (Uok @ Uok.T).abs()
        m = C.shape[0]
        univ = float((C.sum() - m) / (m * (m - 1)))
        res[L] = {"universality": round(univ, 3), "concept_alignment": round(float(np.mean(align)), 3), "n": m}
        print(
            f"L{L:2d}: universality(cross-concept |cos| of collapse dir)={univ:.3f}  "
            f"concept_alignment(|cos| with diff)={np.mean(align):.3f}  n={m}",
            flush=True,
        )

    Path("work/lexcohere2").mkdir(parents=True, exist_ok=True)
    Path("work/lexcohere2/lexcohere2.json").write_text(
        json.dumps({"model": MODEL, "layers": res}, ensure_ascii=False, indent=1)
    )
    print("\nSEMANTIC => low universality + high concept_alignment; ARTIFACT => high universality + low alignment")


if __name__ == "__main__":
    main()
