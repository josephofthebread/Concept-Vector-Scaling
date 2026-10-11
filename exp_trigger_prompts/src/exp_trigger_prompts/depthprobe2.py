from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from .probe import VEC_ROOT, VEC_TAG

MODEL = "Qwen/Qwen3-8B"
STOP = set(
    (
        "a an the to of in on at by for with from as into about over under and or but nor so if then is "
        "am are was were be been being have has had do does did will would can could this that these those "
        "it its there here you he she they we me him her them us my your his their our not no i".split()
    )
)
LETTER = re.compile(r"[a-zA-ZÀ-ɏ]")


def main() -> None:
    stim = json.loads(Path("data/probe_context.json").read_text())
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto"
    )
    model.eval()
    device = model.device
    nlayers = model.config.num_hidden_layers
    with safe_open(VEC_ROOT / VEC_TAG[MODEL] / "diff.safetensors", framework="pt") as h:
        diff = h.get_tensor("diff").float()
    units = {}
    for c in stim.values():
        v = diff[:, c["pair_id"], :].to(device)
        units[c["pair_id"]] = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def measure(text: str, marker: str, unit: torch.Tensor) -> dict[str, list[float]] | None:
        enc = cast("Any", tok(text, return_tensors="pt", return_offsets_mapping=True))
        offs = enc["offset_mapping"][0].tolist()
        pat = r"\b" + re.escape(marker) + (r"\b" if len(marker) <= 2 else r"\w*\b")
        m = re.search(pat, text, re.IGNORECASE)
        pos: int | None = None
        if m:
            for i, (a, b) in enumerate(offs):
                if a <= m.start() < b:
                    pos = i
                    break
        if pos is None:
            return None
        ids = enc["input_ids"].to(device)
        toks = [str(tok.decode([t])) for t in ids[0].tolist()]
        content = [i for i in range(1, ids.shape[1]) if LETTER.search(toks[i]) and toks[i].strip().lower() not in STOP]
        surround = [i for i in content if i != pos]
        if pos not in content or len(surround) < 2:
            return None
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        sidx = torch.tensor(surround, device=device)
        mk, su = [], []
        for L in range(nlayers + 1):
            hs = out.hidden_states[L][0].float()
            u = unit[L]
            hm = hs[pos] / hs[pos].norm().clamp_min(1e-8)
            mk.append(float(hm @ u))
            hsr = hs[sidx] / hs[sidx].norm(dim=-1, keepdim=True).clamp_min(1e-8)
            su.append(float((hsr @ u).mean()))
        return {"marker": mk, "surround": su}

    results: dict[str, Any] = {}
    for name, c in stim.items():
        u = units[c["pair_id"]]
        cond: dict[str, list[dict[str, list[float]]]] = {"in": [], "out": []}
        for k in ("in", "out"):
            for s in c[k]:
                m = measure(s, c["marker"], u)
                if m:
                    cond[k].append(m)
        agg = {}
        for k in ("in", "out"):
            if cond[k]:
                agg[k] = {
                    f: [sum(r[f][L] for r in cond[k]) / len(cond[k]) for L in range(nlayers + 1)]
                    for f in ("marker", "surround")
                }
        results[name] = {"marker": c["marker"], "n_in": len(cond["in"]), "n_out": len(cond["out"]), "agg": agg}
        print(f"{name[:26]:26s} marker={c['marker']:10s} n={len(cond['in'])}/{len(cond['out'])}", flush=True)

    Path("work/depthprobe2").mkdir(parents=True, exist_ok=True)
    Path("work/depthprobe2/depthprobe2.json").write_text(
        json.dumps({"model": MODEL, "nlayers": nlayers, "results": results}, ensure_ascii=False)
    )

    allc = [r["agg"] for r in results.values() if "in" in r["agg"] and "out" in r["agg"]]
    print("\nlayer | markerIN | surrIN surrOUT  surrGAP(IN-OUT)   (mean over concepts)")
    for L in range(nlayers + 1):
        mi = sum(a["in"]["marker"][L] for a in allc) / len(allc)
        si = sum(a["in"]["surround"][L] for a in allc) / len(allc)
        so = sum(a["out"]["surround"][L] for a in allc) / len(allc)
        bar = "#" * max(0, round((si - so) * 80))
        print(f"  {L:2d}  |  {mi:+.3f} |  {si:+.3f} {so:+.3f}   {si - so:+.4f} {bar}")


if __name__ == "__main__":
    main()
