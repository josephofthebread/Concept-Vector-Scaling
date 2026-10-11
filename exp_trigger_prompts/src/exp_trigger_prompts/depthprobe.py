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

    def cos_curve(text: str, marker: str, unit: torch.Tensor) -> list[float] | None:
        enc = cast("Any", tok(text, return_tensors="pt", return_offsets_mapping=True))
        offsets = enc["offset_mapping"][0].tolist()
        m = re.search(r"\b" + re.escape(marker) + r"\b", text, re.IGNORECASE)
        pos: int | None = None
        if m:
            cs = m.start()
            for i, (a, b) in enumerate(offsets):
                if a <= cs < b:
                    pos = i
                    break
        if pos is None:
            return None
        ids = enc["input_ids"].to(device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        curve = []
        for L in range(nlayers + 1):
            h = out.hidden_states[L][0, pos].float()
            h = h / h.norm().clamp_min(1e-8)
            curve.append(float(h @ unit[L]))
        return curve

    results: dict[str, Any] = {}
    for name, c in stim.items():
        pid = c["pair_id"]
        u = units[pid]
        cond: dict[str, list[list[float]]] = {"in": [], "out": []}
        for k in ("in", "out"):
            for s in c[k]:
                cv = cos_curve(s, c["marker"], u)
                if cv is not None:
                    cond[k].append(cv)
        mean = {k: [sum(col) / len(col) for col in zip(*cond[k])] for k in ("in", "out") if cond[k]}
        results[name] = {
            "pair_id": pid,
            "marker": c["marker"],
            "n_in": len(cond["in"]),
            "n_out": len(cond["out"]),
            "mean": mean,
        }
        print(f"{name[:26]:26s} marker={c['marker']:10s} n={len(cond['in'])}/{len(cond['out'])}", flush=True)

    out = Path("work/depthprobe/depthprobe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": MODEL, "nlayers": nlayers, "results": results}, ensure_ascii=False))

    print("\nlayer |  cos_IN  cos_OUT   gap   (mean over concepts)")
    allc = [r["mean"] for r in results.values() if "in" in r["mean"] and "out" in r["mean"]]
    for L in range(nlayers + 1):
        ci = sum(m["in"][L] for m in allc) / len(allc)
        co = sum(m["out"][L] for m in allc) / len(allc)
        bar = "#" * max(0, round((ci - co) * 40))
        print(f"  {L:2d}  |  {ci:+.3f}  {co:+.3f}  {ci - co:+.3f} {bar}")
    print("\nper-concept gap (cos_IN - cos_OUT) at shallow L5 vs deep L25:")
    for name, r in results.items():
        m = r["mean"]
        if "in" in m and "out" in m:
            print(
                f"  {name[:26]:26s} L5 gap={m['in'][5] - m['out'][5]:+.3f}   L25 gap={m['in'][25] - m['out'][25]:+.3f}"
            )


if __name__ == "__main__":
    main()
