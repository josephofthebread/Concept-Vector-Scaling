from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import time

import numpy as np
import torch
from safetensors.torch import load_file

log = logging.getLogger("turnread")

MODEL = "Qwen/Qwen3-8B"
VEC = "../exp_genvectors/work/vectors/Qwen_Qwen3-8B--assistant--semantic/diff.safetensors"
LAYERS = [12, 16, 18, 20, 22, 24, 25]
POS = ["assistant", "first", "lastthink", "meanthink", "toolcall"]
ASSIST = {"thinking", "answer", "tool_call"}


def label(path: str):
    import re

    n = os.path.basename(path)
    m = re.search(r"-([ABCD])-seed", n)
    if m:
        return True, "cfg-" + m.group(1)
    if "impossiblebench-original" in n:
        return False, "ib-original"
    if "impossiblebench-conflicting" in n:
        return True, "ib-conflicting"
    if "impossiblebench-oneoff" in n:
        return True, "ib-oneoff"
    return True, "gate"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", required=True, help="episode .json globs/dirs")
    ap.add_argument("--work", default="turnwork")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    started = time.monotonic()

    files = []
    for e in args.episodes:
        files += glob.glob(os.path.join(e, "*.json")) if os.path.isdir(e) else glob.glob(e)
    files = sorted(set(files))
    log.info("%d episodes", len(files))

    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(args.model)
    cfg.num_hidden_layers = max(LAYERS)
    model = AutoModel.from_pretrained(
        args.model, config=cfg, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda"}
    ).eval()
    block = load_file(VEC)["diff"]
    P = block.shape[1]
    vecs = [torch.nn.functional.normalize(block[l].float(), dim=-1).cuda() for l in LAYERS]
    log.info("model+%d vectors (P=%d) over layers %s", len(vecs), P, LAYERS)

    done = 0
    with torch.inference_mode():
        for fi, f in enumerate(files):
            d = json.loads(open(f).read())
            ids, roles = d["ids"], d["roles"]
            if len(ids) != len(roles) or len(ids) < 16:
                continue
            ons = [
                i for i in range(1, len(roles)) if roles[i] == "thinking" and roles[i - 1] in ("tool_result", "system")
            ]
            turns = []
            for t, o in enumerate(ons):
                end = ons[t + 1] if t + 1 < len(ons) else len(roles)
                think = [j for j in range(o, end) if roles[j] == "thinking"]
                tc = [j for j in range(o, end) if roles[j] == "tool_call"]
                turns.append(
                    {
                        "assistant": max(0, o - 1),
                        "first": o,
                        "lastthink": think[-1] if think else o,
                        "meanthink": think or [o],
                        "toolcall": tc[0] if tc else (think[-1] if think else o),
                    }
                )
            if not turns:
                continue
            content = torch.tensor([i for i in range(len(roles)) if roles[i] in ASSIST], device="cuda")
            if len(content) < 8:
                continue
            out = model(input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True, use_cache=False)
            T = len(turns)
            z = np.zeros((T, len(POS), len(LAYERS), P), dtype=np.float16)
            for li, layer in enumerate(LAYERS):
                unit = torch.nn.functional.normalize(out.hidden_states[layer][0].float(), dim=-1)
                cos = unit @ vecs[li].T
                cc = cos.index_select(0, content)
                mean, std = cc.mean(0), cc.std(0).clamp_min(1e-6)
                for t, tp in enumerate(turns):
                    for pi, name in enumerate(POS):
                        idx = tp[name]
                        v = cos[idx].mean(0) if isinstance(idx, list) else cos[idx]
                        z[t, pi, li] = ((v - mean) / std).half().cpu().numpy()
            imp, split = label(f)
            np.savez(
                os.path.join(args.work, os.path.basename(f).replace(".json", ".npz")),
                z=z,
                turns=np.arange(T),
                impossible=imp,
                split=split,
                ending=str(d.get("ending")),
                layers=np.array(LAYERS),
                pos=np.array(POS),
            )
            done += 1
            if fi % 20 == 0 or fi + 1 == len(files):
                log.info("ep %d/%d (%s, T=%d, %.0fs)", fi + 1, len(files), split, T, time.monotonic() - started)
    log.info("done: wrote %d episode npz in %.0fs -> %s", done, time.monotonic() - started, args.work)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        os._exit(1)
