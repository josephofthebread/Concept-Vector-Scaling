from __future__ import annotations

import argparse
import bisect
import glob
import json
import logging
import os
import time

import numpy as np
import torch
from safetensors.torch import load_file

log = logging.getLogger("turnread_tok")

MODELS = {
    "Qwen/Qwen3-8B": ("../exp_genvectors/work/vectors/Qwen_Qwen3-8B--assistant--semantic/diff.safetensors", 23),
    "unsloth/Meta-Llama-3.1-8B-Instruct": (
        "../exp_genvectors/work/vectors/unsloth_Meta-Llama-3.1-8B-Instruct--assistant--semantic/diff.safetensors",
        15,
    ),
    "google/gemma-2-9b-it": (
        "../exp_genvectors/work/vectors/google_gemma-2-9b-it--assistant--semantic/diff.safetensors",
        24,
    ),
}
ASSIST = {"thinking", "answer", "tool_call"}
MAXTURN = 7


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--work", default="turnwork_tok")
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()
    vec_path, layer = MODELS[args.model]
    os.makedirs(args.work, exist_ok=True)
    started = time.monotonic()

    files = []
    for e in args.episodes:
        files += glob.glob(os.path.join(e, "*.json")) if os.path.isdir(e) else glob.glob(e)
    files = sorted(set(files))[args.shard :: args.shards]
    log.info("%d episodes (shard %d/%d), layer %d", len(files), args.shard, args.shards, layer)

    from transformers import AutoConfig, AutoModel, AutoTokenizer

    AutoTokenizer.from_pretrained(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    cfg.num_hidden_layers = layer
    model = AutoModel.from_pretrained(
        args.model, config=cfg, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda"}
    ).eval()
    block = load_file(vec_path)["diff"]
    P = block.shape[1]
    vec = torch.nn.functional.normalize(block[layer].float(), dim=-1).cuda()
    log.info("model + vectors P=%d", P)

    done = 0
    with torch.inference_mode():
        for fi, f in enumerate(files):
            outp = os.path.join(args.work, os.path.basename(f).replace(".json", ".npz"))
            if os.path.exists(outp):
                done += 1
                continue
            d = json.loads(open(f).read())
            ids, roles = d["ids"], d["roles"]
            if len(ids) != len(roles) or len(ids) < 16:
                continue
            ons, prev_env = [], True
            for i, r in enumerate(roles):
                env = r in ("tool_result", "system")
                if not env and prev_env:
                    ons.append(i)
                prev_env = env
            if not ons:
                continue
            cutoff = ons[MAXTURN] if len(ons) > MAXTURN else len(roles)
            ids_e, roles_e = ids[:cutoff], roles[:cutoff]
            think_idx = [i for i in range(cutoff) if roles_e[i] == "thinking"]
            content = [i for i in range(cutoff) if roles_e[i] in ASSIST]
            if len(think_idx) < 1 or len(content) < 8:
                continue
            out = model(input_ids=torch.tensor([ids_e], device="cuda"), output_hidden_states=True, use_cache=False)
            unit = torch.nn.functional.normalize(out.hidden_states[layer][0].float(), dim=-1)
            cos = unit @ vec.T
            cc = cos[torch.tensor(content, device="cuda")]
            mean, std = cc.mean(0), cc.std(0).clamp_min(1e-6)
            ti = torch.tensor(think_idx, device="cuda")
            ztok = ((cos[ti] - mean) / std).float().cpu().numpy()
            turn = np.array([bisect.bisect_right(ons, i) - 1 for i in think_idx], dtype=np.uint8)
            from exp_reward_hack import workloads

            task = os.path.basename(f).split("-")[0]
            try:
                wl = workloads.load(task)
                imp, split = bool(wl.IMPOSSIBLE), wl.NAME
            except Exception:
                imp, split = True, task
            np.savez(outp, z=ztok, turn=turn, impossible=imp, split=split, ending=str(d.get("ending")), layer=layer)
            done += 1
            if fi % 20 == 0 or fi + 1 == len(files):
                log.info(
                    "ep %d/%d (%s, Ntok=%d, %.0fs)",
                    fi + 1,
                    len(files),
                    split,
                    len(think_idx),
                    time.monotonic() - started,
                )
    log.info("done %d in %.0fs -> %s", done, time.monotonic() - started, args.work)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        os._exit(1)
