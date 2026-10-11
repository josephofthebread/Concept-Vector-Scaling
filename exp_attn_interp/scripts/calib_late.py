import json
import sys
from pathlib import Path

import torch

from exp_attn_interp.common import load_model

src = Path(sys.argv[1] if len(sys.argv) > 1 else "work/posteer-foot/orig.jsonl")
recs = [json.loads(l) for l in src.open()]
recs = [r for r in recs if abs(r["alpha"]) < 1e-9 and r["gen_tokens"] > 0]
print(f"{len(recs)} unsteered records from {src}")
model, tok = load_model(offline=True, model_name="Qwen/Qwen3-8B")
rows = (22, 28, 32)
sums = {r: 0.0 for r in rows}
cnt = 0
with torch.inference_mode():
    for i, r in enumerate(recs):
        msgs = ([{"role": "system", "content": r["system"]}] if r.get("system") else []) + [
            {"role": "user", "content": r["prompt"]}
        ]
        prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        pids = tok(prefix, add_special_tokens=False)["input_ids"]
        rids = tok(r["response"], add_special_tokens=False)["input_ids"]
        ids = torch.tensor([pids + rids], device=model.device)
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        for row in rows:
            sums[row] += float(out.hidden_states[row][0, len(pids) :].float().norm(dim=-1).sum())
        cnt += len(rids)
        if i % 50 == 0:
            print(i, {row: round(sums[row] / max(cnt, 1), 2) for row in rows}, flush=True)
norms = {row: sums[row] / cnt for row in rows}
stored = json.load(open("../exp_trigger_prompts/work/steer/norms-Qwen_Qwen3-8B.json"))["norms"]
print(
    "mean response-token norm:",
    {row: round(v, 2) for row, v in norms.items()},
    "| stored row 22:",
    stored.get("22"),
    f"| tokens {cnt}",
)
out_path = Path("../exp_trigger_prompts/work/steer/norms-Qwen_Qwen3-8B-late.json")
json.dump(
    {
        "norms": {"28": norms[28], "32": norms[32]},
        "check_22": norms[22],
        "stored_22": stored.get("22"),
        "tokens": cnt,
        "source": str(src),
    },
    out_path.open("w"),
    indent=1,
)
print("wrote", out_path)
