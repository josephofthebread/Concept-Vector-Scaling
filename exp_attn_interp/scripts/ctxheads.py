import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp_attn_interp.config import INSTRUCT_MODEL, ONTOLOGY

p = argparse.ArgumentParser()
p.add_argument("--texts", default="work/judge-in/pole.jsonl")
p.add_argument("--ctx", default="work/posteer-ctx/a0/orig.jsonl")
p.add_argument("--markers", type=Path, default=Path("data/markers_presence.json"))
p.add_argument("--blocks", default="22-35")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--limit", type=int, default=0)
args = p.parse_args()
b0, b1 = (int(x) for x in args.blocks.split("-"))
BLOCKS = list(range(b0, b1 + 1))
mk = json.load(args.markers.open())
texts = [json.loads(l) for l in open(args.texts)]
texts = [x for x in texts if x["arm"] == "noctx_bad" and abs(float(x["alpha"])) < 1e-6]
ctx = {(x["pair_id"], x["slot"]): x for x in map(json.loads, open(args.ctx))}
texts = [
    x
    for x in texts
    if (x["pair_id"], x["slot"]) in ctx
    and len(mk["concept"].get(str(x["pair_id"]), [])) >= 2
    and len(mk["antagonist"].get(str(x["pair_id"]), [])) >= 2
]
if args.limit:
    texts = texts[: args.limit]
done = set()
args.out.parent.mkdir(parents=True, exist_ok=True)
if args.out.exists():
    done = {json.loads(l)["key"] for l in args.out.open()}
texts = [x for x in texts if x["key"] not in done]
print(f"{len(texts)} texts ({len(done)} done); blocks {BLOCKS}")
tok = AutoTokenizer.from_pretrained(INSTRUCT_MODEL, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    INSTRUCT_MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager", local_files_only=True
).eval()
dev = model.device
cfg = model.config
NH, NKV, HD = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
GRP = NH // NKV
WuF = model.lm_head.weight.float()
gn = model.model.norm.weight.float()
eps = float(model.model.norm.variance_epsilon)
Umean = (WuF * gn[None, :]).mean(0)
cap = {}
for b in BLOCKS:

    def v_hook(_m, _i, out, b=b):
        cap[("v", b)] = out.detach()

    def attn_hook(_m, _i, out, b=b):
        cap[("attn", b)] = (out[0] if isinstance(out, tuple) else out).detach()

    model.model.layers[b].self_attn.v_proj.register_forward_hook(v_hook)
    model.model.layers[b].self_attn.register_forward_hook(attn_hook)


def final_hook(_m, _i, out):
    cap["final"] = (out[0] if isinstance(out, tuple) else out).detach()


model.model.layers[cfg.num_hidden_layers - 1].register_forward_hook(final_hook)
ont = json.loads(Path(ONTOLOGY).read_text())
items = ont["pairs"] if isinstance(ont, dict) and "pairs" in ont else ont
STOP = set(
    "the a an of and or to in on at for with is are was were be been being by as it its this that these those i you he she "
    "they we them their our your my me him her his one two either side may apply about whether while another not no but so "
    "if than then there here when which who whom what how do does did have has had someone person people".split()
)
sit_ids = {}
for it in items:
    ids = [
        int(t)
        for t in tok(it["narrative_guidance"], add_special_tokens=False)["input_ids"]
        if str(tok.decode([t])).strip().lower().strip('".,:;()') not in STOP and str(tok.decode([t])).strip().isalpha()
    ]
    sit_ids[int(it["pair_id"])] = sorted(set(ids))
sdir = {}
for k in {int(x["pair_id"]) for x in texts}:
    S = torch.tensor(sit_ids[k], device=dev)
    sdir[k] = WuF[S].mean(0) * gn - Umean
Wo = {b: model.model.layers[b].self_attn.o_proj.weight.float() for b in BLOCKS}
CLASSES = ("sys_sit", "sys_other", "template", "user", "response")


def run(msgs, rids, k, with_sys):
    prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(prefix, add_special_tokens=False, return_offsets_mapping=True)
    pids, offs = enc["input_ids"], enc["offset_mapping"]
    pcls = ["template"] * len(pids)
    pos = 0
    for m in msgs:
        start = prefix.find(m["content"], pos)
        end = start + len(m["content"])
        pos = end
        for i, (a, b) in enumerate(offs):
            if a >= start and b <= end and b > a:
                pcls[i] = "system" if m["role"] == "system" else "user"
    ids = torch.tensor([pids + rids], device=dev)
    P = len(pids)
    N = ids.shape[1]
    cap.clear()
    out = model(input_ids=ids, output_attentions=True, use_cache=False)
    sl = slice(P, N - 1)
    hf = cap["final"][0, sl].float()
    rms = torch.sqrt((hf * hf).mean(-1) + eps)
    src_cls = torch.zeros(N, dtype=torch.long, device=dev)
    S = set(sit_ids[k])
    for j in range(N):
        if j >= P:
            src_cls[j] = 4
        elif pcls[j] == "system":
            src_cls[j] = 0 if pids[j] in S else 1
        elif pcls[j] == "user":
            src_cls[j] = 3
        else:
            src_cls[j] = 2
    s = sdir[k]
    res = {}
    for b in BLOCKS:
        A = out.attentions[b][0].float()
        Vv = cap[("v", b)][0].float().view(N, NKV, HD)
        heads = []
        for h in range(NH):
            wo_s = Wo[b][:, h * HD : (h + 1) * HD].T @ s
            c = Vv[:, h // GRP, :] @ wo_s
            contrib = A[h, sl, :] * c[None, :]
            per_pos = contrib / rms[:, None]
            by_cls = [float(per_pos[:, src_cls == ci].sum(-1).mean()) for ci in range(5)]
            heads.append(by_cls)
        tot = (cap[("attn", b)][0, sl].float() @ s / rms).mean()
        res[str(b)] = {"heads": heads, "sublayer": float(tot), "heads_sum": float(sum(sum(hc) for hc in heads))}
    counts = [int((src_cls == ci).sum()) for ci in range(5)]
    return res, counts


t0 = time.monotonic()
checked = False
with torch.inference_mode(), args.out.open("a") as sink:
    for i, x in enumerate(texts):
        k = int(x["pair_id"])
        rids = tok(x["response"], add_special_tokens=False)["input_ids"]
        if len(rids) < 2:
            continue
        cx = ctx[(k, x["slot"])]
        on, counts_on = run(
            [{"role": "system", "content": cx["system"]}, {"role": "user", "content": x["prompt"]}], rids, k, True
        )
        off, counts_off = run([{"role": "user", "content": x["prompt"]}], rids, k, False)
        if not checked:
            for b in ("33", "34"):
                print(
                    f"  check block {b} (CTX): heads sum {on[b]['heads_sum']:+.4f} vs sublayer {on[b]['sublayer']:+.4f}"
                )
            print(f"  source counts CTX {dict(zip(CLASSES, counts_on))}, OFF {dict(zip(CLASSES, counts_off))}")
            checked = True
        sink.write(
            json.dumps(
                {
                    "key": x["key"],
                    "pair_id": k,
                    "slot": x["slot"],
                    "counts_ctx": counts_on,
                    "counts_off": counts_off,
                    "ctx": on,
                    "off": off,
                }
            )
            + "\n"
        )
        if i % 40 == 0:
            sink.flush()
            print(f"{i}/{len(texts)} {(time.monotonic() - t0) / 60:.1f} min", flush=True)
print("done", len(texts), f"{(time.monotonic() - t0) / 60:.1f} min")
