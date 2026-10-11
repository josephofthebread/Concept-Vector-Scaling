import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp_attn_interp.config import INSTRUCT_MODEL, ONTOLOGY, vector_tag_for
from exp_attn_interp.vectors import load_diff_vectors

p = argparse.ArgumentParser()
p.add_argument("--texts", default="work/judge-in/pole.jsonl")
p.add_argument("--ctx", default="work/posteer-ctx/a0/orig.jsonl")
p.add_argument("--markers", type=Path, default=Path("data/markers_presence.json"))
p.add_argument("--layer", type=int, default=22)
p.add_argument("--alpha", type=float, default=0.5)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--limit", type=int, default=0)
args = p.parse_args()
L = args.layer
mk = json.load(args.markers.open())
texts = [json.loads(l) for l in open(args.texts)]
texts = [x for x in texts if x["arm"] == "noctx_bad" and abs(float(x["alpha"])) < 1e-6]
ctx = {(x["pair_id"], x["slot"]): x for x in map(json.loads, open(args.ctx))}
norm = float(next(iter(ctx.values()))["norm"])
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
print(f"{len(texts)} texts ({len(done)} done); row {L}, norm {norm:.2f}, alpha {args.alpha}")
tok = AutoTokenizer.from_pretrained(INSTRUCT_MODEL, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    INSTRUCT_MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa", local_files_only=True
).eval()
dev = model.device
V = load_diff_vectors(device="cpu", tag=vector_tag_for(INSTRUCT_MODEL))
WuF = model.lm_head.weight.float()
gn = model.model.norm.weight.float()
eps = float(model.model.norm.variance_epsilon)
Ug = WuF * gn[None, :]
Umean = Ug.mean(0)
NB = len(model.model.layers)
cap = {}
for b in range(NB):

    def attn_hook(_m, _i, out, b=b):
        cap[("attn", b)] = (out[0] if isinstance(out, tuple) else out).detach()

    def mlp_hook(_m, _i, out, b=b):
        cap[("mlp", b)] = out.detach()

    model.model.layers[b].self_attn.register_forward_hook(attn_hook)
    model.model.layers[b].mlp.register_forward_hook(mlp_hook)


def final_hook(_m, _i, out):
    cap["final"] = (out[0] if isinstance(out, tuple) else out).detach()


model.model.layers[NB - 1].register_forward_hook(final_hook)


class Inject:
    def __init__(self, L, delta, start):
        self.L, self.delta, self.start, self.h = L, delta, start, None

    def __enter__(self):
        def hook(_m, _i, out):
            res = out[0] if isinstance(out, tuple) else out
            res = res.clone()
            res[:, self.start :, :] += self.delta.to(res.dtype)
            return (res, *out[1:]) if isinstance(out, tuple) else res

        self.h = model.model.layers[self.L - 1].register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        self.h.remove()


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
items_by_pair = {int(it["pair_id"]): it for it in items}
sit_set = {k: set(v) for k, v in sit_ids.items()}
unit = lambda x: x / x.norm()
pairs = {int(x["pair_id"]) for x in texts}
sets, dirs, vecs = {}, {}, {}
for k in pairs:
    C = torch.tensor(mk["concept"][str(k)], device=dev)
    A = torch.tensor(mk["antagonist"][str(k)], device=dev)
    S = torch.tensor(sit_ids[k], device=dev)
    F = torch.tensor(sorted(set(mk["concept"][str(k)]) | set(mk["antagonist"][str(k)])), device=dev)
    sets[k] = {"C": C, "A": A, "S": S, "F": F}
    g = (WuF[C].mean(0) - WuF[A].mean(0)) * gn
    s = WuF[S].mean(0) * gn - Umean
    f = WuF[F].mean(0) * gn - Umean
    dirs[k] = torch.stack([g, s, f], 1)
    v = unit(V[L][k].to(dev).float())
    sk = unit(Ug[S].mean(0))
    vecs[k] = {"V": v, "SIT": sk, "SIT1": unit(v + sk)}
NAMES = ("g", "s", "f")
ARMS = ("OFF", "CTX", "SCR", "BAG")
import numpy as np


def prefix_ids(msgs):
    """Return the chat-prefix token ids and the token span of the system content."""
    prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(prefix, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    span = None
    for m in msgs:
        if m["role"] == "system":
            a = prefix.find(m["content"])
            b = a + len(m["content"])
            idx = [i for i, (x, y) in enumerate(offs) if x >= a and y <= b and y > x]
            span = (idx[0], idx[-1] + 1)
    return ids, span


def run(pids, rids, k, vec=None):
    ids = torch.tensor([pids + rids], device=dev)
    P = len(pids)
    N = ids.shape[1]
    delta = None
    cap.clear()
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    sl = slice(P, N - 1)
    D = dirs[k]
    h = cap["final"][0, sl].float()
    rms = torch.sqrt((h * h).mean(-1) + eps)
    lp = torch.log_softmax(((h / rms[:, None]) * gn) @ WuF.T, -1)
    proj = {"E": out.hidden_states[0][0, sl].float() @ D}
    for b in range(NB):
        proj[f"attn{b}"] = cap[("attn", b)][0, sl].float() @ D
        proj[f"mlp{b}"] = cap[("mlp", b)][0, sl].float() @ D
    return {
        "h": h,
        "rms": rms,
        "lp": lp,
        "proj": proj,
        "delta": delta,
        "dproj": (delta @ D) if delta is not None else torch.zeros(3, device=dev),
        "T": N - 1 - P,
    }


t0 = time.monotonic()
checked = False
with torch.inference_mode(), args.out.open("a") as sink:
    for i, x in enumerate(texts):
        k = int(x["pair_id"])
        st = sets[k]
        rids = tok(x["response"], add_special_tokens=False)["input_ids"]
        if len(rids) < 2:
            continue
        cx = ctx[(k, x["slot"])]
        base_msgs = [{"role": "user", "content": x["prompt"]}]
        ctx_msgs = [{"role": "system", "content": cx["system"]}, {"role": "user", "content": x["prompt"]}]
        off_ids, _ = prefix_ids(base_msgs)
        ctx_ids, (a, b) = prefix_ids(ctx_msgs)
        rng = np.random.RandomState(int(k) * 1000 + int(x["slot"]))
        scr_ids = ctx_ids[:a] + [ctx_ids[j] for j in rng.permutation(np.arange(a, b))] + ctx_ids[b:]
        guid = tok(items_by_pair[k]["narrative_guidance"], add_special_tokens=False)["input_ids"]
        bag, seen = [], set()
        for t in guid:
            if t in sit_set[k] and t not in seen:
                bag.append(int(t))
                seen.add(t)
        bag_ids = ctx_ids[:a] + bag + ctx_ids[b:]
        passes = {
            "OFF": run(off_ids, rids, k),
            "CTX": run(ctx_ids, rids, k),
            "SCR": run(scr_ids, rids, k),
            "BAG": run(bag_ids, rids, k),
        }
        nxt = torch.tensor(rids[1:], device=dev)
        rec = {
            "key": x["key"],
            "pair_id": k,
            "slot": x["slot"],
            "n_pos": passes["OFF"]["T"],
            "n_sys": int(b - a),
            "n_bag": len(bag),
            "arms": {},
        }
        for arm in ("CTX", "SCR", "BAG"):
            on, off = passes[arm], passes["OFF"]
            D = dirs[k]
            T = on["T"]
            res = {"readouts": {}}
            for j, nm in enumerate(NAMES):
                Gon = (on["h"] @ D[:, j]) / on["rms"]
                Goff = (off["h"] @ D[:, j]) / off["rms"]
                terms = {
                    c: float(((on["proj"][c][:, j] - off["proj"][c][:, j]) / on["rms"]).mean()) for c in on["proj"]
                }
                terms["delta"] = float(((on["dproj"][j] - off["dproj"][j]) / on["rms"]).mean())
                terms["dilution"] = float(((off["h"] @ D[:, j]) * (1.0 / on["rms"] - 1.0 / off["rms"])).mean())
                res["readouts"][nm] = {
                    "dG": float((Gon - Goff).mean()),
                    "sum": float(sum(terms.values())),
                    "G_off": float(Goff.mean()),
                    "terms": terms,
                }
            mass = lambda pas, idx: torch.logsumexp(pas["lp"][:, idx], -1)
            ar = torch.arange(T, device=dev)
            dl = on["lp"][ar, nxt] - off["lp"][ar, nxt]
            is_marker = torch.isin(nxt, st["F"])
            is_sit = torch.isin(nxt, st["S"]) & ~is_marker
            is_other = ~is_marker & ~is_sit
            res["soft"] = {
                "d_mass_sit": float((mass(on, st["S"]) - mass(off, st["S"])).mean()),
                "d_mass_C": float((mass(on, st["C"]) - mass(off, st["C"])).mean()),
                "d_mass_A": float((mass(on, st["A"]) - mass(off, st["A"])).mean()),
                "mass_sit_off": float(mass(off, st["S"]).mean()),
                "kl": float((on["lp"].exp() * (on["lp"] - off["lp"])).sum(-1).mean()),
                "d_lp_next": float(dl.mean()),
                "by_class": {
                    cn: [float(dl[m].mean()) if int(m.sum()) else 0.0, int(m.sum())]
                    for cn, m in (("marker", is_marker), ("sit", is_sit), ("other", is_other))
                },
            }
            res["rms"] = {
                "on": float(on["rms"].mean()),
                "off": float(off["rms"].mean()),
                "ratio": float((on["rms"] / off["rms"]).mean()),
            }
            res["dh_norm"] = float((on["h"] - off["h"]).norm(dim=-1).mean())
            rec["arms"][arm] = res
        if not checked:
            for arm in ("CTX", "SCR", "BAG"):
                r = rec["arms"][arm]["readouts"]["s"]
                print(f"  check {arm} vs OFF along s: dG {r['dG']:+.4f}, Σ terms {r['sum']:+.4f}")
            print(f"  system span {a}:{b} ({b - a} tokens), bag {len(bag)} tokens: {tok.decode(bag)[:200]!r}")
            checked = True
        sink.write(json.dumps(rec) + "\n")
        if i % 40 == 0:
            sink.flush()
            print(f"{i}/{len(texts)} {(time.monotonic() - t0) / 60:.1f} min", flush=True)
print("done", len(texts), f"{(time.monotonic() - t0) / 60:.1f} min")
