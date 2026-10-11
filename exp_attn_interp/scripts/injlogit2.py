import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp_attn_interp.config import INSTRUCT_MODEL, ONTOLOGY, vector_tag_for
from exp_attn_interp.vectors import load_diff_vectors

p = argparse.ArgumentParser()
p.add_argument("--records", default="work/posteer-foot/orig.jsonl")
p.add_argument("--markers", type=Path, default=Path("data/markers_presence.json"))
p.add_argument("--inj", default="22,28")
p.add_argument("--alpha", type=float, default=0.5)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--limit", type=int, default=0)
args = p.parse_args()
inj_rows = sorted({int(x) for x in args.inj.split(",")})
mk = json.load(args.markers.open())
allrecs = [json.loads(l) for l in open(args.records)]
has_mk = lambda k: len(mk["concept"].get(str(k), [])) >= 2 and len(mk["antagonist"].get(str(k), [])) >= 2
own = [
    x
    for x in allrecs
    if int(x["layer"]) in inj_rows and abs(abs(float(x["alpha"])) - args.alpha) < 1e-9 and has_mk(x["pair_id"])
]
same = [x for x in allrecs if abs(float(x["alpha"])) < 1e-6 and has_mk(x["pair_id"])]
norm_by_row = {}
for x in allrecs:
    if abs(abs(float(x["alpha"])) - args.alpha) < 1e-9:
        norm_by_row.setdefault(int(x["layer"]), float(x["norm"]))
if args.limit:
    own = own[: args.limit]
    same = same[: max(1, args.limit // 4)]
units = [("own", x, [(int(x["layer"]), float(x["alpha"]))]) for x in own]
units += [("same", x, [(L, s * args.alpha) for L in inj_rows for s in (1.0, -1.0)]) for x in same]
rkey = lambda d, x, L, a: f"{d}:{x['key']}:{L}:{a:+.2f}"
done = set()
args.out.parent.mkdir(parents=True, exist_ok=True)
if args.out.exists():
    done = {json.loads(l)["key"] for l in args.out.open()}
units = [(d, x, [(L, a) for L, a in la if rkey(d, x, L, a) not in done]) for d, x, la in units]
units = [u for u in units if u[2]]
print(
    f"{len(units)} texts, {sum(len(u[2]) for u in units)} injections to read ({len(done)} done); rows {inj_rows}, norms {norm_by_row}"
)
tok = AutoTokenizer.from_pretrained(INSTRUCT_MODEL, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    INSTRUCT_MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa", local_files_only=True
).eval()
dev = model.device
V = load_diff_vectors(device="cpu", tag=vector_tag_for(INSTRUCT_MODEL))
Wu = model.lm_head.weight
WuF = Wu.float()
gn = model.model.norm.weight.float()
eps = float(model.model.norm.variance_epsilon)
Umean = WuF.mean(0) * gn
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
pairs = {int(u[1]["pair_id"]) for u in units}
sets, dirs = {}, {}
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
print(f"situation words per pair: mean {sum(len(sit_ids[k]) for k in pairs) / len(pairs):.1f} tokens")
NAMES = ("g", "s", "f")


def run(ids, P, N, L=None, delta=None, D=None):
    cap.clear()
    if delta is not None:
        with Inject(L, delta, P - 1):
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    else:
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    sl = slice(P, N - 1)
    h = cap["final"][0, sl].float()
    rms = torch.sqrt((h * h).mean(-1) + eps)
    logits = ((h / rms[:, None]) * gn) @ WuF.T
    lp = torch.log_softmax(logits, -1)
    proj = {"E": out.hidden_states[0][0, sl].float() @ D}
    for b in range(NB):
        proj[f"attn{b}"] = cap[("attn", b)][0, sl].float() @ D
        proj[f"mlp{b}"] = cap[("mlp", b)][0, sl].float() @ D
    return {"h": h, "rms": rms, "lp": lp, "proj": proj, "ref": out.logits[0, sl].float(), "logits": logits}


t0 = time.monotonic()
checked = False
n_written = 0
with torch.inference_mode(), args.out.open("a") as sink:
    for ui, (design, rec, la) in enumerate(units):
        k = int(rec["pair_id"])
        msgs = ([{"role": "system", "content": rec["system"]}] if rec.get("system") else []) + [
            {"role": "user", "content": rec["prompt"]}
        ]
        prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        pids = tok(prefix, add_special_tokens=False)["input_ids"]
        rids = tok(rec["response"], add_special_tokens=False)["input_ids"]
        if len(rids) < 2:
            continue
        ids = torch.tensor([pids + rids], device=dev)
        P = len(pids)
        N = ids.shape[1]
        nxt = ids[0, P + 1 :]
        D = dirs[k]
        st = sets[k]
        is_marker = torch.isin(nxt, st["F"])
        is_sit = torch.isin(nxt, st["S"]) & ~is_marker
        is_other = ~is_marker & ~is_sit
        off = run(ids, P, N, D=D)
        if not checked:
            print(
                f"  logits check (off): max |mine − model| / max |model| = {float((off['logits'] - off['ref']).abs().max() / off['ref'].abs().max()):.2e}"
            )
        for L, a in la:
            norm = float(rec["norm"]) if design == "own" else norm_by_row[L]
            delta = ((a * norm) * (V[L][k] / V[L][k].norm())).to(dev)
            on = run(ids, P, N, L, delta, D)
            sgn = torch.tensor([1.0 if a > 0 else -1.0, 1.0, 1.0], device=dev)
            dproj = delta.float() @ D
            res = {
                "key": rkey(design, rec, L, a),
                "design": design,
                "pair_id": k,
                "slot": rec["slot"],
                "layer": L,
                "alpha": a,
                "n_pos": int(N - 1 - P),
                "readouts": {},
            }
            for j, nm in enumerate(NAMES):
                Gon = (on["h"] @ D[:, j]) * sgn[j] / on["rms"]
                Goff = (off["h"] @ D[:, j]) * sgn[j] / off["rms"]
                terms = {
                    c: float(((on["proj"][c][:, j] - off["proj"][c][:, j]) * sgn[j] / on["rms"]).mean())
                    for c in on["proj"]
                }
                terms["delta"] = float((dproj[j] * sgn[j] / on["rms"]).mean())
                terms["dilution"] = float(((off["h"] @ D[:, j]) * sgn[j] * (1.0 / on["rms"] - 1.0 / off["rms"])).mean())
                res["readouts"][nm] = {
                    "dG": float((Gon - Goff).mean()),
                    "sum": float(sum(terms.values())),
                    "G_off": float(Goff.mean()),
                    "terms": terms,
                }
                if not checked:
                    tot = sum(on["proj"][c][:, j] for c in on["proj"]) + dproj[j]
                    print(
                        f"  residual check ({nm}, on): |h·d − Σ| / |h·d| = {float(((on['h'] @ D[:, j]) - tot).norm() / (on['h'] @ D[:, j]).norm()):.2e}"
                    )
            checked = True
            pushed, other = (st["C"], st["A"]) if a > 0 else (st["A"], st["C"])
            mass = lambda pas, idx: torch.logsumexp(pas["lp"][:, idx], -1)
            lpn_on = on["lp"][torch.arange(N - 1 - P, device=dev), nxt]
            lpn_off = off["lp"][torch.arange(N - 1 - P, device=dev), nxt]
            dl = lpn_on - lpn_off
            bycls = {}
            for cn, m in (("marker", is_marker), ("sit", is_sit), ("other", is_other)):
                bycls[cn] = [float(dl[m].mean()) if int(m.sum()) else 0.0, int(m.sum())]
            res["soft"] = {
                "d_mass_pushed": float((mass(on, pushed) - mass(off, pushed)).mean()),
                "d_mass_other": float((mass(on, other) - mass(off, other)).mean()),
                "d_mass_sit": float((mass(on, st["S"]) - mass(off, st["S"])).mean()),
                "d_mass_field": float((mass(on, st["F"]) - mass(off, st["F"])).mean()),
                "mass_pushed_off": float(mass(off, pushed).mean()),
                "mass_sit_off": float(mass(off, st["S"]).mean()),
                "mass_field_off": float(mass(off, st["F"]).mean()),
                "kl": float((on["lp"].exp() * (on["lp"] - off["lp"])).sum(-1).mean()),
                "d_lp_next": float(dl.mean()),
                "d_lp_next_by_class": bycls,
            }
            res["rms"] = {
                "on": float(on["rms"].mean()),
                "off": float(off["rms"].mean()),
                "ratio": float((on["rms"] / off["rms"]).mean()),
            }
            res["dh_norm"] = float((on["h"] - off["h"]).norm(dim=-1).mean())
            res["delta_norm"] = float(delta.norm())
            res["dir_norms"] = {nm: float(D[:, j].norm()) for j, nm in enumerate(NAMES)}
            sink.write(json.dumps(res) + "\n")
            n_written += 1
        if ui % 50 == 0:
            sink.flush()
            print(f"{ui}/{len(units)} texts, {n_written} records, {(time.monotonic() - t0) / 60:.1f} min", flush=True)
print("done", n_written, f"{(time.monotonic() - t0) / 60:.1f} min")
