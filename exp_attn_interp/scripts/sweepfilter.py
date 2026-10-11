import collections
import json
import sys

out = sys.argv[1]
opt = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
row = int(opt.get("row", 22))
alphas = {float(x) for x in opt.get("alphas", "0.5,-0.5").split(",")}
every = int(opt.get("every", 3))
nprompt = int(opt.get("prompts", 8))
keep = set(range(0, 1036, every))
src = "../exp_trigger_prompts/work/steer/sweep-Qwen_Qwen3-8B.jsonl"
n_in = n_out = 0
seen = collections.Counter()
with open(out, "w") as sink:
    for line in open(src):
        n_in += 1
        r = json.loads(line)
        k = int(r["pair_id"])
        if k not in keep:
            continue
        slot = int(r["slot"])
        if slot >= nprompt:
            continue
        lay = r.get("layer")
        a = float(r["alpha"])
        base = lay in (None, "None") and abs(a) < 1e-9
        if not base and (lay in (None, "None") or int(lay) != row or a not in alphas):
            continue
        rec = {
            "key": f"{k}:{slot}:L{row if not base else 0}:a{a:g}",
            "pair_id": k,
            "slot": slot,
            "layer": row if not base else row,
            "alpha": 0.0 if base else a,
            "arm": "paper",
            "prompt": r["prompt"],
            "response": r["response"],
            "system": "",
            "context": "",
            "sub": "",
            "ablate": "",
        }
        sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_out += 1
        seen[(k, "base" if base else a)] += 1
        if n_in % 2000000 == 0:
            print(f"{n_in / 1e6:.1f}M read, {n_out} kept", flush=True)
print(
    f"read {n_in} records, kept {n_out}; concepts {len({k for k, _ in seen})}, cells per concept {collections.Counter(len({a for kk, a in seen if kk == k}) for k in {kk for kk, _ in seen})}"
)
