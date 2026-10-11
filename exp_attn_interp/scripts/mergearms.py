import json
import sys

out = sys.argv[1]
n = 0
labels = []
ALPHAS = None
DUPNEG = set()
for a in sys.argv[2:]:
    if a.startswith("--alphas="):
        ALPHAS = {float(x) for x in a.split("=")[1].split(",")}
    if a.startswith("--dup-neg="):
        DUPNEG = set(a.split("=")[1].split(","))
with open(out, "w") as sink:
    for i, spec in enumerate([a for a in sys.argv[2:] if not a.startswith("--")]):
        label, path = spec.split("=", 1)
        labels.append(label)
        for line in open(path):
            r = json.loads(line)
            if ALPHAS is not None and float(r["alpha"]) not in ALPHAS:
                continue
            r["slot"] = int(r["slot"]) + 1000 * i
            r["arm_label"] = label
            if float(r["alpha"]) == 0.0:
                for a in (1e-6, -1e-6):
                    rr = dict(r)
                    rr["alpha"] = a
                    sink.write(json.dumps(rr, ensure_ascii=False) + "\n")
                    n += 1
            else:
                sink.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
                if label in DUPNEG:
                    rr = dict(r)
                    rr["alpha"] = -float(r["alpha"])
                    rr["judged_other_pole"] = True
                    sink.write(json.dumps(rr, ensure_ascii=False) + "\n")
                    n += 1
json.dump(labels, open(out + ".labels.json", "w"))
print("wrote", out, n, "records; arms", labels)
