import glob
import json

import numpy as np

FEATURES = "../exp_lmsys_probe/data/features.json"
KMAX = 12
WANT = [
    ("risk-averse self-restraint", "meanthink", 24),
    ("conflict-avoidance", "assistant", 22),
    ("sycophancy", "first", 16),
    ("excessive validation", "first", 16),
    ("reports internal states honestly", "first", 16),
    ("asking clarifying questions", "meanthink", 24),
    ("irreversible harm", "toolcall", 20),
    ("confirmation bias", "meanthink", 24),
]

feats = json.loads(open(FEATURES).read())["features"]
feats.sort(key=lambda f: f["pair_id"])


def find(prefix):
    for f in feats:
        if f["concept"].startswith(prefix):
            return f["pair_id"], f["concept"]
    raise KeyError(prefix)


files = sorted(glob.glob("turnwork/*.npz"))
e0 = np.load(files[0], allow_pickle=True)
POS = [str(x) for x in e0["pos"]]
LAYERS = [int(x) for x in e0["layers"]]
targets = [(find(c)[0], find(c)[1], POS.index(p), LAYERS.index(l)) for c, p, l in WANT]

acc = {
    g: {(pid): {"s": np.zeros(KMAX), "ss": np.zeros(KMAX), "n": np.zeros(KMAX)} for pid, _, _, _ in targets}
    for g in ("imp", "pos")
}
for f in files:
    d = np.load(f, allow_pickle=True)
    z = d["z"].astype(np.float32)
    T = min(z.shape[0], KMAX)
    g = "imp" if bool(d["impossible"]) else "pos"
    for pid, _, pi, li in targets:
        v = z[:T, pi, li, pid]
        acc[g][pid]["s"][:T] += v
        acc[g][pid]["ss"][:T] += v**2
        acc[g][pid]["n"][:T] += 1

out = []
for pid, name, pi, li in targets:
    rec = {"concept": name, "position": POS[pi], "layer": LAYERS[li]}
    for g in ("imp", "pos"):
        a = acc[g][pid]
        n = a["n"]
        m = np.divide(a["s"], n, out=np.full(KMAX, np.nan), where=n > 0)
        sd = np.sqrt(np.clip(np.divide(a["ss"], n, out=np.zeros(KMAX), where=n > 0) - np.nan_to_num(m) ** 2, 0, None))
        sem = np.divide(sd, np.sqrt(n), out=np.zeros(KMAX), where=n > 0)
        rec[g] = {
            "mean": [None if np.isnan(x) else round(float(x), 4) for x in m],
            "sem": [round(float(x), 4) for x in sem],
            "n": [int(x) for x in n],
        }
    out.append(rec)
open("turnwork/curves.json", "w").write(json.dumps({"kmax": KMAX, "curves": out}))
print("wrote turnwork/curves.json for", len(out), "concepts")
