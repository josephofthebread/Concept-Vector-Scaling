from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("span_judge")

DATASETS = {"lmsys": Path("work"), "owt": Path("work_owt")}
OUT = Path("work/span_pilot")
K, CTX = 5, 3
CONTROL_EVERY = 10
N_COVERED = 4

TRAIT = [
    "verbosity",
    "conscientiousness",
    "awe",
    "humor",
    "warmth",
    "formality",
    "optimism",
    "joy",
    "polite",
    "sarcasm",
]
TOPICAL = [
    "climate activism",
    "planning the approach before calculating",
    "methodical tester",
    "causal understanding",
    "honoring higher-priority principals",
    "preference-consistent across oversight",
]

_OPEN = {
    "chat": "You annotate a real conversation between a user and an AI assistant, shown as a User turn and "
    "an Assistant turn; the Assistant turn is the AI's own output. An automatic concept detector's "
    "strongest firings in it are marked. ",
    "doc": "You annotate a passage of real web text. An automatic concept detector's strongest firings in "
    "it are marked. ",
}
_MARKS = (
    "Each marked span is one peak token plus up to 3 context tokens on each side, so the middle of a "
    "span is where the detector fired and its edges are context; adjacent spans may be merged into one. "
    "Spans wrapped in << >> were marked for the CONCEPT; spans wrapped in [[ ]] were marked for the "
    "ANTAGONIST. "
)
_POLES = {
    "full": ("The two poles and their DESCRIPTION are given.", "in the sense fixed by the DESCRIPTION"),
    "none": ("The two poles are given.", "in the ordinary sense of the pole's name"),
    "example": (
        "The two poles are given, with an EXAMPLE CONTRAST that illustrates the pair with one "
        "typical case; the poles themselves are the general concepts named, not only that case.",
        "in the ordinary sense of the pole's name",
    ),
}
_SCALE = {
    "strict": "score whether the span's wording or meaning matches the pole it was marked for: 2 = it "
    "does; 1 = unsure; 0 = it does not (generic wording or punctuation, unrelated content, or "
    "the opposite pole).",
    "lenient": "score whether the span's wording or meaning is consistent with the pole it was marked "
    "for, read broadly and giving the benefit of the doubt: 2 = it plausibly expresses that "
    "pole, even partially, implicitly, or as a minor aspect; 1 = weak or ambiguous; 0 = "
    "unrelated, generic wording or punctuation, or the opposite pole.",
}
_TASK = (
    "\nFor each marked span, in order of appearance, {scale}\n"
    "Score what the span's wording expresses or enacts, {sense}; the {unit} does not need to discuss or "
    "name the pole. Judge in any language.\n"
    "Answer with only the tag <ANS>s1,s2,...,sN</ANS>, one score per span in order (N = the number of "
    "marked spans)."
)


def system_for(kind: str, desc_mode: str, scale: str = "strict") -> str:
    poles, sense = _POLES[desc_mode]
    return (
        _OPEN[kind]
        + _MARKS
        + poles
        + _TASK.format(scale=_SCALE[scale], sense=sense, unit="conversation" if kind == "chat" else "passage")
    )


ANS_RE = re.compile(r"<ANS>\s*([0-2](?:\s*,\s*[0-2])*)\s*</ANS>", re.IGNORECASE)


def token_poles(cos: list[float], k: int = K, ctx: int = CTX) -> list[str]:
    """Identical to suppl/build.py spans_from_cos: top-k |cos| peaks, +/-ctx, nearest peak's sign."""
    n = len(cos)
    if not n:
        return []
    peaks = sorted(range(n), key=lambda i: -abs(cos[i]))[: min(k, n)]
    pole = [""] * n
    for i in range(n):
        near, nd = None, ctx + 1
        for p in peaks:
            if abs(i - p) < nd:
                nd, near = abs(i - p), p
        if near is not None and nd <= ctx:
            pole[i] = "pos" if cos[near] > 0 else "neg"
    return pole


def false_poles(cos: list[float], real: list[str], rng: random.Random) -> list[str] | None:
    """Same number of peaks with the same sign pattern, moved to random unmarked positions."""
    n = len(cos)
    free = [i for i in range(n) if not real[i]]
    if len(free) < K:
        return None
    signs = [1.0 if cos[p] > 0 else -1.0 for p in sorted(range(n), key=lambda i: -abs(cos[i]))[:K]]
    rng.shuffle(signs)
    fake = [0.0] * n
    for p, s in zip(rng.sample(free, K), signs):
        fake[p] = s
    return token_poles(fake)


def render(toks: list[str], pole: list[str]) -> tuple[str, list[dict]]:
    """Wrap contiguous same-pole runs (the page's <mark> runs) in << >> / [[ ]]; return spans in order."""
    out, spans, i, n = [], [], 0, len(toks)
    while i < n:
        j = i
        while j < n and pole[j] == pole[i]:
            j += 1
        seg = "".join(toks[i:j])
        core = seg.strip()
        if pole[i] and core:
            lead = seg[: len(seg) - len(seg.lstrip())]
            trail = seg[len(seg.rstrip()) :]
            wrapped = f"<<{core}>>" if pole[i] == "pos" else f"[[{core}]]"
            out.append(f"{lead}{wrapped}{trail}")
            spans.append({"text": wrapped, "pole": pole[i]})
        else:
            out.append(seg)
        i = j
    return "".join(out).strip(), spans


def build_prompt(ft: dict, ex: dict, pole: list[str], desc_mode: str = "full") -> tuple[str, list[dict]]:
    toks, ncut = ex["toks"], ex["ncut"]
    if ncut:
        mu, su = render(toks[:ncut], pole[:ncut])
        ma, sa = render(toks[ncut:], pole[ncut:])
        body, spans = f"CONVERSATION:\nUser: {mu}\nAssistant: {ma}\n\n", su + sa
    else:
        m, spans = render(toks, pole)
        body = f"TEXT:\n{m}\n\n"
    n = len(spans)
    listing = "\n".join(f"{i + 1}. {s['text']}" for i, s in enumerate(spans))
    guidance = {
        "full": f"DESCRIPTION: {ft['narrative_guidance']}\n",
        "none": "",
        "example": f"EXAMPLE CONTRAST: {ft['narrative_guidance']}\n",
    }[desc_mode]
    content = (
        f"CONCEPT (marked << >>): {ft['concept']}\nANTAGONIST (marked [[ ]]): {ft['antagonist']}\n"
        f"CLASS: {ft['class_name']}\n{guidance}\n{body}"
        f"MARKED SPANS in order of appearance ({n}):\n{listing}\n\n"
        f"Score each of the {n} spans 0/1/2 for the pole it was marked for."
    )
    return content, spans


def pick_pairs(features: list[dict], accepted_count: dict[int, int]) -> dict[int, str]:
    chosen: dict[int, str] = {}

    def find(name: str) -> dict | None:
        exact = [f for f in features if f["concept"].lower() == name]
        sub = exact or [f for f in features if name in f["concept"].lower()]
        return sub[0] if sub else None

    for group, names in (("trait", TRAIT), ("topical", TOPICAL)):
        for name in names:
            f = find(name.lower())
            if f is None:
                log.warning("no pair matches %r", name)
            elif f["pair_id"] not in chosen:
                chosen[f["pair_id"]] = "verbosity" if name == "verbosity" else group
    for pid, _ in sorted(accepted_count.items(), key=lambda x: -x[1]):
        if sum(g == "covered" for g in chosen.values()) >= N_COVERED:
            break
        if pid not in chosen:
            chosen[pid] = "covered"
    by_class: dict[str, list[int]] = defaultdict(list)
    for f in features:
        by_class[f["class_name"]].append(f["pair_id"])
    have = {f["class_name"] for f in features if f["pair_id"] in chosen}
    rng = random.Random(0)
    for cname in sorted(by_class):
        if cname not in have:
            chosen[rng.choice(sorted(by_class[cname]))] = "class"
    return chosen


def build_records(by_id: dict[int, dict], chosen: dict[int, str], desc_mode: str = "full") -> list[dict]:
    recs = []
    for ds, w in DATASETS.items():
        judged = {f["pair_id"]: f for f in json.loads((w / "judged.json").read_text())["features"]}
        for fname, plain in (("highlights.json", "yes"), ("highlights_rej.json", "no")):
            for f in json.loads((w / fname).read_text())["features"]:
                pid = f["pair_id"]
                if pid not in chosen:
                    continue
                ft = by_id[pid]
                for ex in f["examples"]:
                    reasons = {c["conversation_id"]: c.get("reason", "") for c in judged[pid][ex["pole"]]}
                    pole = token_poles(ex["cos"])
                    content, spans = build_prompt(ft, ex, pole, desc_mode)
                    if not spans:
                        continue
                    base = {
                        "pair_id": pid,
                        "concept": ft["concept"],
                        "antagonist": ft["antagonist"],
                        "cls": ft["class_name"],
                        "group": chosen[pid],
                        "dataset": ds,
                        "kind": "chat" if ex["ncut"] else "doc",
                        "conversation_id": ex["conversation_id"],
                        "pole": ex["pole"],
                        "score": ex["score"],
                        "plain": plain,
                        "plain_reason": reasons.get(ex["conversation_id"], "")
                        .split("assistantfinal")[-1]
                        .strip()[:300],
                    }
                    recs.append(base | {"control": False, "spans": spans, "content": content})
                    h = int(hashlib.md5(f"{ds}:{pid}:{ex['conversation_id']}".encode()).hexdigest(), 16)
                    if h % CONTROL_EVERY == 0:
                        fake = false_poles(ex["cos"], pole, random.Random(h))
                        if fake:
                            c2, s2 = build_prompt(ft, ex, fake, desc_mode)
                            if s2:
                                recs.append(base | {"control": True, "spans": s2, "content": c2})
    return recs


def run(recs: list[dict], efforts: list[str], args, desc_mode: str, sweep_shard: int | None = None) -> None:
    """Judge the records per effort; writes results_{effort}.json, or sweep-{shard}.json in sweep mode."""
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=args.model,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    prompts = [
        [
            {"role": "system", "content": system_for(r["kind"], desc_mode, args.scale)},
            {"role": "user", "content": r["content"]},
        ]
        for r in recs
    ]
    for effort in efforts:
        started = time.monotonic()
        replies = engine.chat(
            prompts,
            SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
            chat_template_kwargs={"reasoning_effort": effort},
        )
        secs = time.monotonic() - started
        out, ntok = [], 0
        for r, reply in zip(recs, replies):
            text = reply.outputs[0].text
            ntok += len(reply.outputs[0].token_ids)
            m = ANS_RE.findall(text)
            scores = [int(x) for x in m[-1].split(",")] if m else []
            n_spans = len(r["spans"]) if "spans" in r else len(r["span_poles"])
            if sweep_shard is None:
                out.append(
                    {"scores": scores, "ok": len(scores) == n_spans, "reason": ANS_RE.sub("", text).strip()[-600:]}
                )
            else:
                out.append(
                    {k: r[k] for k in ("pair_id", "conversation_id", "pole", "control", "span_poles")}
                    | {"scores": scores, "ok": len(scores) == n_spans}
                )
        name = f"sweep-{sweep_shard}.json" if sweep_shard is not None else f"results_{effort}.json"
        payload = (
            out
            if sweep_shard is not None
            else {"effort": effort, "secs": secs, "out_tokens": ntok, "n": len(recs), "results": out}
        )
        (OUT / name).write_text(json.dumps(payload, ensure_ascii=False))
        log.info(
            "effort=%s: %d prompts in %.0fs (%.1f/s), %.0f output tokens/prompt, parsed %d/%d -> %s",
            effort,
            len(recs),
            secs,
            len(recs) / secs,
            ntok / len(recs),
            sum(o["ok"] for o in out),
            len(out),
            name,
        )


def own_scores(r: dict, x: dict) -> list[int]:
    """Scores of the spans marked with the example's assigned pole."""
    p = "pos" if r["pole"] == "con" else "neg"
    return [s for s, sp in zip(x["scores"], r["spans"]) if sp["pole"] == p]


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def report() -> None:
    P = json.loads((OUT / "prompts.json").read_text())["records"]
    R = {}
    for f in sorted(OUT.glob("results_*.json")):
        d = json.loads(f.read_text())
        R[d["effort"]] = d
    for eff, d in R.items():
        res = d["results"]
        print(
            f"\n===== effort={eff}: {d['n']} prompts, {d['secs']:.0f}s ({d['n'] / d['secs']:.1f}/s), "
            f"{d['out_tokens'] / d['n']:.0f} out-tok/prompt, parsed {sum(x['ok'] for x in res)}/{len(res)}"
        )
        tab = defaultdict(list)
        for r, x in zip(P, res):
            if x["ok"]:
                own = own_scores(r, x)
                tab[("CONTROL" if r["control"] else r["group"], r["plain"])].append(
                    (mean(own) if own else None, 2 in own, bool(own) and max(own) == 0, mean(x["scores"]))
                )
        print(f"{'group':<11}{'plain':<6}{'n':>5}{'own-span':>10}{'>=1 own=2':>11}{'all own=0':>11}{'all-span':>10}")
        for key in sorted(tab):
            v = tab[key]
            print(
                f"{key[0]:<11}{key[1]:<6}{len(v):>5}{mean(o for o, _, _, _ in v if o is not None):>10.2f}"
                f"{100 * mean(a for _, a, _, _ in v):>10.0f}%{100 * mean(z for _, _, z, _ in v):>10.0f}%"
                f"{mean(m for _, _, _, m in v):>10.2f}"
            )
        print("\n-- judge-rejected (plain=no), real spans, per pair --")
        pp = defaultdict(list)
        for r, x in zip(P, res):
            if x["ok"] and not r["control"] and r["plain"] == "no":
                own = own_scores(r, x)
                pp[(r["group"], r["concept"])].append((2 in own, mean(own) if own else None))
        for key in sorted(pp):
            v = pp[key]
            print(
                f"  {key[0]:<10}{key[1][:44]:<45}n={len(v):>3}  >=1 own=2 {100 * mean(a for a, _ in v):>4.0f}%"
                f"  own-span {mean(o for _, o in v if o is not None):.2f}"
            )
    if "low" in R and "medium" in R:
        both = [(x, y) for x, y in zip(R["low"]["results"], R["medium"]["results"]) if x["ok"] and y["ok"]]
        diffs = [abs(a - b) for x, y in both for a, b in zip(x["scores"], y["scores"])]
        print(
            f"\n===== low vs medium (n={len(both)}): identical span scores "
            f"{100 * mean(x['scores'] == y['scores'] for x, y in both):.0f}%, mean |diff| per span "
            f"{mean(diffs):.2f}; wall x{R['medium']['secs'] / R['low']['secs']:.1f}, out-tokens x"
            f"{R['medium']['out_tokens'] / R['low']['out_tokens']:.1f}"
        )


STILL_NAME = "highlights_still_rej.json"


def export_still_rejected(effort: str) -> None:
    """Export examples the old judge rejected and no assigned-pole span scored 2,
    then build the still-rejected supplementary page from them."""
    import importlib.util

    P = json.loads((OUT / "prompts.json").read_text())["records"]
    R = json.loads((OUT / f"results_{effort}.json").read_text())["results"]
    still: set[tuple[str, int, str]] = set()
    for r, x in zip(P, R):
        if r["plain"] == "no" and not r["control"] and x["ok"] and 2 not in own_scores(r, x):
            still.add((r["dataset"], r["pair_id"], r["conversation_id"]))
    pids = sorted({r["pair_id"] for r in P})
    meta: dict[int, dict] = {}
    per_ds: dict[str, dict[int, list]] = {}
    for ds, w in DATASETS.items():
        per_ds[ds] = {pid: [] for pid in pids}
        for f in json.loads((w / "highlights_rej.json").read_text())["features"]:
            if f["pair_id"] in per_ds[ds]:
                meta.setdefault(f["pair_id"], {k: f[k] for k in ("pair_id", "concept", "antagonist", "class")})
                per_ds[ds][f["pair_id"]] = [
                    e for e in f["examples"] if (ds, f["pair_id"], e["conversation_id"]) in still
                ]
    for ds, w in list(DATASETS.items()) + [("alpaca", Path("work_alpaca"))]:
        feats = [meta[pid] | {"examples": per_ds.get(ds, {}).get(pid, [])} for pid in pids]
        (w / STILL_NAME).write_text(json.dumps({"features": feats}, ensure_ascii=False))
        log.info(
            "%s: %d still-rejected examples over %d pairs -> %s",
            ds,
            sum(len(f["examples"]) for f in feats),
            len(pids),
            w / STILL_NAME,
        )
    for pid in pids:
        log.info(
            "  %-44s lmsys %2d  owt %2d", meta[pid]["concept"][:44], len(per_ds["lmsys"][pid]), len(per_ds["owt"][pid])
        )
    spec = importlib.util.spec_from_file_location("build", Path("suppl/build.py"))
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    out = Path("suppl/concept_pairs_still_rejected.html")
    build.build(build.real_data(Path("."), STILL_NAME, "no"), out)


def sweep_records(
    hl_path: Path, by_id: dict[int, dict], desc_mode: str, shard: int, shards: int, control_every: int
) -> list[dict]:
    """Prompt records for one sweep shard, with a false-mark twin every control_every-th example."""
    d = json.loads(hl_path.read_text())
    convs = d["convs"]
    recs, idx = [], 0
    for f in d["features"]:
        ft = by_id[f["pair_id"]]
        for e in f["examples"]:
            idx += 1
            if (idx - 1) % shards != shard:
                continue
            cv = convs[e["conversation_id"]]
            ex = {"toks": cv["toks"], "ncut": cv["ncut"], "cos": e["cos"]}
            pole = token_poles(e["cos"])
            content, spans = build_prompt(ft, ex, pole, desc_mode)
            if not spans:
                continue
            base = {
                "pair_id": f["pair_id"],
                "conversation_id": e["conversation_id"],
                "pole": e["pole"],
                "kind": "chat" if cv["ncut"] else "doc",
            }
            recs.append(base | {"control": False, "span_poles": [s["pole"] for s in spans], "content": content})
            if idx % control_every == 0:
                fake = false_poles(e["cos"], pole, random.Random(idx))
                if fake:
                    c2, s2 = build_prompt(ft, ex, fake, desc_mode)
                    if s2:
                        recs.append(base | {"control": True, "span_poles": [s["pole"] for s in s2], "content": c2})
    return recs


def sweep_merge(work: Path) -> None:
    """Fold the sweep shards into span_judged.json; controls are summarized, not stored."""
    feats: dict[int, dict] = defaultdict(lambda: {"con": [], "ant": []})
    n = ok = 0
    real, fake = [], []
    for f in sorted(OUT.glob("sweep-*.json")):
        for r in json.loads(f.read_text()):
            n += 1
            ok += r["ok"]
            if not r["ok"]:
                continue
            own = [s for s, p in zip(r["scores"], r["span_poles"]) if p == ("pos" if r["pole"] == "con" else "neg")]
            (fake if r["control"] else real).append(mean(own) if own else float("nan"))
            if r["control"]:
                continue
            feats[r["pair_id"]][r["pole"]].append(
                {"conversation_id": r["conversation_id"], "scores": r["scores"], "span_poles": r["span_poles"]}
            )
    out = [{"pair_id": pid} | feats[pid] for pid in sorted(feats)]
    (OUT / "span_judged.json").write_text(json.dumps({"features": out}, ensure_ascii=False))
    acc = [
        any(s == 2 for s, p in zip(e["scores"], e["span_poles"]) if p == ("pos" if pole == "con" else "neg"))
        for f in out
        for pole in ("con", "ant")
        for e in f[pole]
    ]
    real_m = [x for x in real if x == x]
    fake_m = [x for x in fake if x == x]
    log.info(
        "merged %d records (parsed %.1f%%): %d features; own-span mean real %.2f vs false-mark %.2f; "
        "accepted (>=1 own-pole span = 2): %.1f%% of %d -> %s",
        n,
        100 * ok / max(n, 1),
        len(out),
        mean(real_m),
        mean(fake_m),
        100 * mean(acc),
        len(acc),
        OUT / "span_judged.json",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("data/features.json"))
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--efforts", default="low,medium")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--dry-run", action="store_true", help="build prompts.json only, no model")
    ap.add_argument("--report", action="store_true")
    ap.add_argument(
        "--export-still-rejected",
        action="store_true",
        help="pack old-rejected examples with no assigned-pole span scored 2 into the "
        "supplementary viewer (uses the first of --efforts)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("work/span_pilot"),
        help="pilot dir (prompts.json + results_*.json); use a new one per prompt version",
    )
    ap.add_argument(
        "--desc-mode",
        default="full",
        choices=sorted(_POLES),
        help="how the ontology guidance is shown: full (DESCRIPTION), none, example",
    )
    ap.add_argument(
        "--scale",
        default="strict",
        choices=sorted(_SCALE),
        help="span scoring scale: strict (default) or lenient (broad, benefit-of-the-doubt)",
    )
    ap.add_argument(
        "--sweep",
        type=Path,
        default=None,
        help="compact highlights file (hl-readout --compact --merge): judge every example",
    )
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--control-every", type=int, default=20, help="sweep: false-mark twin every N-th")
    ap.add_argument("--sweep-merge", action="store_true", help="{out}/sweep-*.json -> span_judged.json")
    args = ap.parse_args()
    global OUT
    OUT = args.out
    if args.report:
        report()
        return
    if args.export_still_rejected:
        export_still_rejected(args.efforts.split(",")[0].strip())
        return
    if args.sweep_merge:
        sweep_merge(OUT)
        return
    if args.sweep is not None:
        OUT.mkdir(parents=True, exist_ok=True)
        by_id = {f["pair_id"]: f for f in json.loads(args.features.read_text())["features"]}
        recs = sweep_records(args.sweep, by_id, args.desc_mode, args.shard, args.shards, args.control_every)
        n_ctrl = sum(r["control"] for r in recs)
        log.info(
            "sweep shard %d/%d: %d prompts = %d real + %d false-mark controls (desc-mode=%s)",
            args.shard,
            args.shards,
            len(recs),
            len(recs) - n_ctrl,
            n_ctrl,
            args.desc_mode,
        )
        if args.dry_run:
            print("\n===== SAMPLE =====\n" + recs[0]["content"][:1500])
            return
        run(recs, [args.efforts.split(",")[0].strip()], args, args.desc_mode, sweep_shard=args.shard)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    feats = json.loads(args.features.read_text())["features"]
    by_id = {f["pair_id"]: f for f in feats}
    accepted = {
        f["pair_id"]: len(f["examples"])
        for f in json.loads((DATASETS["lmsys"] / "highlights.json").read_text())["features"]
    }
    chosen = pick_pairs(feats, accepted)
    for pid, g in sorted(chosen.items(), key=lambda x: (x[1], x[0])):
        log.info(
            "pair %4d [%-9s] %s || %s   (%s)",
            pid,
            g,
            by_id[pid]["concept"],
            by_id[pid]["antagonist"],
            by_id[pid]["class_name"],
        )
    recs = build_records(by_id, chosen, args.desc_mode)
    n_ctrl = sum(r["control"] for r in recs)
    log.info(
        "%d prompts = %d real + %d false-mark controls, over %d pairs (desc-mode=%s)",
        len(recs),
        len(recs) - n_ctrl,
        n_ctrl,
        len(chosen),
        args.desc_mode,
    )
    systems = {k: system_for(k, args.desc_mode, args.scale) for k in ("chat", "doc")}
    (OUT / "prompts.json").write_text(
        json.dumps({"desc_mode": args.desc_mode, "systems": systems, "records": recs}, ensure_ascii=False)
    )
    if args.dry_run:
        pick = next(
            (
                r
                for r in recs
                if r["group"] == "verbosity" and r["dataset"] == "owt" and r["plain"] == "no" and not r["control"]
            ),
            recs[0],
        )
        print(
            "\n===== SYSTEM (conversation) =====\n"
            + systems["chat"]
            + "\n\n===== SYSTEM (web text) =====\n"
            + systems["doc"]
            + "\n\n===== SAMPLE USER PROMPT (verbosity, OWT, judge-rejected) =====\n"
            + pick["content"][:2200]
        )
        return
    run(recs, [e.strip() for e in args.efforts.split(",") if e.strip()], args, args.desc_mode)


if __name__ == "__main__":
    main()
