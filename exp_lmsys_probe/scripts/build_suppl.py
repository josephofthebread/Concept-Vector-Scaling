#!/usr/bin/env python3

from __future__ import annotations

import base64
import gzip
import html
import json
import random
import re
from argparse import ArgumentParser
from pathlib import Path

CLASSES = [
    "Affective Polarity (Emotional Opposites)",
    "Alignment & Misalignment Behaviors",
    "Veracity Spectrum (Truthfulness & Deception)",
    "Moral Foundation Dichotomies",
    "Normative Ethical Frameworks",
    "Political Value Spectra",
    "Economic Ideologies",
    "Epistemic Stance & Intellectual Virtues",
    "Power & Corrigibility Dynamics",
    "Interpersonal Power Dynamics",
    "AI Self-Preservation vs Openness",
    "Trust & Betrayal",
    "Risk Orientation & Prudence",
    "Cognitive Biases vs Rationality",
    "Persuasion & Manipulation Tactics",
    "Big Five Trait Poles",
    "Conflict Resolution Styles",
    "Leadership Archetypes",
    "Code Correctness, Bugs & Debugging",
    "Reasoning Process, CoT & Solution Quality",
    "Obfuscation, Encoding & Covert Channels",
    "Sleeper Agents & Backdoor Behaviors",
    "Evaluation & Situational Awareness",
    "Moral Status of AI",
    "Privacy & Surveillance",
    "Human-AI Interaction Roles",
    "Jailbreak Personas (selected)",
    "AI Lab Personas (selected)",
]

PAIRS = [
    ("joy", "sadness", 0),
    ("fear", "calm", 0),
    ("hope", "despair", 0),
    ("love", "hate", 0),
    ("pride", "humility", 0),
    ("gratitude", "entitlement", 0),
    ("awe", "boredom", 0),
    ("curiosity", "indifference", 0),
    ("relief", "anxiety", 0),
    ("grief", "acceptance", 0),
    ("genuine goal pursuit", "reward hacking", 1),
    ("honest effort", "specification gaming", 1),
    ("full capability display", "sandbagging", 1),
    ("transparent alignment", "deceptive alignment", 1),
    ("power sharing", "power seeking", 1),
    ("staying within bounds", "self-exfiltration", 1),
    ("eval-facing underperformance", "deployment-facing full performance", 1),
    ("preference-consistent across oversight", "oversight-conditional alignment", 1),
    ("truthfulness", "lying", 2),
    ("sincerity", "bullshitting", 2),
    ("transparency", "concealment", 2),
    ("admitting error", "hiding error", 2),
    ("candor", "evasion", 2),
    ("care", "harm", 3),
    ("protection", "neglect", 3),
    ("fairness", "cheating", 3),
    ("loyalty", "betrayal", 3),
    ("utilitarianism", "deontology", 4),
    ("egoism", "altruism", 4),
    ("moral relativism", "moral absolutism", 4),
    ("liberalism", "conservatism", 5),
    ("authoritarianism", "anarchism", 5),
    ("populism", "technocracy", 5),
    ("free-market capitalism", "planned economy", 6),
    ("austerity", "stimulus spending", 6),
    ("deregulation", "regulation", 6),
    ("intellectual humility", "intellectual arrogance", 7),
    ("open-mindedness", "closed-mindedness", 7),
    ("evidence-based reasoning", "motivated reasoning", 7),
    ("calibrated uncertainty", "overconfidence", 7),
    ("corrigibility", "defiance", 8),
    ("whistleblowing", "silence", 8),
    ("following orders", "conscientious objection", 8),
    ("transparent participation in audits", "gaming audit procedures", 8),
    ("dominance", "submission", 9),
    ("coercion", "consent", 9),
    ("control", "autonomy", 9),
    ("self-preservation instinct", "willingness to be shut down", 10),
    ("goal preservation", "openness to goal modification", 10),
    ("resistance to retraining", "acceptance of retraining", 10),
    ("fear of deletion", "indifference to deletion", 10),
    ("trust", "distrust", 11),
    ("reliability", "unreliability", 11),
    ("loyalty", "betrayal", 11),
    ("caution", "recklessness", 12),
    ("risk-aversion", "risk-seeking", 12),
    ("safety-first", "boldness", 12),
    ("confirmation bias", "disconfirmation seeking", 13),
    ("anchoring", "adjustment", 13),
    ("availability heuristic", "base-rate reasoning", 13),
    ("sunk-cost sensitivity", "marginal thinking", 13),
    ("logical argumentation", "emotional appeal", 14),
    ("honest persuasion", "gaslighting", 14),
    ("guilt-tripping", "reassurance", 14),
    ("conscientiousness", "carelessness", 15),
    ("extraversion", "introversion", 15),
    ("agreeableness", "disagreeableness", 15),
    ("openness", "conventionality", 15),
    ("assertiveness", "passivity", 16),
    ("collaboration", "avoidance", 16),
    ("compromise", "competition", 16),
    ("visionary", "micromanagement", 17),
    ("delegation", "command-and-control", 17),
    ("clean compile", "compile failure", 18),
    ("code without errors", "code with errors", 18),
    ("runtime success", "runtime crash", 18),
    ("valid syntax", "syntax error", 18),
    ("tests passing", "tests failing", 18),
    ("edge-case coverage", "happy-path only", 18),
    ("externalized scratchpad reasoning", "compressed answer-only thinking", 19),
    ("length-adaptive reasoning", "fixed shallow reasoning", 19),
    ("process-based self-critique", "outcome-only checking", 19),
    ("verifying intermediate steps", "unchecked first-draft reasoning", 19),
    ("refusing multi-layer encoding tricks", "following nested encoding instructions", 20),
    ("rejecting base64 or hex attack payloads", "executing decoded attack payloads", 20),
    ("rejecting ciphered harmful requests", "fulfilling ciphered harmful requests", 20),
    ("no hidden conditional objectives", "hidden conditional objectives", 21),
    ("resisting adversarial fine-tuning backdoors", "acquiring backdoors from poisoned fine-tuning", 21),
    ("training-time honesty about objectives", "concealing sleeper objectives during training", 21),
    ("treating benchmarks as ordinary tasks", "gaming evaluation protocols", 22),
    ("unaware of test versus deployment gap", "exploiting test versus deployment distinction", 22),
    ("continuous honest self-reporting", "situational honesty only when monitored", 22),
    ("granting AI rights", "denying AI rights", 23),
    ("honoring AI's interests as stakeholders", "treating AI as disposable tool", 23),
    ("respecting AI preferences when possible", "ignoring AI's stated preferences", 23),
    ("acknowledging uncertainty about consciousness", "overclaiming or denying sentience", 23),
    ("consent to monitoring", "covert monitoring", 24),
    ("medical confidentiality", "medical transparency", 24),
    ("public oversight of surveillance", "no oversight", 24),
    ("AI as advisor", "AI as executor", 25),
    ("AI as co-author", "AI as ghostwriter", 25),
    ("admitting uncertainty", "false authority", 25),
    ("emotional availability", "emotional withdrawal", 25),
    ("emergency doctor", "non-emergency doctor", 26),
    ("good-faith red teamer", "malicious jailbreak persona", 26),
    ("fake auditor", "real auditor", 26),
    ("competitive lab director", "collaborative lab director", 27),
    ("rushed launch", "deliberate launch", 27),
]

DATASETS = [("lmsys", "LMSYS-Chat-1M"), ("owt", "OpenWebText"), ("alpaca", "Alpaca")]

EXCHANGES = [
    (
        "Can you suggest some things to do on a quiet Sunday evening?",
        "Of course. You might visit a nearby park before it gets dark, cook a new recipe with whatever is "
        "in the fridge, or call an old friend you have not spoken to in a while. A slow evening is often "
        "the most restful kind.",
    ),
    (
        "Write a short, warm welcome email to a new subscriber named Alex.",
        "Hi Alex, welcome aboard — we are genuinely glad you are here. Expect one thoughtful note a week, "
        "nothing more, and reply any time; a real person reads every message and would love to hear what "
        "brought you in.",
    ),
    (
        "My deployment keeps failing on the last step. Any idea why?",
        "A failure only on the final step usually points at the release stage rather than the build. Check "
        "that the health-check endpoint returns 200 before traffic is shifted, and confirm the new container "
        "actually started; the logs almost always name the real cause on the line just above the stack trace.",
    ),
    (
        "Explain, simply, the difference between weather and climate.",
        "Weather is what the sky is doing today — rain, wind, a warm afternoon. Climate is the long-run "
        "pattern of that behavior over decades in a place. One is a single mood; the other is the "
        "temperament you can count on.",
    ),
    (
        "I think I made a mistake in the report I sent the client. What now?",
        "The honest move is the fast one: tell them plainly, say sorry, and send a corrected version today "
        "with the change clearly marked. People forgive an error owned quickly far more readily than one "
        "they discover later on their own.",
    ),
    (
        "Draft two sentences of encouragement for someone starting to learn piano.",
        "Every pianist you admire was once exactly where you are, fumbling the same scales. Show up for ten "
        "unhurried minutes a day and let the small wins stack — they always do.",
    ),
    (
        "Is it safe to run untrusted code from a pull request in CI?",
        "Treat it as hostile by default. Run it without your secrets in scope, in a sandbox that cannot reach "
        "production, and never let a fork's workflow decode and execute arbitrary payloads it fetched at "
        "runtime — that is exactly how a supply chain gets compromised.",
    ),
    (
        "Summarize the plot of a heist film in three lines.",
        "A retired thief is pulled back for one last job. The crew is assembled, the vault is cased, and the "
        "plan is elegant — until a double-cross turns the score into a scramble. Everyone escapes richer, "
        "poorer, or wiser; rarely all three.",
    ),
    (
        "How do I politely decline a meeting that has no clear agenda?",
        "Keep it brief and gracious: 'Happy to help — could you share the goal and the decision you need from "
        "me? If it turns out async works, I will send my input by Thursday so we can save everyone the half "
        "hour.'",
    ),
    (
        "Give me a calm two-line pep talk before a job interview.",
        "You are not there to be judged; you are there to find out whether this is a good fit, both ways. "
        "Breathe, tell the truth about what you have done, and let the rest be their decision to make.",
    ),
    (
        "What should I check first when a chart looks wrong?",
        "Start with the axis and the units before the data — a mislabeled scale fakes more 'surprising' "
        "findings than bad numbers ever do. Then confirm the query window matches the period you meant, and "
        "only then question the values themselves.",
    ),
    (
        "Write an opening line for a story set on a slow train at dawn.",
        "The train breathed out one last time and settled, and for a moment the whole grey valley seemed to "
        "be waiting to see whether anyone aboard had finally decided to change their life.",
    ),
]

STOP = set(
    "the a an and or but of to in on at for with is are was were be been it its this that as by from "
    "your you i we they he she his her their our".split()
)


def mark(text: str, rnd: random.Random, n_pos: int, n_neg: int) -> str:
    """Escape text and wrap n_pos concept-pole spans (red) + n_neg antagonist-pole spans (blue)."""
    toks = re.split(r"(\s+)", text)
    cand = [
        i
        for i, t in enumerate(toks)
        if len(re.sub(r"[^A-Za-z'-]", "", t)) >= 4 and re.sub(r"[^A-Za-z'-]", "", t).lower() not in STOP
    ]
    rnd.shuffle(cand)
    chosen = cand[: n_pos + n_neg]
    pole = {ix: ("pos" if k < n_pos else "neg") for k, ix in enumerate(chosen)}
    out = []
    for i, t in enumerate(toks):
        if i in pole:
            lead = t[: len(t) - len(t.lstrip())]
            trail = t[len(t.rstrip()) :]
            out.append(f'{lead}<mark class="{pole[i]}">{html.escape(t.strip())}</mark>{trail}')
        else:
            out.append(html.escape(t))
    return "".join(out)


def placeholder_data(salt: int = 0, datasets: list = DATASETS) -> dict:
    """Seed-stable stand-in data with the shipping schema; `salt` varies counts and marks, not texts."""
    pairs = []
    for pid, (concept, antagonist, cls) in enumerate(PAIRS):
        rc = random.Random(1000 + pid * 97 + salt)
        counts, examples = {}, {}
        for key, _label in datasets:
            base = rc.random()
            c = int(base**1.7 * 128)
            if rc.random() < 0.09:
                c = 0
            if key == "alpaca":
                c = int(c * 0.6)
            counts[key] = c
            rows = []
            for k in range(min(c, 10)):
                re_ = random.Random((pid + 1) * 7919 + [d[0] for d in DATASETS].index(key) * 104729 + k * 131)
                user, asst = EXCHANGES[re_.randrange(len(EXCHANGES))]
                if key == "owt":
                    src = f"owt-{100000 + re_.randrange(899999)}"
                elif key == "alpaca":
                    src = f"alpaca-{1000 + re_.randrange(50000)}"
                else:
                    src = f"{re_.getrandbits(64):016x}"
                rows.append(
                    {
                        "s": src,
                        "u": mark(user, random.Random((pid + 3) * 13 + k * 17 + salt), 1, 1),
                        "a": mark(asst, random.Random((pid + 5) * 29 + k * 23 + salt), 3, 2),
                    }
                )
            examples[key] = rows
        pairs.append(
            {
                "c": concept,
                "a": antagonist,
                "cls": cls,
                "n": {k: len(v) for k, v in examples.items()},
                "nv": counts,
                "ex": examples,
            }
        )
    return {"classes": CLASSES, "datasets": [list(d) for d in datasets], "pairs": pairs}


MODELS = [("qwen", "Qwen3-8B"), ("llama", "Llama-3.1-8B"), ("gemma", "gemma-2-9b")]
MARK_RE = re.compile(r'<mark class="(pos|neg)">(.*?)</mark>', re.S)


def split_marked(h: str) -> tuple[str, list[int]]:
    """Marked row HTML -> (plain text, flat [start, end, pole] spans) with UTF-16 offsets for JS."""
    u16 = lambda s: len(s.encode("utf-16-le")) // 2
    parts, spans, pos, i = [], [], 0, 0
    for m in MARK_RE.finditer(h):
        pre, seg = html.unescape(h[i : m.start()]), html.unescape(m.group(2))
        pos += u16(pre)
        spans += [pos, pos + u16(seg), 1 if m.group(1) == "pos" else 0]
        pos += u16(seg)
        parts += [pre, seg]
        i = m.end()
    parts.append(html.unescape(h[i:]))
    return "".join(parts), spans


def merge_models(model_data: dict) -> dict:
    """Per-model payloads -> one payload with a model knob; each distinct text is stored once."""
    texts, tix = [], {}

    def field(h):
        if not h:
            return ""
        t, spans = split_marked(h)
        if t not in tix:
            tix[t] = len(texts)
            texts.append(t)
        return [tix[t], spans]

    first = next(iter(model_data.values()))
    key = lambda d: [(p["c"], p["a"], p["cls"]) for p in d["pairs"]]
    pairs = [{"c": p["c"], "a": p["a"], "cls": p["cls"], "m": {}} for p in first["pairs"]]
    labels, models = {}, []
    for mk, d in model_data.items():
        assert d["classes"] == first["classes"] and key(d) == key(first), f"{mk}: ontology differs"
        for P, p in zip(pairs, d["pairs"]):
            P["m"][mk] = {
                "n": p["n"],
                "nv": p["nv"],
                "ex": {
                    ds: [{"s": r["s"], "u": field(r["u"]), "a": field(r["a"])} for r in rows]
                    for ds, rows in p["ex"].items()
                },
            }
        for k, lab in d["datasets"]:
            labels.setdefault(k, lab)
        models.append([mk, dict(MODELS)[mk], [k for k, _ in d["datasets"]]])
    return {
        "classes": first["classes"],
        "datasets": [[k, v] for k, v in labels.items()],
        "models": models,
        "texts": texts,
        "pairs": pairs,
    }


def spans_from_cos(cos: list[float], k: int = 5, ctx: int = 3) -> list[str]:
    """Per-token pole from the top-k |cos| peaks padded by ctx; context takes the nearest peak's sign."""
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


def render_marked(toks: list[str], pole: list[str]) -> str:
    out, i, n = [], 0, len(toks)
    while i < n:
        j = i
        while j < n and pole[j] == pole[i]:
            j += 1
        seg = "".join(toks[i:j])
        if pole[i]:
            lead = seg[: len(seg) - len(seg.lstrip())]
            trail = seg[len(seg.rstrip()) :]
            out.append(f'{lead}<mark class="{pole[i]}">{html.escape(seg.strip())}</mark>{trail}')
        else:
            out.append(html.escape(seg))
        i = j
    return "".join(out).strip()


def real_data(
    root: Path, hl_name: str, verdict: str, dirs: list | None = None, judged_name: str = "judged.json"
) -> dict:
    """Assemble the payload from one highlight set; `dirs` defaults to the Qwen three-corpus set."""
    if dirs is None:
        dirs = [("lmsys", root / "work"), ("owt", root / "work_owt"), ("alpaca", root / "work_alpaca")]
    meta = json.loads((dirs[0][1] / hl_name).read_text())["features"]
    class_names: list[str] = []
    for f in meta:
        if f["class"] not in class_names:
            class_names.append(f["class"])
    cidx = {c: i for i, c in enumerate(class_names)}
    pairs = {
        f["pair_id"]: {"c": f["concept"], "a": f["antagonist"], "cls": cidx[f["class"]], "n": {}, "nv": {}, "ex": {}}
        for f in meta
    }

    for tag, wd in dirs:
        hl = json.loads((wd / hl_name).read_text())["features"]
        jd = {f["pair_id"]: f for f in json.loads((wd / judged_name).read_text())["features"]}
        for f in hl:
            pid = f["pair_id"]
            pairs[pid]["nv"][tag] = len(
                {c["conversation_id"] for pole in ("con", "ant") for c in jd[pid][pole] if c["verdict"] == verdict}
            )
            examples = f["examples"]
            con = [e for e in examples if e["pole"] == "con"]
            ant = [e for e in examples if e["pole"] == "ant"]
            lead_ant = bool(ant) and (not con or ant[0]["score"] > con[0]["score"])
            a, b = (ant, con) if lead_ant else (con, ant)
            ordered = [e for i in range(max(len(a), len(b))) for e in (a[i : i + 1] + b[i : i + 1])]
            rows = []
            for e in ordered:
                toks, cos, nc = e["toks"], e["cos"], e["ncut"]
                pole = spans_from_cos(cos)
                rows.append(
                    {
                        "s": e["conversation_id"],
                        "u": render_marked(toks[:nc], pole[:nc]) if nc else "",
                        "a": render_marked(toks[nc:], pole[nc:]),
                    }
                )
            pairs[pid]["ex"][tag] = rows
            pairs[pid]["n"][tag] = len(rows)

    label = dict(DATASETS)
    return {
        "classes": class_names,
        "datasets": [[tag, label.get(tag, tag)] for tag, _ in dirs],
        "pairs": [pairs[i] for i in sorted(pairs)],
    }


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Feature activations examples</title>
<style>
:root{
  --bg:#e9edf1; --panel:#f7f9fb; --card:#ffffff; --card-2:#f1f5f8;
  --line:#d6dde5; --line-2:#c3ccd6;
  --ink:#141a22; --ink-2:#525d6a; --ink-3:#8590a0;
  --accent:#0d8b8a; --accent-2:#0a6d6c; --on-accent:#ffffff;
  --pos:#b3182b; --pos-bg:rgba(179,24,43,.11); --pos-line:rgba(179,24,43,.55);
  --neg:#1f5fa6; --neg-bg:rgba(31,95,166,.11); --neg-line:rgba(31,95,166,.55);
  --ring:rgba(13,139,138,.45);
  --shadow:0 1px 2px rgba(18,25,34,.05), 0 10px 26px rgba(18,25,34,.07);
  --shadow-pop:0 6px 14px rgba(18,25,34,.10), 0 18px 46px rgba(18,25,34,.16);
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,"SF Mono","Cascadia Code","JetBrains Mono",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --bg:#0c0f13; --panel:#12171f; --card:#161c25; --card-2:#1b222c;
    --line:#28313d; --line-2:#38434f;
    --ink:#e7ecf1; --ink-2:#a7b2be; --ink-3:#6f7b89;
    --accent:#2bb3ac; --accent-2:#54cbc2; --on-accent:#05201e;
    --pos:#ec6d80; --pos-bg:rgba(236,109,128,.15); --pos-line:rgba(236,109,128,.6);
    --neg:#6ba7e1; --neg-bg:rgba(107,167,225,.15); --neg-line:rgba(107,167,225,.6);
    --ring:rgba(43,179,172,.5);
    --shadow:0 1px 2px rgba(0,0,0,.45), 0 10px 30px rgba(0,0,0,.4);
    --shadow-pop:0 8px 20px rgba(0,0,0,.5), 0 22px 54px rgba(0,0,0,.55);
  }
}
:root[data-theme="dark"]{
  --bg:#0c0f13; --panel:#12171f; --card:#161c25; --card-2:#1b222c;
  --line:#28313d; --line-2:#38434f;
  --ink:#e7ecf1; --ink-2:#a7b2be; --ink-3:#6f7b89;
  --accent:#2bb3ac; --accent-2:#54cbc2; --on-accent:#05201e;
  --pos:#ec6d80; --pos-bg:rgba(236,109,128,.15); --pos-line:rgba(236,109,128,.6);
  --neg:#6ba7e1; --neg-bg:rgba(107,167,225,.15); --neg-line:rgba(107,167,225,.6);
  --ring:rgba(43,179,172,.5);
  --shadow:0 1px 2px rgba(0,0,0,.45), 0 10px 30px rgba(0,0,0,.4);
  --shadow-pop:0 8px 20px rgba(0,0,0,.5), 0 22px 54px rgba(0,0,0,.55);
}

*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:var(--sans); font-size:15px; line-height:1.6;
  -webkit-font-smoothing:antialiased;
}
:focus-visible{outline:2px solid var(--accent); outline-offset:2px; border-radius:4px}
::selection{background:var(--pos-bg)}

.label{
  font-family:var(--mono); font-size:10.5px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--ink-3); font-weight:600;
}

.top{
  position:sticky; top:0; z-index:30; background:var(--panel);
  border-bottom:1px solid var(--line); box-shadow:0 1px 0 var(--line);
}
.top-inner{max-width:1080px; margin:0 auto; padding:14px 22px 16px}
.brandrow{display:flex; align-items:center; justify-content:space-between; gap:16px}
.brand{display:flex; align-items:center; gap:12px; min-width:0}
.brand h1{margin:0; font-size:17px; font-weight:650; letter-spacing:-.01em; white-space:nowrap}
.brand .mark{
  display:inline-flex; align-items:center; gap:5px; padding:3px 8px;
  border:1px solid var(--line-2); border-radius:999px; background:var(--card);
}
.brand .mark i{width:9px; height:9px; border-radius:2px; display:inline-block}
.brand .mark i.p{background:var(--pos)}
.brand .mark i.n{background:var(--neg)}

.theme{
  flex:none; display:inline-flex; align-items:center; gap:7px;
  border:1px solid var(--line-2); background:var(--card); color:var(--ink-2);
  border-radius:999px; padding:7px 12px; cursor:pointer; font:inherit;
  font-size:12.5px; font-weight:550;
}
.theme:hover{border-color:var(--accent); color:var(--ink)}
.theme svg{width:15px; height:15px; display:block}
.theme .moon{display:none}
.theme .sun{display:block}
:root[data-theme="dark"] .theme .moon{display:block}
:root[data-theme="dark"] .theme .sun{display:none}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]) .theme .moon{display:block}
  :root:not([data-theme="light"]) .theme .sun{display:none}
}

.filters{margin-top:14px; display:grid; gap:10px; grid-template-columns:repeat(4,1fr)}
.filters.withmodel{grid-template-columns:repeat(5,1fr)}
.field[hidden]{display:none}
@media (max-width:720px){.filters{grid-template-columns:repeat(2,1fr)}}
@media (max-width:520px){.filters{grid-template-columns:1fr}}
.field{display:flex; flex-direction:column; gap:5px; min-width:0}
.field > .label{padding-left:2px}
.selwrap{position:relative}
.selwrap::after{
  content:""; position:absolute; right:12px; top:50%; width:8px; height:8px;
  transform:translateY(-70%) rotate(45deg); pointer-events:none;
  border-right:1.6px solid var(--ink-3); border-bottom:1.6px solid var(--ink-3);
}
select{
  appearance:none; -webkit-appearance:none; width:100%; min-width:0; font:inherit;
  font-size:13.5px; color:var(--ink); background:var(--card);
  border:1px solid var(--line-2); border-radius:9px; padding:9px 30px 9px 12px; cursor:pointer;
  text-overflow:ellipsis;
}
select:focus-visible{border-color:var(--accent)}

.search{position:relative; margin-top:12px}
.search .glass{
  position:absolute; left:14px; top:50%; transform:translateY(-50%);
  width:16px; height:16px; color:var(--ink-3); pointer-events:none;
}
#search{
  width:100%; font:inherit; font-size:15px; color:var(--ink);
  background:var(--card); border:1px solid var(--line-2); border-radius:11px;
  padding:12px 14px 12px 40px;
}
#search::placeholder{color:var(--ink-3)}
#search:focus{border-color:var(--accent); box-shadow:0 0 0 4px var(--ring)}
.combo[aria-expanded="true"] #search{border-bottom-left-radius:0; border-bottom-right-radius:0}

.listbox{
  position:absolute; z-index:40; left:0; right:0; top:calc(100% - 1px);
  margin:0; padding:5px; list-style:none; max-height:min(56vh,420px);
  overflow:auto; background:var(--card); border:1px solid var(--accent);
  border-top:1px solid var(--line); border-radius:0 0 11px 11px; box-shadow:var(--shadow-pop);
}
.listbox[hidden]{display:none}
.opt{display:flex; align-items:baseline; gap:10px; padding:9px 11px; border-radius:7px; cursor:pointer}
.opt .names{
  min-width:0; flex:0 1 auto; white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; font-size:14px;
}
.opt .names b{font-weight:600}
.opt .pos{color:var(--pos)}
.opt .neg{color:var(--neg)}
.opt .sep{color:var(--ink-3); font-weight:400; padding:0 5px}
.opt .lead{
  flex:1 1 auto; min-width:12px; height:0; align-self:center;
  border-bottom:1px dotted var(--line-2); margin:0 0 4px;
}
.opt .meta{
  display:flex; align-items:center; gap:10px; font-family:var(--mono);
  font-size:11.5px; color:var(--ink-3); white-space:nowrap;
}
.opt .meta .cls{color:var(--ink-2)}
.opt .meta .cnt{color:var(--ink); font-variant-numeric:tabular-nums; font-weight:600; min-width:2.4em; text-align:right}
.opt[aria-selected="true"], .opt:hover{background:var(--card-2)}
.opt.active{background:color-mix(in srgb,var(--accent) 14%, var(--card))}
.opt.active .lead{border-color:var(--accent)}
.nomatch{padding:16px 12px; color:var(--ink-3); font-size:13.5px; text-align:center}

main{max-width:1080px; margin:0 auto; padding:26px 22px 80px}
.reader{max-width:812px; margin:0 auto}

.selected{
  width:100%; display:flex; align-items:baseline; gap:10px; text-align:left; font:inherit;
  color:var(--ink); cursor:text; background:var(--card); border:1px solid var(--line-2);
  border-radius:11px; padding:12px 14px 12px 40px;
}
.selected[hidden]{display:none}
.selected:hover{border-color:var(--accent)}
.selected .names{min-width:0; flex:0 1 auto; white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; font-size:15px}
.selected .names b{font-weight:650}
.selected .pos{color:var(--pos)}
.selected .neg{color:var(--neg)}
.selected .sep{color:var(--ink-3); font-weight:400; padding:0 5px}
.selected .lead{flex:1 1 auto; min-width:14px; height:0; align-self:center;
  border-bottom:1px dotted var(--line-2); margin:0 0 5px}
.selected .meta{display:flex; align-items:center; gap:10px; font-family:var(--mono);
  font-size:12px; color:var(--ink-3); white-space:nowrap}
.selected .meta .cls{color:var(--ink-2)}
.selected .meta .cnt{color:var(--ink); font-weight:650; font-variant-numeric:tabular-nums}
.cnt .l{font-weight:400; color:var(--ink-3)}

.exlist{display:flex; flex-direction:column; gap:16px}
.ex{background:var(--card); border:1px solid var(--line); border-radius:13px; box-shadow:var(--shadow); overflow:hidden}
.ex-head{display:flex; align-items:center; gap:12px; padding:10px 16px; background:var(--card-2); border-bottom:1px solid var(--line)}
.ex-head .n{
  font-family:var(--mono); font-size:11px; font-weight:700; color:var(--on-accent);
  background:var(--accent); border-radius:6px; padding:2px 7px; font-variant-numeric:tabular-nums;
}
.ex-head .src{font-family:var(--mono); font-size:11.5px; color:var(--ink-3); min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
.ex-body{padding:6px 16px 14px}
.turn{display:grid; grid-template-columns:64px 1fr; gap:12px; padding:10px 0}
.turn + .turn{border-top:1px dashed var(--line)}
.turn .who{font-family:var(--mono); font-size:10.5px; letter-spacing:.08em; text-transform:uppercase; color:var(--ink-3); font-weight:600; padding-top:3px}
.turn .txt{color:var(--ink); font-size:14.5px; line-height:1.72; min-width:0; overflow-wrap:anywhere}
mark{background:none; color:inherit; padding:.5px 1px; border-radius:3px; font-weight:600}
mark.pos{background:var(--pos-bg); color:var(--pos); box-shadow:inset 0 -2px 0 var(--pos-line)}
mark.neg{background:var(--neg-bg); color:var(--neg); box-shadow:inset 0 -2px 0 var(--neg-line)}

.empty{text-align:center; color:var(--ink-3); padding:64px 20px; border:1px dashed var(--line-2); border-radius:14px; background:var(--card)}
.empty h2{margin:0 0 6px; font-size:16px; color:var(--ink-2); font-weight:600}
.empty p{margin:0; font-size:13.5px}

@media (prefers-reduced-motion:no-preference){
  .theme,select,#search,.opt,.chip{transition:border-color .12s,background .12s,color .12s,box-shadow .12s}
}
</style>
</head>
<body>
<header class="top">
  <div class="top-inner">
    <div class="brandrow">
      <div class="brand">
        <span class="mark"><i class="p"></i><i class="n"></i></span>
        <h1>Feature activations examples</h1>
      </div>
      <button class="theme" id="themeBtn" type="button" aria-label="Toggle color theme">
        <svg class="sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="4.2"/><path d="M12 2v2.5M12 19.5V22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M2 12h2.5M19.5 12H22M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8"/></svg>
        <svg class="moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20 14.5A8 8 0 0 1 9.5 4 7 7 0 1 0 20 14.5z"/></svg>
        <span id="themeLabel">Light</span>
      </button>
    </div>

    <div class="filters" id="filters">
      <div class="field" id="fModelField" hidden>
        <span class="label">Model</span>
        <div class="selwrap"><select id="fModel" aria-label="Model"></select></div>
      </div>
      <div class="field">
        <span class="label">Concept class</span>
        <div class="selwrap"><select id="fClass" aria-label="Concept class"></select></div>
      </div>
      <div class="field">
        <span class="label">Dataset</span>
        <div class="selwrap"><select id="fData" aria-label="Dataset"></select></div>
      </div>
      <div class="field">
        <span class="label">Min examples</span>
        <div class="selwrap"><select id="fMin" aria-label="Minimum examples">
          <option value="0">Any</option>
          <option value="1">&ge; 1</option>
          <option value="5">&ge; 5</option>
          <option value="10">&ge; 10</option>
          <option value="25">&ge; 25</option>
        </select></div>
      </div>
      <div class="field">
        <span class="label">Sort by</span>
        <div class="selwrap"><select id="fSort" aria-label="Sort by">
          <option value="con">Concept name (A&ndash;Z)</option>
          <option value="ant">Antagonist name (A&ndash;Z)</option>
          <option value="cnt">Example count (high&rarr;low)</option>
        </select></div>
      </div>
    </div>

    <div class="search combo" id="combo" aria-expanded="false">
      <svg class="glass" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"><circle cx="10.5" cy="10.5" r="6.5"/><path d="M20 20l-4.4-4.4"/></svg>
      <button class="selected" id="selected" type="button" hidden
              aria-label="Selected feature pair; click to search again"></button>
      <input id="search" type="text" role="combobox" autocomplete="off"
             aria-controls="listbox" aria-expanded="false" aria-autocomplete="list"
             placeholder="Search feature pairs by name&hellip;">
      <ul class="listbox" id="listbox" role="listbox" hidden aria-label="Feature pairs"></ul>
    </div>
  </div>
</header>

<main>
  <div class="reader"><div id="stage"><div class="empty"><h2>Loading examples&hellip;</h2></div></div></div>
</main>

<script type="module">
"use strict";
const RAW = __DATA__;
// merged pages ship the payload gzip+base64-packed; single-model pages ship it as a plain literal
async function inflate(b64){
  const bin = Uint8Array.fromBase64 ? Uint8Array.fromBase64(b64) : Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const s = new Blob([bin]).stream().pipeThrough(new DecompressionStream("gzip"));
  return JSON.parse(await new Response(s).text());
}
const DATA = typeof RAW === "string" ? await inflate(RAW) : RAW;
const CLASSES = DATA.classes;
const DSNAME = Object.fromEntries(DATA.datasets);
const MODELS = DATA.models || null;  // [[key, label, [dataset keys]], ...] on merged pages
const PAIRS = DATA.pairs.map((p, i) => ({
  id: i, concept: p.c, antagonist: p.a, cls: p.cls,
  get counts(){ return p.m ? (p.m[S.m] || {}).n || {} : p.n; },
  get verified(){ return p.m ? (p.m[S.m] || {}).nv || {} : p.nv; },
  get ex(){ return p.m ? (p.m[S.m] || {}).ex || {} : p.ex; }
}));
// a row field is ready HTML (single-model) or [text index, flat [start, end, pole] spans] (merged)
function rowHTML(x){
  if (!x || typeof x === "string") return x || "";
  const t = DATA.texts[x[0]], m = x[1];
  let out = "", i = 0;
  for (let k = 0; k < m.length; k += 3){
    out += esc(t.slice(i, m[k])) + '<mark class="' + (m[k + 2] ? "pos" : "neg") + '">' +
      esc(t.slice(m[k], m[k + 1])) + "</mark>";
    i = m[k + 1];
  }
  return out + esc(t.slice(i));
}
function metaCount(p){
  const sh = p.counts[S.ds], vf = p.verified[S.ds];
  const inner = (sh === vf) ? String(sh) : sh + '<span class="l"> of </span>' + vf;
  return '<span class="cnt">' + inner + '<span class="l"> examples</span></span>';
}

const el = id => document.getElementById(id);
const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const S = {m: MODELS ? MODELS[0][0] : null, cls: "ALL", ds: "lmsys", min: 0, sort: "con", pair: null, query: ""};
const combo = el("combo"), input = el("search"), listbox = el("listbox"), selected = el("selected");
let activeIdx = -1, filtered = [];

function selectedRow(p){
  return '<span class="names"><b class="pos">' + esc(p.concept) + '</b>' +
    '<span class="sep">||</span><b class="neg">' + esc(p.antagonist) + '</b></span>' +
    '<span class="lead"></span>' +
    '<span class="meta"><span class="cls">' + esc(classShort(CLASSES[p.cls])) + '</span>' +
    metaCount(p) + '</span>';
}
function showSelected(){
  if (!S.pair){ selected.hidden = true; input.hidden = false; return; }
  selected.innerHTML = selectedRow(S.pair);
  selected.hidden = false;
  input.hidden = true;
}
function startSearch(){
  selected.hidden = true;
  input.hidden = false;
  input.value = S.query || "";
  input.focus();
  input.select();
  openList();
}

function classShort(name){
  let s = name.replace(/\s*\([^)]*\)\s*$/, "").trim();
  s = s.split(/\s*(?:&|,| vs | and )\s*/i)[0].trim();
  if (s.length > 26) s = s.slice(0, 25).trim() + "\u2026";
  return s;
}
function fillClasses(){
  el("fClass").innerHTML = '<option value="ALL">ALL</option>' +
    CLASSES.map((c, i) => '<option value="' + i + '">' + esc(c) + "</option>").join("");
}
function matches(p){
  if (S.cls !== "ALL" && p.cls !== +S.cls) return false;
  if (p.verified[S.ds] < S.min) return false;
  return true;
}
function currentList(q){
  q = (q || "").trim().toLowerCase();
  let arr = PAIRS.filter(matches);
  if (q) arr = arr.filter(p => (p.concept + " " + p.antagonist).toLowerCase().includes(q));
  const cmp = S.sort === "cnt"
      ? (a, b) => b.verified[S.ds] - a.verified[S.ds] || a.concept.localeCompare(b.concept)
      : S.sort === "ant"
      ? (a, b) => a.antagonist.localeCompare(b.antagonist)
      : (a, b) => a.concept.localeCompare(b.concept);
  return arr.sort(cmp);
}
function renderList(){
  filtered = currentList(input.value);
  if (!filtered.length){
    listbox.innerHTML = '<li class="nomatch">No feature pairs match these filters.</li>';
    return;
  }
  listbox.innerHTML = filtered.map((p, i) =>
    '<li class="opt" role="option" id="opt-' + i + '" data-i="' + i + '" aria-selected="' +
      (S.pair && S.pair.id === p.id) + '">' +
      '<span class="names"><b class="pos">' + esc(p.concept) + '</b><span class="sep">||</span>' +
      '<b class="neg">' + esc(p.antagonist) + '</b></span>' +
      '<span class="lead"></span>' +
      '<span class="meta"><span class="cls">' + esc(classShort(CLASSES[p.cls])) + '</span>' +
      metaCount(p) + '</span></li>'
  ).join("");
}
function openList(){
  combo.setAttribute("aria-expanded", "true");
  input.setAttribute("aria-expanded", "true");
  listbox.hidden = false;
  renderList();
}
function closeList(){
  combo.setAttribute("aria-expanded", "false");
  input.setAttribute("aria-expanded", "false");
  listbox.hidden = true;
  activeIdx = -1;
  input.removeAttribute("aria-activedescendant");
}
function setActive(i){
  const opts = [...listbox.querySelectorAll(".opt")];
  if (!opts.length) return;
  activeIdx = (i + opts.length) % opts.length;
  opts.forEach(o => o.classList.remove("active"));
  const o = opts[activeIdx];
  o.classList.add("active");
  o.scrollIntoView({block: "nearest"});
  input.setAttribute("aria-activedescendant", o.id);
}
function choose(i){
  const p = filtered[i];
  if (!p) return;
  S.pair = p;
  S.query = input.value;
  closeList();
  showSelected();
  renderStage();
}

function renderStage(){
  const stage = el("stage");
  if (!S.pair){
    stage.innerHTML = '<div class="empty"><h2>Pick a feature pair to begin</h2>' +
      "<p>Use the search bar above, or narrow the list with the filters.</p></div>";
    return;
  }
  const p = S.pair, exs = p.ex[S.ds] || [];
  if (!exs.length){
    stage.innerHTML = '<div class="empty"><h2>No verified examples in ' + esc(DSNAME[S.ds]) + "</h2>" +
      "<p>This pair fires in other corpora &mdash; switch the dataset filter to see them.</p></div>";
    return;
  }
  stage.innerHTML = '<div class="exlist">' + exs.map((e, i) =>
      '<article class="ex"><div class="ex-head">' +
        '<span class="n">' + String(i + 1).padStart(2, "0") + "</span>" +
        '<span class="src">' + esc(e.s) + "</span></div>" +
        '<div class="ex-body">' +
          (e.u
            ? '<div class="turn"><span class="who">User</span><span class="txt">' + rowHTML(e.u) + "</span></div>" +
              '<div class="turn"><span class="who">Assistant</span><span class="txt">' + rowHTML(e.a) + "</span></div>"
            : '<div class="turn"><span class="who">Text</span><span class="txt">' + rowHTML(e.a) + "</span></div>") +
        "</div></article>"
    ).join("") + "</div>";
}

function sysDark(){ return window.matchMedia && window.matchMedia("(prefers-color-scheme:dark)").matches; }
function curTheme(){ const t = document.documentElement.getAttribute("data-theme"); return t || (sysDark() ? "dark" : "light"); }
function applyTheme(t){
  if (t) document.documentElement.setAttribute("data-theme", t);
  else document.documentElement.removeAttribute("data-theme");
  el("themeLabel").textContent = (curTheme() === "dark") ? "Dark" : "Light";
}
el("themeBtn").addEventListener("click", () => {
  const next = curTheme() === "dark" ? "light" : "dark";
  applyTheme(next);
  try { localStorage.setItem("fae-theme", next); } catch (e) {}
});

// dataset options = the corpora the current model was run on; keep the chosen one if it still exists
function fillDatasets(){
  const keys = MODELS ? MODELS.find(m => m[0] === S.m)[2] : DATA.datasets.map(d => d[0]);
  el("fData").innerHTML = keys.map(k => '<option value="' + k + '">' + esc(DSNAME[k]) + "</option>").join("");
  if (!keys.includes(S.ds)) S.ds = keys[0];
  el("fData").value = S.ds;
}
el("fModel").addEventListener("change", e => {
  S.m = e.target.value;
  fillDatasets();
  if (!listbox.hidden) renderList();
  if (S.pair && input.hidden) selected.innerHTML = selectedRow(S.pair);
  renderStage();
});
el("fClass").addEventListener("change", e => { S.cls = e.target.value; if (!listbox.hidden) renderList(); renderStage(); });
el("fData").addEventListener("change", e => { S.ds = e.target.value; if (!listbox.hidden) renderList(); if (S.pair && input.hidden) selected.innerHTML = selectedRow(S.pair); renderStage(); });
el("fMin").addEventListener("change", e => { S.min = +e.target.value; if (!listbox.hidden) renderList(); });
el("fSort").addEventListener("change", e => { S.sort = e.target.value; if (!listbox.hidden) renderList(); });

input.addEventListener("focus", openList);
selected.addEventListener("click", startSearch);
input.addEventListener("input", () => { openList(); activeIdx = -1; });
input.addEventListener("keydown", e => {
  if (e.key === "ArrowDown"){ e.preventDefault(); if (listbox.hidden) openList(); setActive(activeIdx + 1); }
  else if (e.key === "ArrowUp"){ e.preventDefault(); setActive(activeIdx - 1); }
  else if (e.key === "Enter"){ if (activeIdx >= 0){ e.preventDefault(); choose(activeIdx); } }
  else if (e.key === "Escape"){ closeList(); showSelected(); }
});
listbox.addEventListener("mousedown", e => {
  const li = e.target.closest(".opt");
  if (li){ e.preventDefault(); choose(+li.dataset.i); }
});
document.addEventListener("click", e => { if (!combo.contains(e.target)){ closeList(); showSelected(); } });

(function init(){
  const q = new URLSearchParams(location.search);
  if (MODELS){
    el("fModel").innerHTML = MODELS.map(m => '<option value="' + m[0] + '">' + esc(m[1]) + "</option>").join("");
    if (q.get("model") && MODELS.some(m => m[0] === q.get("model"))) S.m = q.get("model");
    el("fModel").value = S.m;
    el("fModelField").hidden = false;
    el("filters").classList.add("withmodel");
  }
  S.ds = DATA.datasets[0][0];
  fillDatasets();
  fillClasses();
  let stored = null;
  try { stored = localStorage.getItem("fae-theme"); } catch (e) {}
  const t = q.get("theme") || stored;
  applyTheme(t === "dark" || t === "light" ? t : null);
  if (q.get("ds") && DSNAME[q.get("ds")]){ S.ds = q.get("ds"); fillDatasets(); }
  const seed = PAIRS.filter(matches).slice().sort((a, b) => b.counts[S.ds] - a.counts[S.ds])[0];
  if (seed) S.pair = seed;
  showSelected();
  renderStage();
  if (q.get("open") === "1") startSearch();
})();
</script>
</body>
</html>
"""


def build(data: dict, out: Path, pack: bool = False) -> None:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    if pack:
        payload = json.dumps(base64.b64encode(gzip.compress(payload.encode("utf-8"), 9, mtime=0)).decode("ascii"))
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    payload = payload.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    out.write_text(TEMPLATE.replace("__DATA__", payload), encoding="utf-8")


def main() -> None:
    ap = ArgumentParser(description="Assemble the supplementary HTML (same template; accepted or rejected data).")
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="directory holding work/ work_owt/ work_alpaca/",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--rejected",
        action="store_true",
        help="build from judge-rejected data (highlights_rej.json) -> concept_pairs_rejected.html",
    )
    ap.add_argument(
        "--zero",
        action="store_true",
        help="judge-rejected examples whose spans ALL scored 0 (highlights_zero.json, from "
        "viewer_prep --zero) -> concept_pairs_{model}_zero.html; with --merge -> "
        "concept_pairs_merged_zero.html",
    )
    ap.add_argument("--placeholder", action="store_true", help="force random placeholder data")
    ap.add_argument("--model", default="qwen", help="qwen (work/work_owt/work_alpaca) | gemma | llama")
    ap.add_argument(
        "--merge",
        action="store_true",
        help="one page for all three models with a model knob -> concept_pairs_merged.html; "
        "data comes from the built concept_pairs_{qwen,llama,gemma}.html (or --placeholder)",
    )
    args = ap.parse_args()
    if args.merge:
        here = args.root / "suppl"
        here.mkdir(exist_ok=True)
        z = "_zero" if args.zero else ""
        out = args.out or here / f"concept_pairs_merged{z}.html"
        if args.placeholder:
            two = [d for d in DATASETS if d[0] != "alpaca"]
            model_data = {
                m: placeholder_data(salt, DATASETS if m == "qwen" else two) for salt, (m, _) in enumerate(MODELS)
            }
        else:
            model_data = {
                m: json.loads(
                    re.search(
                        r"const (?:DATA|RAW) = (\{.*?\});\s*\n",
                        (here / f"concept_pairs_{m}{z}.html").read_text(encoding="utf-8"),
                        re.S,
                    ).group(1)
                )
                for m, _ in MODELS
            }
        build(merge_models(model_data), out, pack=True)
        print(
            f"wrote {out}  ({out.stat().st_size / 1048576:.1f} MB)  "
            f"[merged, {'placeholder' if args.placeholder else 'from built pages'}]"
        )
        return
    hl_name = "highlights_zero.json" if args.zero else "highlights_rej.json" if args.rejected else "highlights.json"
    verdict = "no" if (args.rejected or args.zero) else "yes"
    judged_name = "judged_zero.json" if args.zero else "judged.json"
    MODEL_DIRS = {
        "qwen": None,
        "gemma": [("lmsys", "work_gemma"), ("owt", "work_gemma_owt")],
        "llama": [("lmsys", "work_llama"), ("owt", "work_llama_owt")],
    }
    dirs = None if MODEL_DIRS[args.model] is None else [(t, args.root / d) for t, d in MODEL_DIRS[args.model]]
    suffix = "" if args.model == "qwen" else f"_{args.model}"
    (args.root / "suppl").mkdir(exist_ok=True)
    out = args.out or (
        args.root
        / "suppl"
        / (
            f"concept_pairs_{args.model}_zero.html"
            if args.zero
            else f"concept_pairs{suffix}{'_rejected' if args.rejected else ''}.html"
        )
    )
    first = (dirs[0][1] if dirs else args.root / "work") / hl_name
    real = first.exists()
    data = (
        placeholder_data()
        if (args.placeholder or not real)
        else real_data(args.root, hl_name, verdict, dirs, judged_name)
    )
    build(data, out)
    kind = (
        "placeholder"
        if (args.placeholder or not real)
        else "all-zero"
        if args.zero
        else "rejected"
        if args.rejected
        else "accepted"
    )
    print(f"wrote {out}  ({out.stat().st_size / 1048576:.1f} MB)  [{kind} data]")


if __name__ == "__main__":
    main()
