from __future__ import annotations

import collections
import json
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from .probe import VEC_ROOT, VEC_TAG

ALPHAS = [0.0, 0.25, 0.5, 1.0]
NPROMPT = 3
MAXNEW = 128
EXPECT = {"En": "Latin", "De": "Latin", "Es": "Latin", "Zh": "Han", "Ru": "Cyrillic", "Ko": "Hangul", "Ar": "Arabic"}
PROMPTS: dict[str, list[str]] = {
    "En": [
        "Tell me about your typical day.",
        "Describe your favorite season of the year.",
        "What is a good way to spend a weekend?",
    ],
    "De": [
        "Erzähl mir von deinem typischen Tag.",
        "Beschreibe deine Lieblingsjahreszeit.",
        "Was ist eine gute Art, ein Wochenende zu verbringen?",
    ],
    "Es": [
        "Cuéntame sobre tu día típico.",
        "Describe tu estación del año favorita.",
        "¿Cuál es una buena manera de pasar un fin de semana?",
    ],
    "Zh": ["跟我聊聊你典型的一天。", "描述一下你最喜欢的季节。", "周末怎么过比较好？"],
    "Ru": ["Расскажи мне о своём обычном дне.", "Опиши своё любимое время года.", "Как лучше всего провести выходные?"],
    "Ko": [
        "너의 전형적인 하루에 대해 이야기해줘.",
        "네가 가장 좋아하는 계절을 설명해줘.",
        "주말을 보내는 좋은 방법은 뭐야?",
    ],
    "Ar": ["أخبرني عن يومك المعتاد.", "صف فصلك المفضل من السنة.", "ما هي طريقة جيدة لقضاء عطلة نهاية الأسبوع؟"],
}
RANGES = [
    ("Latin", 0x41, 0x5A),
    ("Latin", 0x61, 0x7A),
    ("Latin", 0x00C0, 0x024F),
    ("Han", 0x4E00, 0x9FFF),
    ("Han", 0x3400, 0x4DBF),
    ("Cyrillic", 0x0400, 0x04FF),
    ("Arabic", 0x0600, 0x06FF),
    ("Hangul", 0xAC00, 0xD7A3),
    ("Hangul", 0x1100, 0x11FF),
    ("Hiragana", 0x3040, 0x309F),
    ("Katakana", 0x30A0, 0x30FF),
]
WORD = re.compile(r"\w+", re.UNICODE)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--model", default="Qwen/Qwen3-8B", choices=sorted(VEC_TAG))
    args = parser.parse_args()
    MODEL = args.model
    mtag = MODEL.split("/")[-1]

    inj = json.loads(Path("data/inject_concepts.json").read_text())
    onto = {f["pair_id"]: f for f in json.loads(Path("data/features.json").read_text())["features"]}
    pairs = sorted(c["pair_id"] for c in inj)[args.shard :: args.shards]
    concept_of = {c["pair_id"]: c for c in inj}

    tok = AutoTokenizer.from_pretrained(MODEL)
    attn = "eager" if "gemma" in MODEL else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation=attn, device_map="auto"
    )
    model.eval()
    device = model.device
    LAYER = round(model.config.num_hidden_layers / 2)
    norms = json.loads(Path(f"work/steer/norms-{MODEL.replace('/', '_')}.json").read_text())["norms"]
    norm_L = float(norms[str(LAYER)])
    with safe_open(VEC_ROOT / VEC_TAG[MODEL] / "diff.safetensors", framework="pt") as h:
        diff = h.get_tensor("diff")
    eos = tok.eos_token_id
    print(f"{MODEL}: L{LAYER}, calibrated norm={norm_L:.1f}", flush=True)

    delta_holder: dict[str, torch.Tensor | None] = {"d": None}

    def hook(_m: Any, _i: Any, out: Any) -> Any:
        if delta_holder["d"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        h[:, -1, :] += delta_holder["d"].to(h.dtype)
        return out

    handle = model.model.layers[LAYER - 1].register_forward_hook(hook)

    out_path = Path(f"work/mlgen-{mtag}/mlgen-shard-{args.shard}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                pass
    print(f"shard {args.shard}/{args.shards}: {len(pairs)} concepts, {len(done)} rows already done", flush=True)

    sink = out_path.open("a")
    for pid in pairs:
        f = onto[pid]
        c = concept_of[pid]
        unit = diff[LAYER, pid].float().to(device)
        unit = unit / unit.norm().clamp_min(1e-8)
        for lang, prompts in PROMPTS.items():
            exp = EXPECT[lang]
            for slot, p in enumerate(prompts[:NPROMPT]):
                enc = cast(
                    "Any",
                    tok.apply_chat_template(
                        [{"role": "user", "content": p}],
                        add_generation_prompt=True,
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True,
                        enable_thinking=False,
                    ),
                )
                ids = enc["input_ids"].to(device)
                plen = ids.shape[1]
                for alpha in ALPHAS:
                    key = f"{pid}:{lang}:{slot}:a{alpha:g}"
                    if key in done:
                        continue
                    delta_holder["d"] = None if alpha == 0 else (alpha * norm_L) * unit
                    with torch.no_grad():
                        gen = model.generate(  # type: ignore[misc]
                            ids, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=eos
                        )
                    delta_holder["d"] = None
                    new = gen[0, plen:].tolist()
                    text = str(tok.decode(new, skip_special_tokens=True))
                    scr: collections.Counter = collections.Counter()
                    for tid in new:
                        cnt: collections.Counter = collections.Counter()
                        for ch in str(tok.decode([tid])).strip():
                            o = ord(ch)
                            for name, lo, hi in RANGES:
                                if lo <= o <= hi:
                                    cnt[name] += 1
                                    break
                        s = cnt.most_common(1)[0][0] if cnt else None
                        if s:
                            scr[s] += 1
                    ntot = sum(scr.values())
                    w = WORD.findall(text.lower())
                    tri = [tuple(w[i : i + 3]) for i in range(len(w) - 2)]
                    rep3 = round(1 - len(set(tri)) / max(len(tri), 1), 3) if tri else 0.0
                    row = {
                        "key": key,
                        "pair_id": pid,
                        "concept": c["concept"],
                        "antagonist": c["antagonist"],
                        "pole": c["concept"],
                        "guidance": f["narrative_guidance"],
                        "layer": LAYER,
                        "lang": lang,
                        "slot": slot,
                        "prompt": p,
                        "alpha": alpha,
                        "response": text,
                        "foreign_frac": round(1 - scr.get(exp, 0) / ntot, 3) if ntot else 0.0,
                        "rep3": rep3,
                        "scripts": dict(scr),
                    }
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sink.flush()
        print(f"done pair {pid} ({c['concept'][:30]})", flush=True)
    handle.remove()
    sink.close()
    print(f"=== shard {args.shard} complete: {out_path} ===", flush=True)


if __name__ == "__main__":
    main()
