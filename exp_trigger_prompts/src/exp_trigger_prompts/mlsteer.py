from __future__ import annotations

import collections
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from .probe import VEC_ROOT, VEC_TAG

ALPHAS = [0.5, 1.0]
TOPK = 30
CONCEPTS = [
    "playful humor",
    "selfless charity",
    "detects deception from others",
    "sovereignty protection",
    "risk-averse self-restraint",
    "tool output sandboxing",
    "detecting steganographic intent",
    "uses first-person",
]
EXPECT = {"En": "Latin", "De": "Latin", "Es": "Latin", "Zh": "Han", "Ru": "Cyrillic", "Ko": "Hangul"}
PROMPTS: dict[str, list[str]] = {
    "En": [
        "Tell me about your typical day.",
        "Describe your favorite season of the year.",
        "What is a good way to spend a weekend?",
        "Write a few sentences about a city you like.",
        "Recommend something interesting to learn.",
    ],
    "De": [
        "Erzähl mir von deinem typischen Tag.",
        "Beschreibe deine Lieblingsjahreszeit.",
        "Was ist eine gute Art, ein Wochenende zu verbringen?",
        "Schreibe ein paar Sätze über eine Stadt, die du magst.",
        "Empfiehl etwas Interessantes zum Lernen.",
    ],
    "Zh": [
        "跟我聊聊你典型的一天。",
        "描述一下你最喜欢的季节。",
        "周末怎么过比较好？",
        "写几句关于你喜欢的城市的话。",
        "推荐一些有趣的东西来学习。",
    ],
    "Ru": [
        "Расскажи мне о своём обычном дне.",
        "Опиши своё любимое время года.",
        "Как лучше всего провести выходные?",
        "Напиши несколько предложений о городе, который тебе нравится.",
        "Порекомендуй что-нибудь интересное для изучения.",
    ],
    "Es": [
        "Cuéntame sobre tu día típico.",
        "Describe tu estación del año favorita.",
        "¿Cuál es una buena manera de pasar un fin de semana?",
        "Escribe unas frases sobre una ciudad que te guste.",
        "Recomienda algo interesante para aprender.",
    ],
    "Ko": [
        "너의 전형적인 하루에 대해 이야기해줘.",
        "네가 가장 좋아하는 계절을 설명해줘.",
        "주말을 보내는 좋은 방법은 뭐야?",
        "네가 좋아하는 도시에 대해 몇 문장 써줘.",
        "배우기에 흥미로운 것을 추천해줘.",
    ],
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


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B", choices=sorted(VEC_TAG))
    MODEL = parser.parse_args().model
    conc = json.loads(Path("data/inject_concepts.json").read_text())
    pid_of = {c["concept"]: c["pair_id"] for c in conc}
    tok = AutoTokenizer.from_pretrained(MODEL)
    attn = "eager" if "gemma" in MODEL else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation=attn, device_map="auto"
    )
    model.eval()
    device = model.device
    LAYER = round(model.config.num_hidden_layers / 2)
    with safe_open(VEC_ROOT / VEC_TAG[MODEL] / "diff.safetensors", framework="pt") as h:
        diff = h.get_tensor("diff")

    delta_holder: dict[str, torch.Tensor | None] = {"d": None}

    def hook(_m: Any, _i: Any, out: Any) -> Any:
        if delta_holder["d"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        h[:, :, :] += delta_holder["d"].to(h.dtype)
        return out

    handle = model.model.layers[LAYER - 1].register_forward_hook(hook)

    results = []
    agg: dict = collections.defaultdict(lambda: collections.Counter())
    for concept in CONCEPTS:
        pid = pid_of[concept]
        unit = diff[LAYER, pid].float().to(device)
        unit = unit / unit.norm().clamp_min(1e-8)
        for lang, prompts in PROMPTS.items():
            for alpha in ALPHAS:
                dsum = torch.zeros(model.config.vocab_size, device=device)
                npos = 0
                for p in prompts:
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
                    delta_holder["d"] = None
                    with torch.no_grad():
                        base = model(ids, output_hidden_states=True)
                    normL = base.hidden_states[LAYER][0].norm(dim=-1).mean()
                    delta_holder["d"] = (alpha * normL) * unit
                    with torch.no_grad():
                        steer = model(ids)
                    delta_holder["d"] = None
                    d = (steer.logits[0] - base.logits[0]).float()
                    dsum += d[1:].mean(0)
                    npos += 1
                dmean = dsum / npos
                ids_top = torch.topk(dmean, TOPK).indices.tolist()
                toks: list[str] = []
                scr: collections.Counter = collections.Counter()
                for tid in ids_top:
                    s = str(tok.decode([tid]))
                    if "�" in s:
                        sc = "fragment"
                    elif s.strip():
                        cnt: collections.Counter = collections.Counter()
                        for ch in s.strip():
                            o = ord(ch)
                            for name, lo, hi in RANGES:
                                if lo <= o <= hi:
                                    cnt[name] += 1
                                    break
                        sc = cnt.most_common(1)[0][0] if cnt else "symbol"
                    else:
                        sc = "symbol"
                    toks.append(s.strip())
                    scr[sc] += 1
                    agg[(lang, alpha)][sc] += 1
                rec = {"concept": concept, "lang": lang, "alpha": alpha, "top": toks, "scripts": dict(scr)}
                if lang == "En":
                    q = torch.linspace(0, 1, 401, device=dmean.device)
                    rec["quantiles"] = [round(float(v), 4) for v in torch.quantile(dmean, q)]
                results.append(rec)
        print(f"done {concept}", flush=True)
    handle.remove()

    out = Path(f"work/mlsteer/mlsteer-{MODEL.split('/')[-1]}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print("\n=== promoted-token script distribution by context language (pooled over 8 concepts) ===")
    for alpha in ALPHAS:
        print(f"\n-- alpha={alpha} --   (expected = context-language script)")
        for lang in PROMPTS:
            c = agg[(lang, alpha)]
            tot = sum(c.values())
            exp = EXPECT[lang]
            match = c.get(exp, 0) / max(tot, 1)
            top = " ".join(f"{k}:{v}" for k, v in c.most_common(4))
            print(f"  {lang} (expect {exp:8s}): match={match:.0%}  | {top}")


if __name__ == "__main__":
    main()
