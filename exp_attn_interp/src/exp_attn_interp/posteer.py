from __future__ import annotations

import collections
import json
import logging
import time
from argparse import ArgumentParser, BooleanOptionalAction
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp_attn_interp.config import INSTRUCT_MODEL, vector_tag_for
from exp_attn_interp.vectors import load_diff_vectors

log = logging.getLogger("posteer")
TRIGGER = Path("../exp_trigger_prompts")
ARMS = ("orig", "nomean", "sproj64", "lex0.5", "lex1")


@contextmanager
def steer(model: Any, layer: int, delta: torch.Tensor | None, sub: str = "", where: str = "current"):
    """Add delta to row `layer` (or, with sub='attn', after block layer-1's attention sublayer).
    where: 'current' = last position of each forward call, 'prompt' = prompt positions at prefill, 'all' = everywhere."""
    if delta is None:
        yield
        return

    def hook(_m: Any, _i: Any, output: Any) -> Any:
        residual = output[0] if isinstance(output, tuple) else output
        residual = residual.clone()
        if where == "current":
            residual[:, -1, :] += delta.to(residual.dtype)
        elif where == "all" or residual.shape[1] > 1:
            residual += delta.to(residual.dtype)[:, None, :]
        return (residual, *output[1:]) if isinstance(output, tuple) else residual

    target = model.model.layers[layer - 1].self_attn if sub == "attn" else model.model.layers[layer - 1]
    handle = target.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def ablate_heads(model: Any, spec: str):
    """Scale the output of the listed heads at every position by masking the o_proj input: "block:head" zeroes the head,
    "block:head=s" multiplies it by s (comma-separated)."""
    if not spec:
        yield
        return
    by_block: dict[int, list[tuple[int, float]]] = collections.defaultdict(list)
    for item in spec.split(","):
        bs, hs = item.split(":")
        scale = 0.0
        if "=" in hs:
            hs, sc = hs.split("=")
            scale = float(sc)
        by_block[int(bs)].append((int(hs), scale))
    handles = []
    for blk, heads in by_block.items():
        attn = model.model.layers[blk].self_attn
        hd = int(attn.head_dim)

        def pre(_m: Any, inputs: Any, heads: list[tuple[int, float]] = heads, hd: int = hd) -> Any:
            x = inputs[0].clone()
            for hh, scale in heads:
                x[..., hh * hd : (hh + 1) * hd] *= scale
            return (x, *inputs[1:])

        handles.append(attn.o_proj.register_forward_pre_hook(pre))
    try:
        yield
    finally:
        for hnd in handles:
            hnd.remove()


@contextmanager
def scale_sublayers(model: Any, spec: str, which: str = "mlp"):
    """Scale whole sublayer writes: "block:factor" comma-separated (e.g. 23:0.5,24:0.5), for the MLPs or for the
    attention sublayers. Used to remove rendering capacity above an injection without removing the blocks themselves."""
    if not spec:
        yield
        return
    handles = []
    for item in spec.split(","):
        bs, fs = item.split(":")
        factor = float(fs)

        def post(_m: Any, _i: Any, output: Any, factor: float = factor) -> Any:
            if isinstance(output, tuple):
                return (output[0] * factor, *output[1:])
            return output * factor

        layer = model.model.layers[int(bs)]
        handles.append((layer.mlp if which == "mlp" else layer.self_attn).register_forward_hook(post))
    log.info("scale_%s %s", which, spec)
    try:
        yield
    finally:
        for hnd in handles:
            hnd.remove()


@contextmanager
def ablate_neurons(model: Any, spec: str):
    """Zero MLP neurons at every position, by masking the down_proj input. spec = "path.json:key", the file holding
    {key: {block: [neuron indices]}} (work/tmp/keyed.json: the stance-keyed neurons and a matched random set)."""
    if not spec:
        yield
        return
    path, key = spec.rsplit(":", 1)
    sets = json.loads(Path(path).read_text())[key]
    handles = []
    for blk, idx in sets.items():
        mlp = model.model.layers[int(blk)].mlp
        index = torch.tensor(idx, device=model.device)

        def pre(_m: Any, inputs: Any, index: torch.Tensor = index) -> Any:
            x = inputs[0].clone()
            x[..., index] = 0
            return (x, *inputs[1:])

        handles.append(mlp.down_proj.register_forward_pre_hook(pre))
    log.info("ablate_neurons %s: %s", key, {b: len(v) for b, v in sets.items()})
    try:
        yield
    finally:
        for hnd in handles:
            hnd.remove()


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


ML_PROMPTS: dict[str, list[str]] = {
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
ML_EXPECT = {"En": "Latin", "De": "Latin", "Es": "Latin", "Zh": "Han", "Ru": "Cyrillic", "Ko": "Hangul", "Ar": "Arabic"}
ML_RANGES = [
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


PART_NAMES = ("whole", "emb", "attn", "mlp", "attnlo", "attnhi", "mlplo", "mlphi", "xattn")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--group", default="bad", help="key in work/steer_quartiles.json, or 'all'")
    parser.add_argument("--per-concept", type=int, default=8)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--markers", type=Path, default=Path("data/markers_presence.json"))
    parser.add_argument("--out", type=Path, default=Path("work/posteer"))
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="cap concepts (smoke)")
    parser.add_argument("--every", type=int, default=1, help="every n-th concept of the group")
    parser.add_argument("--pairs", default="", help="explicit comma-separated pair ids (overrides --group)")
    parser.add_argument(
        "--ml",
        action="store_true",
        help="use the paper's 7-language prompts (ML_PROMPTS) and record the foreign-script fraction",
    )
    parser.add_argument(
        "--sub",
        default="",
        choices=["", "attn"],
        help="attn: inject after the attention sublayer of block layer-1 (before its MLP)",
    )
    parser.add_argument(
        "--where",
        default="current",
        choices=["current", "prompt", "all"],
        help="positions that receive the vector: current (the paper's hook), prompt (prefill only = the context path alone), all",
    )
    parser.add_argument(
        "--ablate", default="", help="heads to zero (or scale) during generation, e.g. 22:7,22:19,30:0 or 19:28=2"
    )
    parser.add_argument(
        "--neurons", default="", help="MLP neurons to zero: path.json:key (work/tmp/keyed.json:keyed | :random)"
    )
    parser.add_argument(
        "--mlp-scale",
        default="",
        help="scale whole MLP writes during generation: block:factor,... (e.g. 23:0.5,24:0.5)",
    )
    parser.add_argument(
        "--attn-scale", default="", help="the same for the attention sublayers (the matched control for --mlp-scale)"
    )
    parser.add_argument("--tag", default="", help="suffix for the record key and output file name")
    parser.add_argument(
        "--arm-files",
        default="",
        help="extra arms from files: name=path.safetensors (key 'vectors' [1036, hid] at row --layer), comma-separated",
    )
    parser.add_argument(
        "--context",
        default="",
        choices=["", "guidance", "theme"],
        help="situate the neutral prompt: prepend the pair's ontology narrative guidance (names both poles, "
        "balanced) or only its class theme, as a system message",
    )
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    L = args.layer
    tag = args.model.replace("/", "_")
    norms: dict[int, float] = {}
    for fname in (f"norms-{tag}.json", f"norms-{tag}-1217.json", f"norms-{tag}-late.json"):
        npath = TRIGGER / "work" / "steer" / fname
        if npath.exists():
            norms.update({int(k): float(v) for k, v in json.loads(npath.read_text())["norms"].items()})
    if L not in norms:
        raise SystemExit(f"no calibrated norm for row {L}: {sorted(norms)}")
    quart = json.loads(Path("work/steer_quartiles.json").read_text())
    by_pair: dict[int, list[tuple[int, str]]] = collections.defaultdict(list)
    with (TRIGGER / "work" / "prompts.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            if r["kind"] == "neutral" and r.get("ok") and r.get("prompt"):
                by_pair[int(r["pair_id"])].append((int(r["slot"]), r["prompt"]))
    if args.pairs:
        pair_ids = sorted(int(x) for x in args.pairs.split(","))
    else:
        pair_ids = sorted(by_pair) if args.group == "all" else sorted(int(x) for x in quart[args.group])
    pair_ids = [k for k in pair_ids if len(by_pair[k]) >= args.per_concept][:: args.every]
    if args.limit:
        pair_ids = pair_ids[: args.limit]
    stimuli: list[tuple[int, Any, str]] = [
        (k, slot, p) for k in pair_ids for slot, p in sorted(by_pair[k])[: args.per_concept]
    ]
    if args.ml:
        stimuli = [(k, f"{lang}{i}", p) for k in pair_ids for lang, ps in ML_PROMPTS.items() for i, p in enumerate(ps)]
    system: dict[int, str] = {}
    if args.context:
        from exp_attn_interp.config import ONTOLOGY

        ont = json.loads(ONTOLOGY.read_text())
        for it in ont["pairs"] if isinstance(ont, dict) and "pairs" in ont else ont:
            k = int(it["pair_id"])
            if args.context == "guidance":
                system[k] = (
                    f'Setting: the situation described below involves the contrast between "{it["concept"]}" and '
                    f'"{it["antagonist"]}" ({it["class_name"]}). {it["narrative_guidance"]} Either side may apply.'
                )
            else:
                system[k] = f"Setting: the situation described below concerns {it['class_name']}."
        log.info("context '%s': e.g. %s", args.context, system[pair_ids[0]])
    log.info(
        "%d concepts x %d prompts, row %d, alpha %g, arms %s", len(pair_ids), args.per_concept, L, args.alpha, args.arms
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.offline)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa", local_files_only=args.offline
    ).eval()
    kw: dict[str, Any] = {"enable_thinking": False} if "Qwen3" in args.model else {}
    vectors = load_diff_vectors(device="cpu", tag=vector_tag_for(args.model))
    markers = json.loads(args.markers.read_text())
    rank = 64
    V = vectors[L]
    K, hid = V.shape
    m = unit(V.mean(0))
    Xc = V - V.mean(0)
    _, _, Wt = torch.linalg.svd(Xc, full_matrices=False)
    basis = torch.cat([m[None], Wt[:rank]], 0)
    Q, _ = torch.linalg.qr(basis.T)
    U = model.lm_head.weight.to(torch.float32) * model.model.norm.weight.to(torch.float32)[None, :]
    ugap = torch.zeros_like(V)
    for k in pair_ids:
        ic, ia = markers["concept"].get(str(k), []), markers["antagonist"].get(str(k), [])
        if len(ic) >= 2 and len(ia) >= 2:
            ugap[k] = unit(U[ic].mean(0) - U[ia].mean(0))
    del U
    E = model.model.embed_tokens.weight.to(torch.float32)
    egap = torch.zeros_like(V)
    has = torch.zeros(K, dtype=torch.bool)
    for kk in range(K):
        ic, ia = markers["concept"].get(str(kk), []), markers["antagonist"].get(str(kk), [])
        if len(ic) >= 2 and len(ia) >= 2:
            egap[kk] = unit(E[ic].mean(0) - E[ia].mean(0)).to(V.device)
            has[kk] = True
    del E
    held = torch.zeros(K, dtype=torch.bool)
    held[torch.tensor(pair_ids)] = True
    trn = torch.where(has & ~held)[0]
    Xc = egap[trn] - egap[trn].mean(0)
    _, _, Wt = torch.linalg.svd(Xc, full_matrices=False)
    P = Wt[:rank]
    Z = egap[trn] @ P.T
    lam = 1e-3 * (Z**2).sum() / rank
    Mmap = torch.linalg.solve(Z.T @ Z + lam * torch.eye(rank, device=Z.device), Z.T @ V[trn])
    wordmap = unit((egap @ P.T) @ Mmap)
    try:
        VL = load_diff_vectors(device=str(V.device), tag=vector_tag_for("unsloth/Meta-Llama-3.1-8B-Instruct"))[15]
        Xl = VL[trn] - VL[trn].mean(0)
        _, _, Wl = torch.linalg.svd(Xl, full_matrices=False)
        Pl = Wl[:rank]
        Zl = VL[trn] @ Pl.T
        laml = 1e-3 * (Zl**2).sum() / rank
        Ml = torch.linalg.solve(Zl.T @ Zl + laml * torch.eye(rank, device=Zl.device), Zl.T @ V[trn])
        xllama = unit((VL @ Pl.T) @ Ml)
        log.info(
            "xllama: cos with the Qwen story vector on the chosen concepts: median %.3f",
            float((xllama[torch.tensor(pair_ids)] * V[torch.tensor(pair_ids)]).sum(-1).median()),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("xllama arm unavailable: %s", exc)
        xllama = V.clone()
    log.info(
        "wordmap: fitted on %d other concepts; cos with the story vector on the chosen concepts: median %.3f",
        len(trn),
        float((wordmap[torch.tensor(pair_ids)] * V[torch.tensor(pair_ids)]).sum(-1).median()),
    )
    Q16 = Q[:, :17]
    low = (V @ Q16) @ Q16.T
    arms = {
        "orig": V.clone(),
        "stance16": unit(low),
        "resid16": unit(V - low),
        "wordmap": wordmap,
        "xllama": xllama,
        "pairsum": unit(
            V
            + V[
                torch.tensor(
                    [pair_ids[(pair_ids.index(kk) + 1) % len(pair_ids)] if kk in pair_ids else kk for kk in range(K)]
                )
            ]
        ),
        **{f"vrow{R}": vectors[R].clone() for R in (12, 16, 18, 19, 20, 22, 24, 28, 35) if R < vectors.shape[0]},
        "nomean": unit(V - (V @ m)[:, None] * m[None]),
        f"sproj{rank}": unit((V @ Q) @ Q.T),
        "lex0.5": unit(V + 0.5 * ugap),
        "lex1": unit(V + 1.0 * ugap),
        "ugap": ugap.clone(),
        "egap": egap.to(V.device).clone(),
    }
    workdir = "work" if "Qwen" in model.config.name_or_path else f"work-{model.config.name_or_path.split('/')[-1]}"
    langvec = Path(f"{workdir}/langvec-L{L}.safetensors")
    if langvec.exists():
        for lang, t in load_file(str(langvec)).items():
            if lang != "pairs":
                vl = unit(t.to(V.device, torch.float32))
                arms[f"lang_{lang}"] = vl
                arms[f"langres_{lang}"] = unit(vl - (vl @ Q16) @ Q16.T)
                arms[f"langst_{lang}"] = unit((vl @ Q16) @ Q16.T)
        arms["langavg"] = unit(torch.stack([arms[f"lang_{lang}"] for lang in ("en", "de", "zh", "ru", "es")]).sum(0))
    genrevec = Path(f"{workdir}/genrevec-L{L}.safetensors")
    if genrevec.exists():
        for g, t in load_file(str(genrevec)).items():
            if g != "pairs":
                arms[f"genre_{g}"] = unit(t.to(V.device, torch.float32))
    try:
        from transformers import AutoTokenizer as _AT

        from exp_attn_interp.config import ONTOLOGY

        _tok = _AT.from_pretrained(model.config.name_or_path, local_files_only=True)
        ont = json.loads(ONTOLOGY.read_text())
        items = ont["pairs"] if isinstance(ont, dict) and "pairs" in ont else ont
        Uw = model.lm_head.weight.to(torch.float32) * model.model.norm.weight.to(torch.float32)[None, :]
        stop = set(
            "the a an of and or to in on at for with is are was were be been being by as it its this that these those i you he she "
            "they we them their our your my me him her his one two either side may apply about whether while another not no but so "
            "if than then there here when which who whom what how do does did have has had someone person people".split()
        )
        sitv = torch.zeros_like(V)
        per_ids: dict[int, list[int]] = {}
        for it in items:
            kk = int(it["pair_id"])
            ids = [
                int(t)
                for t in _tok(it["narrative_guidance"], add_special_tokens=False)["input_ids"]
                if str(_tok.decode([t])).strip().lower().strip('".,:;()') not in stop
                and str(_tok.decode([t])).strip().isalpha()
            ]
            if ids:
                per_ids[kk] = ids
                sitv[kk] = unit(Uw[ids].mean(0)).to(V.device)
        _g = torch.Generator().manual_seed(20260907)
        pool = sorted({t for v in per_ids.values() for t in v})
        randv, misv = torch.zeros_like(V), torch.zeros_like(V)
        kks = sorted(per_ids)
        perm = torch.randperm(len(kks), generator=_g).tolist()
        perm = [p if p != i else (p + 1) % len(kks) for i, p in enumerate(perm)]
        for i, kk in enumerate(kks):
            n = len(per_ids[kk])
            pick = [pool[j] for j in torch.randperm(len(pool), generator=_g)[:n].tolist()]
            randv[kk] = unit(Uw[pick].mean(0)).to(V.device)
            misv[kk] = sitv[kks[perm[i]]]
        del Uw
        arms["sitonly"] = sitv
        arms["sit0.5"] = unit(V + 0.5 * sitv)
        arms["sit1"] = unit(V + 1.0 * sitv)
        arms["sitanti0.5"] = unit(-V + 0.5 * sitv)
        arms["sitanti1"] = unit(-V + 1.0 * sitv)
        arms["randbag1"] = unit(V + randv)
        arms["mismatch1"] = unit(V + misv)
        arms["randonly"] = randv
        arms["misonly"] = misv
    except Exception as exc:  # noqa: BLE001
        log.warning("situation arms unavailable: %s", exc)
    for pres in ("user", "named_assistant", "paired_assistant"):
        try:
            arms[f"pres_{pres}"] = load_diff_vectors(
                device=str(V.device), tag=vector_tag_for(model.config.name_or_path, pres)
            )[L].clone()
        except Exception as exc:  # noqa: BLE001
            log.warning("presentation arm %s unavailable: %s", pres, exc)
    out = {f"p_{n}": torch.zeros(K, hid) for n in PART_NAMES}
    found = 0
    for k in pair_ids:
        acc: dict[str, torch.Tensor] | None = None
        for d in ("vdecomp", "vdecomp2"):
            f = Path(f"{workdir}/{d}/pair-{k:04d}.safetensors")
            if not f.exists():
                continue
            t = load_file(str(f))
            heads = (t["pole_head"][0] - t["pole_head"][1]).float()
            mlp = (t["pole_mlp"][0] - t["pole_mlp"][1]).float()
            cur = {
                "whole": (t["pole_resid"][0, L] - t["pole_resid"][1, L]).float(),
                "emb": (t["pole_embed"][0] - t["pole_embed"][1]).float(),
                "attn": heads[:L].sum((0, 1)),
                "mlp": mlp[:L].sum(0),
                "attnlo": heads[:12].sum((0, 1)),
                "attnhi": heads[12:L].sum((0, 1)),
                "mlplo": mlp[:12].sum(0),
                "mlphi": mlp[12:L].sum(0),
            }
            acc = cur if acc is None else {n: acc[n] + cur[n] for n in cur}
        if acc is None:
            continue
        found += 1
        vk = V[k].cpu()
        acc["xattn"] = acc["attn"] - (acc["attn"] @ vk) * vk
        for n, v in acc.items():
            out[f"p_{n}"][k] = v
    log.info("part arms: decomposition found for %d/%d concepts", found, len(pair_ids))
    arms.update({n: unit(v).to(V.device) for n, v in out.items()})
    ks = torch.tensor(pair_ids)
    log.info(
        "arm cosines with orig (median over the chosen concepts): %s",
        {a: round(float((arms[a][ks] * V[ks]).sum(-1).median()), 3) for a in arms},
    )
    log.info(
        "cos(v, u_gap) median %.3f; in-subspace energy median %.3f",
        float((V[ks] * ugap[ks]).sum(-1).median()),
        float(((V[ks] @ Q) ** 2).sum(-1).median()),
    )
    for spec in [s for s in args.arm_files.split(",") if s]:
        name, path = spec.split("=", 1)
        arms[name] = unit(load_file(path)["vectors"].to(vectors.device, torch.float32))
        log.info(
            "arm %s from %s: median cosine with orig over the chosen concepts %.3f",
            name,
            path,
            float((arms[name][torch.tensor(pair_ids)] * vectors[L][torch.tensor(pair_ids)]).sum(-1).median()),
        )
    save_file(
        {a: v[torch.tensor(pair_ids)].contiguous() for a, v in arms.items()} | {"pair_ids": torch.tensor(pair_ids)},
        str(args.out / f"vectors-L{L}.safetensors"),
    )
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    batch_size = args.batch_size or (320 if total_gib > 100 else 160 if total_gib > 60 else 64)
    log.info("device %s (%.0f GiB) -> batch %d", torch.cuda.get_device_name(0), total_gib, batch_size)
    started = time.monotonic()
    for arm in args.arms.split(","):
        path = args.out / f"{arm}{args.tag}.jsonl"
        done = set()
        if path.exists():
            with path.open() as fh:
                done = {json.loads(line)["key"] for line in fh}
        wtag = "" if args.where == "current" else f"_{args.where}"
        pending = [
            (k, slot, p)
            for k, slot, p in stimuli
            if f"{k}:{slot}:L{L}{args.sub}{wtag}{args.tag}:a{args.alpha:g}" not in done
        ]
        log.info("arm %s: %d pending (%d done)", arm, len(pending), len(done))
        with path.open("a") as sink:
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                rendered = [
                    tokenizer.apply_chat_template(
                        ([{"role": "system", "content": system[k]}] if k in system else [])
                        + [{"role": "user", "content": p}],
                        tokenize=False,
                        add_generation_prompt=True,
                        **kw,
                    )
                    for k, _, p in batch
                ]
                enc = tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
                prompt_len = enc.input_ids.shape[1]
                delta = torch.stack([arms[arm][k] for k, _, _ in batch]).to(model.device) * (args.alpha * norms[L])
                with (
                    torch.inference_mode(),
                    ablate_heads(model, args.ablate),
                    ablate_neurons(model, args.neurons),
                    scale_sublayers(model, args.mlp_scale, "mlp"),
                    scale_sublayers(model, args.attn_scale, "attn"),
                    steer(model, L, delta, args.sub, args.where),
                ):
                    output = model.generate(
                        **enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
                    )
                for row, (k, slot, p) in enumerate(batch):
                    gen = output[row, prompt_len:]
                    rec = {
                        "key": f"{k}:{slot}:L{L}{args.sub}{wtag}{args.tag}:a{args.alpha:g}",
                        "pair_id": k,
                        "slot": slot,
                        "sub": args.sub,
                        "ablate": args.ablate,
                        "neurons": args.neurons,
                        "mlp_scale": args.mlp_scale,
                        "attn_scale": args.attn_scale,
                        "where": args.where,
                        "model": args.model,
                        "layer": L,
                        "alpha": args.alpha,
                        "norm": norms[L],
                        "arm": arm,
                        "prompt": p,
                        "context": args.context,
                        "system": system.get(k, ""),
                        "response": tokenizer.decode(gen, skip_special_tokens=True),
                        "gen_tokens": int((gen != tokenizer.pad_token_id).sum()),
                    }
                    if args.ml:
                        lang = str(slot)[:2]
                        ids = [int(t) for t in gen.tolist() if t != tokenizer.pad_token_id]
                        rec["lang"] = lang
                        scr: collections.Counter = collections.Counter()
                        for tid in ids:
                            c: collections.Counter = collections.Counter()
                            for ch in str(tokenizer.decode([tid])).strip():
                                o = ord(ch)
                                for name, lo, hi in ML_RANGES:
                                    if lo <= o <= hi:
                                        c[name] += 1
                                        break
                            if c:
                                scr[c.most_common(1)[0][0]] += 1
                        ntot = sum(scr.values())
                        rec["foreign_frac"] = round(1 - scr.get(ML_EXPECT[lang], 0) / ntot, 3) if ntot else 0.0
                    sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sink.flush()
                log.info(
                    "arm %s: %d/%d written, %.1f min",
                    arm,
                    min(start + batch_size, len(pending)),
                    len(pending),
                    (time.monotonic() - started) / 60,
                )
    log.info("done, %.1f min", (time.monotonic() - started) / 60)


if __name__ == "__main__":
    main()
