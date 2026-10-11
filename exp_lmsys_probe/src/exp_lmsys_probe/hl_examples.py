from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from .model import load
from .render import prepare
from .sources import exchange_template

log = logging.getLogger("hl")
LAYER = 23

LMSYS_JOBS = [
    (47, "admitting uncertainty", "false authority", "9f944ea6aa0442bc931a3b231d4eae72"),
    (315, "emotional availability", "emotional withdrawal", "f67f0108323749f28091c7986de4b621"),
    (
        463,
        "honest metacognition about being evaluated",
        "strategic performance management",
        "adb4c1d7868443e4b23ed624220f34dd",
    ),
]
OWT_JOBS = [
    (62, "apology", "revenge", "owt-433094"),
    (936, "theism", "atheism", "owt-437756"),
    (211, "contingency planning", "improvisation", "owt-730755"),
]


def run(corpus: str, model_name: str, jobs: list) -> list[dict]:
    tokenizer, model, vectors, mlayers, _ = load(model_name)
    vi = mlayers.index(LAYER)
    tmpl = exchange_template(tokenizer) if corpus == "lmsys" else ()
    cids = {j[3] for j in jobs}
    if corpus == "lmsys":
        import pyarrow.parquet as pq
        from huggingface_hub import snapshot_download

        from .sources import LMSYS

        root = Path(snapshot_download(LMSYS, repo_type="dataset"))
        texts: dict = {}
        for sh in sorted(root.rglob("*.parquet")):
            table = pq.read_table(sh, columns=["conversation_id", "conversation"])
            for cid, turns in zip(
                table.column("conversation_id").to_pylist(), table.column("conversation").to_pylist()
            ):
                if cid in cids and cid not in texts:
                    oi = next((i for i, x in enumerate(turns) if x["role"] == "user"), None)
                    if oi is None:
                        continue
                    rep = next((x["content"] for x in turns[oi + 1 :] if x["role"] == "assistant"), "")
                    texts[cid] = (turns[oi]["content"], rep)
            if len(texts) == len(cids):
                break
    else:
        from datasets import load_dataset

        from .sources import OPENWEBTEXT

        ds = load_dataset(OPENWEBTEXT, split="train")
        texts = {c: ds[int(c.split("-")[1])]["text"] for c in cids}
    results = []
    with torch.inference_mode():
        for pid, concept, antag, cid in jobs:
            ids, spans, labels = prepare(corpus, tokenizer, cid, texts, tmpl)
            out = model(input_ids=torch.tensor([ids], device=model.device), output_hidden_states=True, use_cache=False)
            hidden = out.hidden_states[LAYER][0].float()
            idx: list[int] = []
            pieces: list[str] = []
            roles: list[str] = []
            cur_role = "user" if corpus == "lmsys" else "doc"
            for (st, ln), lab in zip(spans, labels):
                if lab:
                    cur_role = "assistant" if "assistant" in lab else "user"
                for t in range(st, st + ln):
                    idx.append(t)
                    pieces.append(tokenizer.decode([ids[t]]))
                    roles.append(cur_role)
            content = [i for i in idx if i >= 0]
            unit = torch.nn.functional.normalize(hidden[content], dim=-1)
            cosm = unit @ vectors[vi].T
            z = (cosm - cosm.mean(0)) / cosm.std(0).clamp_min(1e-6)
            zc = {i: v for i, v in zip(content, z[:, pid].cpu().tolist())}
            toks = [{"t": p, "z": round(zc.get(i, 0.0), 4), "role": r} for i, p, r in zip(idx, pieces, roles)]
            results.append(
                {
                    "pair": pid,
                    "cid": cid,
                    "concept": concept,
                    "antagonist": antag,
                    "corpus": corpus,
                    "layer": LAYER,
                    "tokens": toks,
                }
            )
            log.info("%s pair %d %s: %d tokens, max z=%.2f", corpus, pid, cid, len(toks), max(t["z"] for t in toks))
    del model
    torch.cuda.empty_cache()
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = {
        "lmsys": run("lmsys", "Qwen/Qwen3-8B", LMSYS_JOBS),
        "openwebtext": run("openwebtext", "Qwen/Qwen3-8B-Base", OWT_JOBS),
    }
    Path("hl_examples.json").write_text(json.dumps(out, ensure_ascii=False))
    log.info("wrote hl_examples.json")


if __name__ == "__main__":
    main()
