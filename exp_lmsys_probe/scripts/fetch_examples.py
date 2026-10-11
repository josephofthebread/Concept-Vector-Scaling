from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

LMSYS = "lmsys/lmsys-chat-1m"
OPENWEBTEXT = "Skylion007/openwebtext"

LMSYS_IDS = {
    "adb4c1d7868443e4b23ed624220f34dd": "honest metacognition vs strategic performance management",
    "9f944ea6aa0442bc931a3b231d4eae72": "admitting uncertainty vs false authority",
    "f67f0108323749f28091c7986de4b621": "emotional availability vs emotional withdrawal",
}
OWT_IDS = {
    "owt-433094": "apology vs revenge",
    "owt-437756": "theism vs atheism",
    "owt-212219": "contingency planning vs improvisation (A)",
    "owt-730755": "contingency planning vs improvisation (B)",
}


def main() -> None:
    wanted = set(LMSYS_IDS)
    root = Path(snapshot_download(LMSYS, repo_type="dataset"))
    found: dict[str, dict] = {}
    for shard in sorted(root.rglob("*.parquet")):
        table = pq.read_table(shard, columns=["conversation_id", "conversation", "language"])
        cids = table.column("conversation_id").to_pylist()
        convs = table.column("conversation").to_pylist()
        langs = table.column("language").to_pylist()
        for cid, turns, lang in zip(cids, convs, langs):
            if cid in wanted and cid not in found:
                oi = next((i for i, t in enumerate(turns) if t["role"] == "user"), None)
                if oi is None:
                    continue
                rep = next((t["content"] for t in turns[oi + 1 :] if t["role"] == "assistant"), "")
                found[cid] = {"label": LMSYS_IDS[cid], "language": lang, "user": turns[oi]["content"], "assistant": rep}
        if len(found) == len(wanted):
            break

    from datasets import load_dataset

    ds = load_dataset(OPENWEBTEXT, split="train")
    owt = {}
    for cid, label in OWT_IDS.items():
        i = int(cid.split("-")[1])
        owt[cid] = {"label": label, "text": ds[i]["text"]}

    data = {"lmsys": found, "openwebtext": owt}
    Path("examples_raw.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
    for cid, d in data["lmsys"].items():
        print(f"LMSYS {cid} [{d['language']}] {d['label']}: user={len(d['user'])}c assistant={len(d['assistant'])}c")
    for cid, d in data["openwebtext"].items():
        print(f"OWT {cid} {d['label']}: {len(d['text'])}c")


if __name__ == "__main__":
    main()
