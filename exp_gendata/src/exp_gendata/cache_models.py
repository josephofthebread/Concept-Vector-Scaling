from huggingface_hub import snapshot_download

from exp_gendata.config import MODELS


def main() -> None:
    for spec in MODELS:
        path = snapshot_download(
            spec.checkpoint, allow_patterns=["*.json", "*.model", "*.safetensors", "*.txt", "tokenizer.*"]
        )
        print(f"{spec.checkpoint}\t{path}", flush=True)


if __name__ == "__main__":
    main()
