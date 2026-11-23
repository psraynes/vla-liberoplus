#!/usr/bin/env python3
"""Check the gated PI05 tokenizer dependency without loading the teacher or using a GPU."""
import sys

from huggingface_hub import get_token, hf_hub_download
from transformers import AutoTokenizer


REPO = "google/paligemma-3b-pt-224"


def main() -> None:
    if not get_token():
        raise SystemExit(
            "No Hugging Face login in this container's HF_HOME. "
            "Run `hf auth login` inside the container with the same HF_HOME."
        )
    try:
        hf_hub_download(repo_id=REPO, filename="config.json")
        AutoTokenizer.from_pretrained(REPO)
    except Exception as exc:
        print(f"PI05 tokenizer access check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    print(f"PASS: authenticated access and tokenizer load for {REPO}", flush=True)


if __name__ == "__main__":
    main()
