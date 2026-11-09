from __future__ import annotations

MAX_LEN = 36736
PAD_ID = 248044
TOKENIZER_ID = "Qwen/Qwen3.5-9B-Base"
MODEL_TAG = "sr-judger"
DEFAULT_CKPT = "Xkev/SR-judger"


def build_prompt(spec: str, code: str) -> str:
    return f"### TASK\n{spec}\n\n### IMPLEMENTATION\n{code}\n\n### PREDICTED PERCENTILE OF VERIFIED SCORE:"


__all__ = ["DEFAULT_CKPT", "MAX_LEN", "MODEL_TAG", "PAD_ID", "TOKENIZER_ID", "build_prompt"]
