from __future__ import annotations

import time

import numpy as np

from .model import load_selector, load_tokenizer, pad_batch
from .prompt import DEFAULT_CKPT, MAX_LEN, PAD_ID, TOKENIZER_ID


class SRInputRejected(ValueError):
    pass


class Scorer:
    def __init__(self, ckpt: str = DEFAULT_CKPT, tokenizer_id: str = TOKENIZER_ID, max_len: int = MAX_LEN,
                 mb: int = 8, mb_tokens: int | None = 65536, device: str = "cuda", overlength: str = "reject",
                 log=print):
        if overlength not in ("reject", "truncate"):
            raise ValueError(f"overlength must be 'reject' or 'truncate', not {overlength!r}")
        self.log = log
        self.max_len, self.mb, self.mb_tokens, self.device = max_len, mb, mb_tokens or None, device
        self.overlength = overlength
        self.tok = load_tokenizer(tokenizer_id)
        if self.tok.pad_token_id != PAD_ID:
            raise ValueError(f"tokenizer pad id {self.tok.pad_token_id} != {PAD_ID}")
        t0 = time.time()
        self.model = load_selector(ckpt, device, pad_token_id=self.tok.pad_token_id)
        self.log(f"[scorer] loaded {ckpt} in {time.time() - t0:.0f}s; score head "
                 f"{tuple(self.model.score.weight.shape)}; overlength={overlength}")

    def encode(self, prompt: str) -> list[int]:
        if self.overlength == "truncate":
            return self.tok(prompt, truncation=True, max_length=self.max_len)["input_ids"]
        return self.tok(prompt)["input_ids"]

    def check(self, ids: list[int]) -> int:
        n = len(ids)
        if self.overlength == "reject":
            if n > self.max_len:
                raise SRInputRejected(f"{n} tokens > max_len {self.max_len}")
            if PAD_ID in ids:
                raise SRInputRejected(f"pad/eos id {PAD_ID} occurs inside the input")
        return n

    def _forward(self, chunk: list[list[int]]) -> np.ndarray:
        import torch
        inp, att = pad_batch(chunk, self.tok.pad_token_id)
        with torch.no_grad():
            out = self.model(input_ids=inp.to(self.device), attention_mask=att.to(self.device), use_cache=False)
            z = out.logits.squeeze(-1).float().cpu().numpy()
        if not (z.ndim == 1 and z.shape[0] == len(chunk)):
            raise RuntimeError(f"score batch has shape {z.shape} for {len(chunk)} sequences")
        return z

    def _forward_oom_safe(self, chunk: list[list[int]]) -> np.ndarray:
        import torch
        if len(chunk) == 1:
            return self._forward(chunk)
        try:
            return self._forward(chunk)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            h = len(chunk) // 2
            self.log(f"[scorer] CUDA OOM on a micro-batch of {len(chunk)} "
                     f"(max len {max(len(x) for x in chunk)}); halving")
            return np.concatenate([self._forward_oom_safe(chunk[:h]), self._forward_oom_safe(chunk[h:])])

    def score_ids(self, ids_list: list[list[int]]) -> list[float]:
        order = sorted(range(len(ids_list)), key=lambda i: len(ids_list[i]))
        out: list[float | None] = [None] * len(ids_list)
        i = 0
        while i < len(order):
            idx = [order[i]]
            L = len(ids_list[order[i]])
            j = i + 1
            while j < len(order) and len(idx) < self.mb:
                L_next = max(L, len(ids_list[order[j]]))
                if self.mb_tokens and L_next * (len(idx) + 1) > self.mb_tokens:
                    break
                idx.append(order[j])
                L = L_next
                j += 1
            z = self._forward_oom_safe([ids_list[k] for k in idx])
            for k, v in zip(idx, z):
                out[k] = float(v)
            i = j
        return out

    def score_prompts(self, prompts: list[str]) -> tuple[list[float], list[int]]:
        ids = [self.encode(p) for p in prompts]
        n_tokens = [self.check(x) for x in ids]
        return self.score_ids(ids), n_tokens


__all__ = ["SRInputRejected", "Scorer"]
