from __future__ import annotations


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _single_zero_head(m) -> None:
    import torch
    w = m.score.weight
    if w.shape[0] != 1:
        m.score = torch.nn.Linear(w.shape[1], 1, bias=False).to(w.device, w.dtype)
    m.config.num_labels = 1
    with torch.no_grad():
        m.score.weight.zero_()
        if getattr(m.score, "bias", None) is not None:
            m.score.bias.zero_()


def load_backbone(model_id: str, gradient_checkpointing: bool = True, pad_token_id: int | None = None):
    import torch
    from transformers import AutoModelForSequenceClassification
    if pad_token_id is None:
        pad_token_id = load_tokenizer(model_id).pad_token_id
    m = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=1, dtype=torch.bfloat16,
                                                           trust_remote_code=True)
    m.config.pad_token_id = pad_token_id
    _single_zero_head(m)
    if gradient_checkpointing:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return m


def param_groups(model, lr: float, head_lr: float) -> list[dict]:
    head = [p for n, p in model.named_parameters() if n.startswith("score")]
    body = [p for n, p in model.named_parameters() if not n.startswith("score")]
    return [{"params": body, "lr": lr}, {"params": head, "lr": head_lr}]


def make_optimizer(model, lr: float, head_lr: float):
    import torch
    return torch.optim.AdamW(param_groups(model, lr, head_lr), weight_decay=0.0, betas=(0.9, 0.98))


def save_selector(model, tok, out_dir: str) -> None:
    import os
    os.makedirs(out_dir, exist_ok=True)
    model.config.num_labels = 1
    model.config.id2label = {0: "score"}
    model.config.label2id = {"score": 0}
    model.config.problem_type = "regression"
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)


def load_selector(ckpt: str, device: str = "cuda", pad_token_id: int | None = None):
    import torch
    from transformers import AutoModelForSequenceClassification
    m = AutoModelForSequenceClassification.from_pretrained(ckpt, dtype=torch.bfloat16, trust_remote_code=True)
    m = m.to(device).eval()
    if pad_token_id is not None:
        m.config.pad_token_id = pad_token_id
    return m


def pad_batch(chunk: list[list[int]], pad_id: int):
    import torch
    L = max(len(x) for x in chunk)
    inp = torch.full((len(chunk), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(chunk), L), dtype=torch.long)
    for j, x in enumerate(chunk):
        inp[j, :len(x)] = torch.tensor(x)
        att[j, :len(x)] = 1
    return inp, att


__all__ = ["load_backbone", "load_selector", "load_tokenizer", "make_optimizer", "pad_batch",
           "param_groups", "save_selector"]
