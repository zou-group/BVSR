from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ..cli import HelpFormatter
from .data import load_train, sft_split
from .prompt import MAX_LEN, build_prompt


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bvsr.sr.train_sft", formatter_class=HelpFormatter,
                                 description="Train the SFT baseline (MSE on the standardized within-run percentile).")
    ap.add_argument("--data", default="data/sr_data", help="SR dataset directory (train.jsonl.gz)")
    ap.add_argument("--save_dir", required=True, help="output directory of the trained regressor")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    ap.add_argument("--epochs", type=float, default=0.6)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--max_len", type=int, default=MAX_LEN)
    ap.add_argument("--val_frac", type=float, default=0.05,
                    help="held-out share of the training candidates")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import scipy.stats as st
    import torch
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, set_seed
    from .model import load_backbone, load_tokenizer, make_optimizer, save_selector

    set_seed(a.seed)
    val, train = sft_split(load_train(a.data), a.seed, a.val_frac)
    n_val = len(val)
    ys = np.array([r["label"] for r in train], dtype=np.float32)
    mu, sd = float(ys.mean()), float(ys.std() + 1e-6)
    print(f"[sft] train={len(train)} val={len(val)} model={a.model} epochs={a.epochs} lr={a.lr} "
          f"bs=1x{a.grad_accum} max_len={a.max_len}", flush=True)

    tok = load_tokenizer(a.model)

    class DS(torch.utils.data.Dataset):
        def __init__(self, rows_, labels):
            self.rows, self.labels = rows_, labels

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]
            e = tok(build_prompt(r["spec"], r["code"]), truncation=True, max_length=a.max_len)
            return {"input_ids": e["input_ids"], "attention_mask": e["attention_mask"],
                    "labels": float((self.labels[i] - mu) / sd)}

    m = load_backbone(a.model, gradient_checkpointing=False, pad_token_id=tok.pad_token_id)
    m.config.problem_type = "regression"
    opt = make_optimizer(m, a.lr, a.head_lr)
    args = TrainingArguments(output_dir=a.save_dir, num_train_epochs=a.epochs, learning_rate=a.lr,
                             per_device_train_batch_size=1, gradient_accumulation_steps=a.grad_accum,
                             bf16=True, logging_steps=50, save_strategy="no", report_to=[], seed=a.seed,
                             warmup_ratio=0.03, lr_scheduler_type="cosine", weight_decay=0.0, max_grad_norm=5.0,
                             gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                             dataloader_num_workers=2, remove_unused_columns=False)
    Trainer(model=m, args=args, train_dataset=DS(train, ys), data_collator=DataCollatorWithPadding(tok),
            optimizers=(opt, None)).train()
    m.eval()
    save_selector(m, tok, a.save_dir)
    print(f"[sft] saved regressor -> {a.save_dir}", flush=True)

    def predict(rows_):
        out = [0.0] * len(rows_)
        order = sorted(range(len(rows_)), key=lambda i: len(rows_[i]["code"]))
        with torch.no_grad():
            for i in range(0, len(order), 8):
                idx = order[i:i + 8]
                e = tok([build_prompt(rows_[j]["spec"], rows_[j]["code"]) for j in idx], truncation=True,
                        max_length=a.max_len, padding=True, return_tensors="pt").to(m.device)
                z = m(**e).logits.squeeze(-1).float().cpu().numpy()
                for j, v in zip(idx, z):
                    out[j] = float(v * sd + mu)
        return out

    rho = float(st.spearmanr(predict(val), [r["label"] for r in val]).statistic) if val else float("nan")
    print(f"[sft] held-out 5% Spearman(prediction, label) = {rho:+.3f}", flush=True)
    with open(os.path.join(a.save_dir, "train_summary.json"), "w") as f:
        json.dump({"args": vars(a), "n_train": len(train), "n_val": n_val, "label_mean": mu, "label_sd": sd,
                   "val_spearman": rho}, f, indent=1)


if __name__ == "__main__":
    main()
