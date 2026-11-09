from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np

from ..cli import HelpFormatter
from .data import group_runs, load_train, sr_split, top_mask
from .prompt import MAX_LEN, build_prompt


def pairwise_logistic_loss(s, pos):
    import torch
    pos = torch.as_tensor(pos, device=s.device)
    ti = torch.where(pos)[0]
    oj = torch.where(~pos)[0]
    diff = s[ti][:, None] - s[oj][None, :]
    return torch.nn.functional.softplus(-diff).mean()


def two_pass_backward(forward, items, pos, mb: int, scale: float = 1.0) -> float:
    import torch
    with torch.no_grad():
        s_det = forward(items, grad=False)
    s_var = s_det.clone().requires_grad_(True)
    loss = pairwise_logistic_loss(s_var, pos)
    loss.backward()
    g = s_var.grad.detach() * scale
    for i in range(0, len(items), mb):
        z = forward(items[i:i + mb], grad=True)
        (z * g[i:i + mb]).sum().backward()
    return float(loss)


def lr_lambda(total_steps: int):
    warm = max(1, int(0.03 * total_steps))

    def f(t: int) -> float:
        return min(1.0, (t + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, t / max(1, total_steps))))
    return f


def discarded_fraction(pred, scores, frac: float, discard: float) -> float:
    n = len(scores)
    tgt = set(np.where(top_mask(scores, frac))[0].tolist())
    d = int(round(discard * n))
    ranked = np.argsort(-np.asarray(pred), kind="stable")
    disc = set(ranked[n - d:].tolist()) if d else set()
    return len(tgt & disc) / len(tgt)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bvsr.sr.train_sr", formatter_class=HelpFormatter,
                                 description="Train the SR selector (pairwise logistic loss on true top-5% vs rest "
                                             "pairs).")
    ap.add_argument("--data", default="data/sr_data", help="SR dataset directory (train.jsonl.gz)")
    ap.add_argument("--save_dir", required=True, help="output directory of the trained selector")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    ap.add_argument("--top_frac", type=float, default=0.05, help="positives = the run's true top fraction")
    ap.add_argument("--others", type=int, default=40, help="negatives drawn per run and step")
    ap.add_argument("--cells_per_step", type=int, default=2, help="runs per optimizer step")
    ap.add_argument("--passes", type=float, default=1.0, help="passes over the training runs")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--max_len", type=int, default=MAX_LEN, help="inputs are cut at this many tokens")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mb", type=int, default=4, help="micro-batch (sequences); 1 fits smaller GPUs")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    import torch
    from transformers import set_seed
    from .model import load_backbone, load_tokenizer, make_optimizer, pad_batch, save_selector

    set_seed(a.seed)
    rng = random.Random(a.seed)
    rows = load_train(a.data)
    runs = group_runs(rows)
    val_keys, train_keys = sr_split(runs, rng)
    print(f"[sr-train] runs: train={len(train_keys)} val={len(val_keys)} "
          f"({sum(len(runs[k]) for k in train_keys)} / {sum(len(runs[k]) for k in val_keys)} candidates); "
          f"rows in file={len(rows)}; model={a.model}", flush=True)

    tok = load_tokenizer(a.model)
    m = load_backbone(a.model, gradient_checkpointing=True, pad_token_id=tok.pad_token_id).to(a.device)
    opt = make_optimizer(m, a.lr, a.head_lr)
    n_order = int(a.passes * len(train_keys))
    total_steps = int(a.passes * len(train_keys) / a.cells_per_step)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total_steps))

    def encode(rs):
        return [tok(build_prompt(r["spec"], r["code"]), truncation=True, max_length=a.max_len)["input_ids"]
                for r in rs]

    def forward(ids_list, grad):
        out = []
        for i in range(0, len(ids_list), a.mb):
            inp, att = pad_batch(ids_list[i:i + a.mb], tok.pad_token_id)
            z = m(input_ids=inp.to(a.device), attention_mask=att.to(a.device)).logits.squeeze(-1).float()
            out.append(z if grad else z.detach())
        return torch.cat(out)

    def sample_run(key):
        rs = runs[key]
        pos = top_mask([r["score"] for r in rs], a.top_frac)
        top_idx = list(np.where(pos)[0])
        other_idx = [i for i in range(len(rs)) if not pos[i]]
        rng.shuffle(other_idx)
        idx = np.array(top_idx + other_idx[:a.others])
        return [rs[i] for i in idx], pos[idx]

    order: list[str] = []
    for _ in range(int(np.ceil(a.passes))):
        ks = list(train_keys)
        rng.shuffle(ks)
        order += ks
    order = order[:n_order]

    m.train()
    t0, step, losses = time.time(), 0, []
    for s0 in range(0, len(order), a.cells_per_step):
        batch_keys = order[s0:s0 + a.cells_per_step]
        opt.zero_grad(set_to_none=True)
        stats = []
        for key in batch_keys:
            sub, pos = sample_run(key)
            ids = encode(sub)
            loss = two_pass_backward(forward, ids, pos, a.mb, scale=1.0 / len(batch_keys))
            stats.append((loss, len(sub)))
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
        sched.step()
        step += 1
        losses.append(float(np.mean([x[0] for x in stats])))
        if step % 5 == 0 or step == 1:
            el = time.time() - t0
            print(f"[sr-train] step {step}/{total_steps} loss={losses[-1]:.4f} n={[x[1] for x in stats]} "
                  f"{el / 60:.1f} min, eta {el / step * max(0, total_steps - step) / 60:.0f} min", flush=True)

    m.eval()
    save_selector(m, tok, a.save_dir)
    print(f"[sr-train] saved selector -> {a.save_dir}", flush=True)

    def predict(rs):
        ids = encode(rs)
        order_ = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        out = [0.0] * len(ids)
        with torch.no_grad():
            for i in range(0, len(order_), 8):
                idx = order_[i:i + 8]
                z = forward([ids[j] for j in idx], grad=False).cpu().numpy()
                for j, v in zip(idx, z):
                    out[j] = float(v)
        return out

    val = {}
    for frac in (0.05, 0.10):
        for d in (0.3, 0.5):
            val[f"top{int(frac * 100):02d}@{int(d * 100)}"] = []
    for key in val_keys:
        rs = runs[key]
        p = predict(rs)
        sc = [r["score"] for r in rs]
        for frac in (0.05, 0.10):
            for d in (0.3, 0.5):
                val[f"top{int(frac * 100):02d}@{int(d * 100)}"].append(discarded_fraction(p, sc, frac, d))
    val_mean = {k: float(np.mean(v)) for k, v in val.items()}
    print("[sr-train] held-out runs (%d): fraction of the true top-5%% / top-10%% discarded at 30%% / 50%% "
          "discard: %s" % (len(val_keys), " ".join(f"{k}={v:.3f}" for k, v in val_mean.items())), flush=True)
    with open(os.path.join(a.save_dir, "train_summary.json"), "w") as f:
        json.dump({"args": vars(a), "steps": step, "train_runs": len(train_keys), "val_runs": val_keys,
                   "val_discarded": val_mean, "loss_curve": losses}, f, indent=1)


if __name__ == "__main__":
    main()
