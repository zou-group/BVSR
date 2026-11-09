from __future__ import annotations

import argparse
import csv
import json
import math
import os

from ..cli import HelpFormatter
from ..models import INFO_BITS_MODELS
from ..tasks import N_CONTEXTS, TASK_NAMES, get_task

COLUMNS = ["bench", "context", "scorer_model", "task_bits"]

PRE = "Consider the following context given to a research assistant.\n\n"
Q = "\n\nThe underlying benchmark task is: "

MEASURED_WITHOUT_TRAILING_BLANK = {"mls_anomaly", "mls_calibration", "zap_h16", "zap_h32"}

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def task_texts() -> dict:
    with open(os.path.join(_HERE, "tasks", "task_texts.json"), encoding="utf-8") as f:
        return json.load(f)


def measured_context(task: str, i: int) -> str:
    ctx = get_task(task).context(i)
    if task in MEASURED_WITHOUT_TRAILING_BLANK and ctx.endswith("\n\n"):
        ctx = ctx[:-2]
    return ctx


def seq_nll_bits(model, tok, prefix: str, target: str) -> float:
    import torch
    dev = model.device
    p_ids = tok(prefix, return_tensors="pt").input_ids.to(dev)
    t_ids = tok(target, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    ids = torch.cat([p_ids, t_ids], dim=1)
    with torch.inference_mode():
        logits = model(ids).logits[:, :-1]
    lp = torch.log_softmax(logits.float(), dim=-1)
    tgt = ids[:, 1:]
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
    n_t = t_ids.shape[1]
    return float(-tok_lp[-n_t:].sum().item() / math.log(2))


def task_bits(model, tok, ctx: str, desc: str) -> float:
    return seq_nll_bits(model, tok, PRE + Q, desc) - seq_nll_bits(model, tok, PRE + ctx + Q, desc)


def load_model(name: str, device: str = "cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map=device)
    model.eval()
    return tok, model


def read_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: str, rows: list[dict]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(path + ".tmp", path)


def read_bits(path: str, scorers=INFO_BITS_MODELS) -> dict[tuple[str, int], float]:
    import numpy as np
    per: dict[tuple[str, int], dict[str, float]] = {}
    for r in read_rows(path):
        per.setdefault((r["bench"], int(r["context"])), {})[r["scorer_model"]] = float(r["task_bits"])
    bits = {}
    for k, e in per.items():
        vs = [e[m] for m in scorers if m in e]
        if vs:
            bits[k] = float(np.mean(vs))
    return bits


def measure(models, tasks, contexts, out: str, device: str = "cuda") -> list[dict]:
    import torch
    texts = task_texts()
    rows = read_rows(out) if os.path.exists(out) else []
    index = {(r["bench"], str(r["context"]), r["scorer_model"]): r for r in rows}
    for name in models:
        print(f"== loading {name}", flush=True)
        tok, model = load_model(name, device)
        for t in tasks:
            desc = texts[t]["short"]
            for i in contexts:
                b = task_bits(model, tok, measured_context(t, i), desc)
                row = index.get((t, str(i), name))
                if row is None:
                    row = {"bench": t, "context": i, "scorer_model": name}
                    index[(t, str(i), name)] = row
                    rows.append(row)
                row.update(task_bits=round(b, 2))
                print(f"{name.split('/')[-1]:24s} {t:16s} context {i} {b:+8.2f} bits", flush=True)
        del model
        torch.cuda.empty_cache()
        write_rows(out, rows)
    print(f"wrote {out}")
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bvsr.bv.info_bits",
                                 description="Information content of a context in bits "
                                             "(paper Eq. 4 and App. A).",
                                 formatter_class=HelpFormatter)
    ap.add_argument("--out", required=True, help="CSV file (created or updated)")
    ap.add_argument("--models", nargs="+", default=INFO_BITS_MODELS, help="Hugging Face model ids")
    ap.add_argument("--tasks", nargs="+", default=TASK_NAMES, choices=TASK_NAMES)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    measure(a.models, a.tasks, range(N_CONTEXTS), a.out, a.device)


if __name__ == "__main__":
    main()
