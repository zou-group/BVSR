from __future__ import annotations

import argparse

from ..cli import HelpFormatter
from .data import CANDIDATES, eval_cells, load_eval, load_specs, ranking_row, write_jsonl
from .prompt import DEFAULT_CKPT, MAX_LEN, TOKENIZER_ID, build_prompt


def rank_cells(rows: list[dict], preds: list[float], method: str) -> list[dict]:
    pred = {id(r): float(p) for r, p in zip(rows, preds)}
    out = []
    for rs in eval_cells(rows).values():
        ranked = sorted(rs, key=lambda r: -pred[id(r)])
        out.append(ranking_row(method, rs[0]["cell"], [r["id"] for r in ranked]))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bvsr.sr.score", formatter_class=HelpFormatter,
                                 description="Rank every evaluation cell with a trained SR selector (or SFT regressor).")
    ap.add_argument("--data", default="data/sr_data", help=f"SR dataset directory ({CANDIDATES})")
    ap.add_argument("--out", required=True, help="rankings file to write (.jsonl or .jsonl.gz)")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="checkpoint dir or Hugging Face id")
    ap.add_argument("--method", default="SR", help="method name recorded in the rankings")
    ap.add_argument("--tokenizer", default=TOKENIZER_ID)
    ap.add_argument("--max_len", type=int, default=MAX_LEN)
    ap.add_argument("--overlength", default="truncate", choices=["truncate", "reject"])
    ap.add_argument("--mb", type=int, default=4, help="sequences per micro-batch")
    ap.add_argument("--mb_tokens", type=int, default=0, help="padded-token cap per micro-batch (0 = none)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    rows = load_eval(a.data)
    from .scorer import Scorer
    specs = load_specs()
    scorer = Scorer(a.ckpt, tokenizer_id=a.tokenizer, max_len=a.max_len, mb=a.mb, mb_tokens=a.mb_tokens,
                    device=a.device, overlength=a.overlength)
    preds, n_tok = scorer.score_prompts([build_prompt(specs[r["bench"]], r["code"]) for r in rows])
    n_cut = sum(n >= a.max_len for n in n_tok) if a.overlength == "truncate" else 0
    print(f"[sr-score] scored {len(rows)} candidates; longest input {max(n_tok)} tokens; "
          f"{n_cut} at the {a.max_len}-token limit", flush=True)
    out = rank_cells(rows, preds, a.method)
    write_jsonl(a.out, out)
    print(f"[sr-score] {a.method}: {len(out)} cells -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
