from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ..cli import HelpFormatter
from .data import CANDIDATES, RANKINGS, eval_cells, load_eval, read_rankings

GRID = np.linspace(0, 1.0, 41)
PANELS = [("crea", "Top Creativity"), ("util", "Top Utility")]
DEFAULT_ORDER = "SR,SFT"


def discard_curve(ranked: list[str], target: set, n: int) -> np.ndarray:
    ys = []
    for x in GRID:
        k = int(round(x * n))
        disc = set(ranked[n - k:]) if k else set()
        ys.append(len(target & disc) / len(target) if target else np.nan)
    return np.array(ys)


def targets(ids: list[str], rewards: dict, crea: dict, frac: float) -> tuple[set, set | None]:
    n = len(ids)
    k = max(1, int(round(frac * n)))
    util = set(sorted(ids, key=lambda i: -rewards[i])[:k])
    pos = [i for i in ids if crea.get(i, 0) > 0]
    cr = set(sorted(pos, key=lambda i: -crea[i])[:k]) if len(pos) >= k else None
    return util, cr


def boot_index(n: int, B: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(B)]


def cell_curves(kind: str, ranks: dict, info: dict, frac: float) -> tuple[list[np.ndarray], list]:
    ys, ycells = [], []
    for c in sorted(k for k in ranks if k in info):
        ids, rewards, crea = info[c]
        u, cr = targets(ids, rewards, crea, frac)
        tgt = u if kind == "util" else cr
        if tgt is None:
            continue
        ranked = [i for i in ranks[c] if i in rewards]
        if kind == "crea":
            ranked = [i for i in ranked if i in crea]
            tgt = tgt & set(ranked)
            if not tgt:
                continue
        ys.append(discard_curve(ranked, tgt, len(ranked)))
        ycells.append(c)
    return ys, ycells


def creativity_of(rows: list[dict]) -> dict | None:
    out: dict = {}
    for r in rows:
        if r.get("c") is not None:
            out.setdefault((r["model"], r["bench"]), {})[r["id"]] = float(r["c"])
    return out or None


def evaluate(rows: list[dict], rankings: dict[str, dict], crea: dict | None, frac: float = 0.05,
             B: int = 2000, order: list[str] | None = None) -> dict:
    cells = eval_cells(rows)
    info = {}
    for c, rs in cells.items():
        rewards = {r["id"]: float(r["reward"]) for r in rs}
        info[c] = (list(rewards), rewards, (crea or {}).get(c, {}))
    order = [m for m in (order or []) if m in rankings] + [m for m in rankings if m not in (order or [])]
    panels = []
    for kind, title in PANELS:
        if kind == "crea" and crea is None:
            continue
        elig = [c for c in sorted(info) if kind == "util" or targets(*info[c], frac)[1] is not None]
        methods = {}
        for lab in order:
            ys, ycells = cell_curves(kind, rankings[lab], info, frac)
            if not ys:
                continue
            Y = np.array(ys)
            mean = np.nanmean(Y, 0)
            bm = np.array([np.nanmean(Y[ix], 0) for ix in boot_index(len(ycells), B, seed=0)])
            lo, hi = np.percentile(bm, 2.5, 0), np.percentile(bm, 97.5, 0)
            methods[lab] = {"mean": mean.tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "n_cells": len(ycells)}
        panels.append({"kind": kind, "title": title, "eligible_cells": len(elig),
                       "methods": {m: methods[m] for m in order if m in methods}})
    return {"grid": GRID.tolist(), "frac": frac, "B": B, "order": order, "panels": panels}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bvsr.sr.evaluate", formatter_class=HelpFormatter,
                                 description="Discard curves and bootstrap bands of SR rankings (Fig. 3).")
    ap.add_argument("--data", default="data/sr_data", help=f"SR dataset directory ({CANDIDATES})")
    ap.add_argument("--rankings", nargs="+", default=None,
                    help=f"rankings files (.jsonl or .jsonl.gz); default: the dataset's {RANKINGS}")
    ap.add_argument("--out", required=True, help="output JSON")
    ap.add_argument("--frac", type=float, default=0.05, help="target size as a fraction of the cell")
    ap.add_argument("--B", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--order", default=DEFAULT_ORDER,
                    help="methods first in this order; the others follow in the order of the rankings files")
    a = ap.parse_args(argv)

    rows = load_eval(a.data)
    crea = creativity_of(rows)
    if crea is None:
        print("[sr-eval] no creativity labels in the evaluation candidates: utility panel only", flush=True)
    cell_keys = {r["cell"]: (r["model"], r["bench"]) for r in rows}
    rankings = read_rankings(a.rankings or [os.path.join(a.data, RANKINGS)], cell_keys)
    res = evaluate(rows, rankings, crea, frac=a.frac, B=a.B, order=[m for m in a.order.split(",") if m])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
