from __future__ import annotations

import argparse
import os

import numpy as np

from ..cli import HelpFormatter
from ..metrics import top_mean
from ..tasks import FULL_CONTEXT
from .label_creativity import read_candidates
from .layout import BV_ARM

ROWS = [
    ("BV (Limited Information)", lambda r: r["arm"] == BV_ARM),
    ("BV (Full Information)", lambda r: r["arm"] == BV_ARM and r["context"] == FULL_CONTEXT),
    ("Analogical Reasoning", lambda r: r["arm"] == "ar"),
    ("Cross-Domain", lambda r: r["arm"] == "cross_domain"),
    ("No-Domain", lambda r: r["arm"] == "no_domain"),
]
COLUMNS = ["Creativity", "Utility", "Originality", "Non-obviousness"]


def cell_stats(cell: list[dict], frac: float = 0.01) -> tuple:
    return (top_mean([r["c"] for r in cell], frac), top_mean([r["u"] for r in cell], frac),
            np.mean([r["op"] for r in cell]), np.mean([r["nv"] for r in cell]))


def table(rows: list[dict], se_over: str = "cells", frac: float = 0.01, min_cell: int = 20) -> dict:
    out = {}
    for name, keep in ROWS:
        cells: dict[tuple, list[dict]] = {}
        for r in rows:
            if keep(r):
                cells.setdefault((r["model"], r["bench"]), []).append(r)
        cells = {k: v for k, v in cells.items() if len(v) >= min_cell}
        if not cells:
            continue
        stats = {k: cell_stats(v, frac) for k, v in cells.items()}
        if se_over == "cells":
            units = np.array(list(stats.values()))
        else:
            models = sorted({k[0] for k in cells})
            units = np.array([np.mean([stats[k] for k in cells if k[0] == m], axis=0) for m in models])
        se = units.std(axis=0, ddof=1) / np.sqrt(len(units)) if len(units) > 1 else np.zeros(4)
        out[name] = {"mean": units.mean(axis=0).tolist(), "se": se.tolist(), "n": len(units),
                     "cells": len(cells), "median_n": int(np.median([len(v) for v in cells.values()]))}
    return out


def to_markdown(tab: dict, se_over: str) -> str:
    best = [max(v["mean"][j] for v in tab.values()) for j in range(len(COLUMNS))]
    lines = ["| Method | " + " | ".join(f"{c} (↑)" for c in COLUMNS) + " |",
             "|---|" + "---|" * len(COLUMNS)]
    for name, v in tab.items():
        cells = []
        for j, (m, s) in enumerate(zip(v["mean"], v["se"])):
            txt = f"{m:.3f} ± {s:.3f}"
            cells.append(f"**{txt}**" if np.isclose(m, best[j]) else txt)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    unit = "cells (proposer x task)" if se_over == "cells" else "proposers"
    lines.append("")
    lines.append(f"Values are mean ± SE over {unit}; the highest mean of each column is in bold.")
    lines.append("Cells per row (median candidates per cell): "
                 + "; ".join(f"{name} {v['cells']} ({v['median_n']})" for name, v in tab.items()) + ".")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bvsr.bv.table1",
                                 description="Table 1: blind variation against the prompting baselines.",
                                 formatter_class=HelpFormatter)
    ap.add_argument("--candidates", required=True,
                    help="candidates table: candidates.csv of bvsr.bv.label_creativity or the released "
                         "bv_results/candidates.csv.gz")
    ap.add_argument("--out", required=True, help="output markdown file")
    ap.add_argument("--se_over", choices=["cells", "models"], default="cells")
    ap.add_argument("--min_cell", type=int, default=20, help="cells with fewer candidates are left out")
    ap.add_argument("--top_frac", type=float, default=0.01)
    a = ap.parse_args(argv)
    tab = table(read_candidates(a.candidates), a.se_over, a.top_frac, a.min_cell)
    md = to_markdown(tab, a.se_over)
    d = os.path.dirname(a.out)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(a.out, "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
