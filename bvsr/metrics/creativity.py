from __future__ import annotations

import functools
import json
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
NONOBV_THRESHOLD = 0.25
TOP_FRAC = 0.01


def originality_from_p(p: float, p_min_in_cell: float) -> float:
    n = round(1.0 / p_min_in_cell)
    g = max(1, round(p * n))
    return 1 - math.log(g) / math.log(n) if n > 1 else 1.0


def utility(score: float, base: float, best: float) -> float:
    return float(np.clip((score - base) / max(1e-12, best - base), 0, 1))


def non_obviousness(v: float) -> float:
    return 1.0 if v < NONOBV_THRESHOLD else 0.0


def creativity(op: float, u: float, nv: float) -> float:
    return op * u * nv


def top_mean(values, frac: float = TOP_FRAC) -> float:
    v = np.sort(np.asarray(list(values), dtype=float))[::-1]
    if v.size == 0:
        return float("nan")
    k = max(1, math.ceil(frac * v.size))
    return float(v[:k].mean())


@functools.lru_cache(maxsize=1)
def _anchors_file() -> dict:
    with open(os.path.join(_HERE, "utility_anchors.json"), encoding="utf-8") as f:
        return json.load(f)


def load_anchors() -> dict[str, dict]:
    return {k: dict(v) for k, v in _anchors_file().items()}


def population_anchors(rows: list[dict]) -> dict[str, dict]:
    packaged = load_anchors()
    out = {}
    for b in sorted({r["bench"] for r in rows}):
        scores = [float(r["score"]) for r in rows if r["bench"] == b and math.isfinite(float(r["score"]))]
        bb = packaged.get(b, {}).get("base", 0.0)
        out[b] = {"base": float(bb), "best": max(scores) if scores else float("nan")}
    return out


def add_metrics(rows: list[dict], anchors: dict | None = None) -> list[dict]:
    anchors = anchors or load_anchors()
    cells: dict[tuple, list[dict]] = {}
    for r in rows:
        r["score"], r["p"], r["v"] = float(r["score"]), float(r["p"]), float(r["v"])
        cells.setdefault((r["bench"], r["arm"], r.get("context"), r["model"]), []).append(r)
    for rs in cells.values():
        p_min = min(x["p"] for x in rs)
        for r in rs:
            r["op"] = originality_from_p(r["p"], p_min)
    for r in rows:
        a = anchors[r["bench"]]
        r["u"] = utility(r["score"], a["base"], a["best"])
        r["nv"] = non_obviousness(r["v"])
        r["c"] = creativity(r["op"], r["u"], r["nv"])
    return rows


__all__ = ["NONOBV_THRESHOLD", "TOP_FRAC", "add_metrics", "creativity", "load_anchors",
           "non_obviousness", "originality_from_p", "population_anchors", "top_mean", "utility"]
