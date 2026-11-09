from __future__ import annotations

import gzip
import json
import os
import random

import numpy as np

from ..bv.layout import parse_run_id

TRAIN = "train.jsonl.gz"
CANDIDATES = "eval_candidates.jsonl.gz"
RANKINGS = "eval_rankings.jsonl.gz"
KEY_FIELDS = ("bench", "arm", "context", "model")


def read_jsonl(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    if path.endswith(".gz"):
        with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
            gz.write(text.encode("utf-8"))
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


def run_fields(run: str) -> dict:
    return dict(zip(KEY_FIELDS, parse_run_id(run)))


def load_specs() -> dict[str, str]:
    from ..tasks import TASK_NAMES, task_spec
    return {t: task_spec(t) for t in TASK_NAMES}


def _dataset_file(data_dir: str, name: str) -> str:
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: --data must be an SR dataset directory "
                         f"(the released one is data/sr_data after bash reproduce.sh)")
    return path


def load_train(data_dir: str) -> list[dict]:
    rows = read_jsonl(_dataset_file(data_dir, TRAIN))
    for r in rows:
        if not all(k in r for k in KEY_FIELDS):
            r.update(run_fields(r["run"]))
    for rs in group_runs(rows).values():
        for r, lab in zip(rs, percentile_labels([x["score"] for x in rs])):
            r["label"] = lab
    specs = load_specs()
    for r in rows:
        r["spec"] = specs[r["bench"]]
    return rows


def load_eval(data_dir: str) -> list[dict]:
    rows = read_jsonl(_dataset_file(data_dir, CANDIDATES))
    for r in rows:
        if "model" not in r or "bench" not in r:
            f = run_fields(r["cell"])
            r["model"], r["bench"] = f["model"], f["bench"]
    return rows


def run_key(r: dict) -> tuple:
    return r["run"], r.get("context")


def _sort_key(k):
    if isinstance(k, tuple):
        return k[0], -1 if k[1] is None else k[1]
    return k


def group_runs(rows: list[dict]) -> dict[tuple, list[dict]]:
    runs: dict[tuple, list[dict]] = {}
    for r in rows:
        runs.setdefault(run_key(r), []).append(r)
    return runs


def eligible_runs(runs: dict, min_n: int = 20, min_distinct: int = 3) -> dict:
    return {k: v for k, v in runs.items() if len(v) >= min_n and len({x["score"] for x in v}) > min_distinct}


def split_runs(keys, val_cells: int = 20, seed: int = 0, rng: random.Random | None = None
               ) -> tuple[list, list]:
    rng = rng if rng is not None else random.Random(seed)
    ks = sorted(keys, key=_sort_key)
    rng.shuffle(ks)
    return ks[:val_cells], ks[val_cells:]


def sr_split(runs: dict, rng: random.Random) -> tuple[list, list]:
    rows = [r for rs in runs.values() for r in rs]
    if not rows or not all("split_sr" in r for r in rows):
        raise SystemExit("the training file must give every row its SR split (field split_sr)")
    marks = {k: {r["split_sr"] for r in rs} for k, rs in runs.items()}
    mixed = [k for k, m in marks.items() if len(m) != 1]
    if mixed:
        raise SystemExit(f"{len(mixed)} runs have rows in different SR splits, e.g. {mixed[0]}")
    order = sorted((k for k, m in marks.items() if m & {"train", "val"}), key=_sort_key)
    rng.shuffle(order)
    return [k for k in order if marks[k] == {"val"}], [k for k in order if marks[k] == {"train"}]


def sft_split(rows: list[dict], seed: int = 0, val_frac: float = 0.05) -> tuple[list[dict], list[dict]]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n_val = int(len(rows) * val_frac)
    return rows[:n_val], rows[n_val:]


def top_mask(scores, frac: float = 0.05) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    n = len(s)
    k = max(1, int(round(frac * n)))
    order = np.argsort(-s, kind="stable")
    m = np.zeros(n, dtype=bool)
    m[order[:k]] = True
    return m


def percentile_labels(scores) -> list[float]:
    sc = np.asarray(scores, dtype=float)
    return [float((sc < s).mean() + 0.5 * (sc == s).mean()) for s in sc]


def eval_cells(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    cells: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        cells.setdefault((r["model"], r["bench"]), []).append(r)
    return cells


def ranking_row(method: str, cell: str, ranked: list[str]) -> dict:
    return {"method": method, "cell": cell, "ranked": list(ranked)}


def read_rankings(paths: list[str], cell_keys: dict[str, tuple[str, str]]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in paths:
        for r in read_jsonl(p):
            if r["cell"] not in cell_keys:
                raise SystemExit(f"{p}: the {r['method']} ranking is for cell {r['cell']!r}, "
                                 f"which is not an evaluation cell")
            out.setdefault(r["method"], {})[cell_keys[r["cell"]]] = list(r["ranked"])
    return out


__all__ = ["CANDIDATES", "KEY_FIELDS", "RANKINGS", "TRAIN", "eligible_runs", "eval_cells",
           "group_runs", "load_eval", "load_specs", "load_train", "percentile_labels", "ranking_row",
           "read_jsonl", "read_rankings", "run_fields", "run_key", "sft_split", "split_runs", "sr_split",
           "top_mask", "write_jsonl"]
