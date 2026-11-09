from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os

from ..bv.layout import BV_ARM, context_dirs, run_labels, run_name
from ..cli import HelpFormatter
from ..models import short_name
from ..tasks import FULL_CONTEXT, N_CONTEXTS
from .data import CANDIDATES, KEY_FIELDS, TRAIN, eligible_runs, eval_cells, group_runs, split_runs, write_jsonl

SL_TASKS = ("sl_lrbsz", "sl_ushape")
FAILURE = -1e6
TRAIN_FIELDS = ["run", "score", "split_sr", "code", *KEY_FIELDS]
EVAL_FIELDS = ["cell", "id", "reward", "c", "code", "model", "bench"]


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def run_config(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "config.json")) as f:
        return json.load(f)


def run_candidates(run_dir: str):
    for cp in sorted(glob.glob(os.path.join(run_dir, "cand", "*.py"))):
        stem = os.path.basename(cp)[:-3]
        rp = os.path.join(run_dir, "res", stem + ".json")
        if not os.path.exists(rp):
            continue
        try:
            with open(cp, errors="ignore") as f:
                code = f.read()
            with open(rp) as f:
                res = json.load(f)
        except (OSError, ValueError):
            continue
        if code.strip():
            yield stem, code, res


def eval_reward(task: str, res: dict) -> float | None:
    r = (res.get("metrics") or {}).get("r2_test") if task in SL_TASKS else res.get("reward")
    return float(r) if _finite(r) else None


def train_score(task: str, res: dict) -> float | None:
    m = res.get("metrics") or {}
    if task in SL_TASKS:
        r = m.get("r2_test")
        return float(r) if _finite(r) else FAILURE
    if res.get("reward") is None:
        return None
    if task == "svg" and (m.get("n_distinct") is None or m["n_distinct"] < 100):
        return None
    return float(res["reward"]) if _finite(res["reward"]) else None


def _md5(code: str) -> str:
    return hashlib.md5(code.encode()).hexdigest()


def build_eval(eval_runs: list[str], cap: int, min_cands: int = 40) -> list[dict]:
    rows, seen = [], {}
    for rd in eval_runs:
        cfg = run_config(rd)
        mk, task, cell_id = short_name(cfg["model"]), cfg["task"], run_name(rd)
        if (mk, task) in seen:
            raise SystemExit(f"two evaluation runs for ({mk}, {task}): {seen[(mk, task)]} and {cell_id}")
        seen[(mk, task)] = cell_id
        cell = []
        for stem, code, res in run_candidates(rd):
            r = eval_reward(task, res)
            if r is not None:
                cell.append({"model": mk, "bench": task, "cell": cell_id, "id": stem, "reward": r, "c": None,
                             "code": code})
        if len(cell) < min_cands:
            print(f"[build] skip evaluation run {cell_id}: {len(cell)} verified candidates < {min_cands}", flush=True)
            continue
        rows += cell[:cap]
    rows.sort(key=lambda r: (r["model"], r["bench"]))
    return rows


def build_train(runs: list[str], exclude: set, eval_hash: set):
    rows, stats = [], {"dropped_identical_to_eval": 0, "skipped_runs": []}
    for rd in runs:
        run = run_name(rd)
        if run_labels(rd) & exclude:
            stats["skipped_runs"].append(run)
            continue
        cfg = run_config(rd)
        task = cfg["task"]
        for stem, code, res in run_candidates(rd):
            if f"{run}/{stem}" in exclude:
                continue
            s = train_score(task, res)
            if s is None:
                continue
            if _md5(code) in eval_hash:
                stats["dropped_identical_to_eval"] += 1
                continue
            rows.append({"bench": task, "arm": cfg["arm"], "context": cfg.get("context"), "run": run,
                         "model": short_name(cfg["model"]), "score": s, "code": code})
    return rows, stats


def assign_splits(rows: list[dict], seed: int = 0) -> None:
    val, train = split_runs(eligible_runs(group_runs(rows)), val_cells=20, seed=seed)
    mark = {**{k: "val" for k in val}, **{k: "train" for k in train}}
    for r in rows:
        r["split_sr"] = mark.get((r["run"], r.get("context")), "unused")


def add_creativity(eval_rows: list[dict], table: str) -> None:
    from ..bv.label_creativity import KEY_COLUMNS, LABEL_COLUMNS, _key_of, _read_table
    from ..metrics import add_metrics, population_anchors
    fields, rows = _read_table(table)
    own_keys = all(c in fields for c in KEY_COLUMNS)
    lab = [dict(_key_of(r, own_keys), **{k: r[k] for k in LABEL_COLUMNS})
           for r in rows if all(r.get(k) not in (None, "") for k in LABEL_COLUMNS)]
    anchors = population_anchors([r for r in lab if r["arm"] == BV_ARM])
    lab = [r for r in lab if r["bench"] in anchors]
    add_metrics(lab, anchors=anchors)
    by_id = {r["id"]: r for r in lab}
    for e in eval_rows:
        t = by_id.get(f"{e['cell']}/{e['id']}")
        if t is not None:
            e["c"] = t["c"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bvsr.sr.build_dataset", formatter_class=HelpFormatter,
                                 description="Build an SR dataset (layout of the released SR data) from BV run directories.")
    ap.add_argument("--runs", nargs="+", required=True, help="run directories of bvsr.bv.run")
    ap.add_argument("--out_dir", default="data/sr_built")
    ap.add_argument("--eval_context", type=int, default=FULL_CONTEXT, help="context of the evaluation cells")
    ap.add_argument("--train_contexts", default=",".join(str(k) for k in range(FULL_CONTEXT)),
                    help="contexts whose candidates form the training pool")
    ap.add_argument("--creativity", default="",
                    help="candidates table: candidates.csv of bvsr.bv.label_creativity or the released "
                         "bv_results/candidates.csv.gz")
    ap.add_argument("--exclude", default="", help="text file of run names or run/id entries to leave out")
    ap.add_argument("--cap", type=int, default=112, help="candidates per evaluation cell")
    ap.add_argument("--min_cands", type=int, default=40,
                    help="evaluation runs with fewer verified candidates give no cell")
    ap.add_argument("--seed", type=int, default=0, help="seed of the SR split")
    a = ap.parse_args(argv)
    train_contexts = {int(x) for x in a.train_contexts.split(",") if x.strip()}
    if a.eval_context in train_contexts or not 0 <= a.eval_context < N_CONTEXTS:
        raise SystemExit("--eval_context must be a context index that is not in --train_contexts")

    exclude = set()
    if a.exclude:
        with open(a.exclude) as f:
            exclude = {line.strip() for line in f if line.strip() and not line.startswith("#")}
    eval_dirs, train_dirs = [], []
    for d in a.runs:
        for cd in context_dirs(d):
            ctx = run_config(cd).get("context")
            if ctx == a.eval_context:
                eval_dirs.append(cd)
            elif ctx is None or ctx in train_contexts:
                train_dirs.append(cd)
    ev = build_eval(eval_dirs, a.cap, a.min_cands)
    train, stats = build_train(train_dirs, exclude, {_md5(r["code"]) for r in ev})
    assign_splits(train, a.seed)
    if a.creativity:
        add_creativity(ev, a.creativity)

    os.makedirs(a.out_dir, exist_ok=True)
    write_jsonl(os.path.join(a.out_dir, TRAIN), [{k: r[k] for k in TRAIN_FIELDS} for r in train])
    write_jsonl(os.path.join(a.out_dir, CANDIDATES), [{k: r[k] for k in EVAL_FIELDS} for r in ev])
    stats.update({"n_train": len(train), "n_train_runs": len({(r["run"], r["context"]) for r in train}),
                  "n_eval": len(ev), "eval_cells": len(eval_cells(ev)), "eval_context": a.eval_context,
                  "train_contexts": sorted(train_contexts), "cap": a.cap,
                  "split_sr": {s: sum(r["split_sr"] == s for r in train) for s in ("train", "val", "unused")}})
    with open(os.path.join(a.out_dir, "build_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "skipped_runs"}, indent=1))


__all__ = ["add_creativity", "assign_splits", "build_eval", "build_train", "eval_reward", "train_score"]


if __name__ == "__main__":
    main()
