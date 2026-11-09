from __future__ import annotations

import argparse
import glob
import json
import math
import os
import threading

from .. import tasks
from ..cli import HelpFormatter
from ..tasks import get_task
from .layout import context_dirs


def read_config(run_dir: str) -> dict:
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: not a context directory written by bvsr.bv.run")
    with open(path) as f:
        return json.load(f)


def score_summary(task_name: str, run_dir: str) -> dict:
    task = get_task(task_name)
    cand = sorted(glob.glob(os.path.join(run_dir, "cand", "*.py")))
    res_dir = os.path.join(run_dir, "res")
    n_res = len(glob.glob(os.path.join(res_dir, "*.json")))
    rewards = [float(r["reward"]) for r in task.collect(res_dir)] if os.path.isdir(res_dir) else []
    rewards = [x for x in rewards if math.isfinite(x)]
    best_baseline = max(task.baselines().values())
    out = {"task": task_name, "candidates": len(cand), "results": n_res, "scored": len(rewards),
           "best_baseline": best_baseline}
    if rewards:
        out.update(mean=sum(rewards) / len(rewards), max=max(rewards),
                   above_baseline=sum(x > best_baseline for x in rewards))
    return out


def print_summary(s: dict) -> None:
    line = (f"[{s['task']}] candidates {s['candidates']}, results {s['results']}, "
            f"scored {s['scored']}")
    if s["scored"]:
        line += (f" | mean {s['mean']:.3f} max {s['max']:.3f} > best baseline "
                 f"({s['best_baseline']:.4g}): {s['above_baseline']}")
    print(line, flush=True)


def run_local(task: str, run_dir: str, workers: int, timeout: float | None = None,
              poll: float = 60.0) -> None:
    cand = os.path.join(run_dir, "cand")
    res = os.path.join(run_dir, "res")
    logs = os.path.join(run_dir, "logs")
    os.makedirs(logs, exist_ok=True)
    gpu = tasks.EVALUATORS[task].gpu

    def shard(i: int) -> None:
        log = os.path.join(logs, f"verify_shard{i}.log")
        try:
            if gpu:
                with tasks.gpu_slot():
                    p = tasks.run_shard(task, cand, res, shard=i, nshard=workers, timeout=timeout)
            else:
                p = tasks.run_shard(task, cand, res, shard=i, nshard=workers, timeout=timeout)
            text, rc = (p.stdout or "") + (p.stderr or ""), p.returncode
        except Exception as e:
            text, rc = f"{type(e).__name__}: {e}\n", -1
        with open(log, "w") as f:
            f.write(text)
        print(f"  shard {i}/{workers} finished rc={rc} (log {log})", flush=True)

    threads = [threading.Thread(target=shard, args=(i,), daemon=True) for i in range(workers)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        for t in threads:
            t.join(timeout=poll / max(1, len(threads)))
        n_res = len(glob.glob(os.path.join(res, "*.json")))
        n_cand = len(glob.glob(os.path.join(cand, "*.py")))
        if any(t.is_alive() for t in threads):
            print(f"  [verify] {n_res}/{n_cand} results", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bvsr.bv.verify",
                                 description="Verify a run directory: run the task's benchmark evaluator over "
                                             "every candidate program.",
                                 formatter_class=HelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--workers", type=int, default=4, help="parallel evaluator processes")
    ap.add_argument("--timeout", type=float, default=None,
                    help="wall-clock limit of one shard in seconds (per-candidate limits are "
                         "enforced inside the evaluator)")
    a = ap.parse_args(argv)

    dirs = context_dirs(a.run_dir)
    if not dirs:
        raise SystemExit(f"{a.run_dir}: no config.json and no context_<k>/config.json; not a run "
                         f"directory written by bvsr.bv.run")
    for d in dirs:
        task = read_config(d)["task"]
        print(f"[verify] task={task} workers={a.workers} run_dir={d}", flush=True)
        run_local(task, d, a.workers, a.timeout)
        print_summary(score_summary(task, d))


if __name__ == "__main__":
    main()
