from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time

from bvsr.tasks import EVALUATORS, gpu_slot, run_shard
from bvsr.stes.config import BRIDGE_TIMEOUT

FAILURE = -1e6
EVAL_RETRIES = 3


def _result(rec: dict | None, error: str | None = None) -> dict:
    if rec is None:
        return {"combined_score": FAILURE, "error": error or "no evaluator result"}
    reward = rec.get("reward")
    if reward is None:
        return {"combined_score": FAILURE, "error": str(rec.get("note") or "candidate produced no score")}
    out = {"combined_score": float(reward)}
    for k, v in (rec.get("metrics") or {}).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out.setdefault(k, v)
    return out


def evaluate_with(task: str, program_path: str, use_test_data: bool = False) -> dict:
    timeout = BRIDGE_TIMEOUT[task]
    tries = EVAL_RETRIES + 1
    work = tempfile.mkdtemp(prefix=f"bvsr_{task}_")
    cand, res = os.path.join(work, "cand"), os.path.join(work, "res")
    os.makedirs(cand)
    os.makedirs(res)
    out = os.path.join(res, "prog.json")
    try:
        shutil.copyfile(program_path, os.path.join(cand, "prog.py"))
        cause = ""
        for attempt in range(1, tries + 1):
            try:
                if EVALUATORS[task].gpu:
                    with gpu_slot():
                        proc = run_shard(task, cand, res, timeout=timeout)
                else:
                    proc = run_shard(task, cand, res, timeout=timeout)
                cause = (proc.stderr or "").strip()[-800:]
            except subprocess.TimeoutExpired:
                cause = f"evaluator killed after {timeout} s"
            except OSError as e:
                cause = f"evaluator could not start: {e}"
            if os.path.exists(out):
                try:
                    with open(out) as fh:
                        return _result(json.load(fh))
                except ValueError:
                    os.unlink(out)
            if attempt < tries:
                time.sleep(10 * attempt)
        return dict(_result(None, f"no evaluator result after {tries} attempts"), cause=cause)
    except Exception as e:
        return _result(None, f"bridge error: {type(e).__name__}: {e}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


__all__ = ["BRIDGE_TIMEOUT", "FAILURE", "evaluate_with"]
