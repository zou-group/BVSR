from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field

from .astro import AstroCassiniTask, AstroVoyager2Task
from .base import FULL_CONTEXT, N_CONTEXTS, Task
from .ccc import CCCTask
from .labelproj import LabelProjTask
from .mls_anomaly import MLSAnomalyTask
from .mls_calibration import MLSCalibrationTask
from .parsing import parse_code, parse_search
from .pertpred import PertPredTask
from .scaling_law import SLLrBszTask, SLUShapeTask
from .svg import SVGTask
from .zapbench import ZapH16Task, ZapH32Task

TASK_NAMES = ["svg", "labelproj", "pertpred", "ccc", "mls_anomaly", "mls_calibration",
              "sl_lrbsz", "sl_ushape", "astro_cassini", "astro_voyager2", "zap_h16", "zap_h32"]

TASKS: dict[str, type[Task]] = {c.name: c for c in (
    SVGTask, LabelProjTask, PertPredTask, CCCTask, MLSAnomalyTask, MLSCalibrationTask,
    SLLrBszTask, SLUShapeTask, AstroCassiniTask, AstroVoyager2Task, ZapH16Task, ZapH32Task)}
assert list(TASKS) == TASK_NAMES


def get_task(name: str) -> Task:
    try:
        return TASKS[name]()
    except KeyError:
        raise KeyError(f"unknown task {name!r}; choose from {TASK_NAMES}") from None


def task_spec(name: str) -> str:
    return get_task(name).context(FULL_CONTEXT)


HERE = os.path.dirname(os.path.abspath(__file__))

_ONE_THREAD = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
_FOUR_THREADS = {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}


@dataclass(frozen=True)
class Evaluator:
    task: str
    script: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    gpu: bool = False


EVALUATORS: dict[str, Evaluator] = {e.task: e for e in (
    Evaluator("svg", "eval_svg.py", ("--timeout", "900"), _ONE_THREAD),
    Evaluator("labelproj", "eval_labelproj.py", ("--timeout", "2400"), _ONE_THREAD),
    Evaluator("pertpred", "eval_pertpred.py", ("--timeout", "1500"), _ONE_THREAD),
    Evaluator("ccc", "eval_ccc.py", ("--timeout", "2400"), _ONE_THREAD),
    Evaluator("mls_anomaly", "eval_mls_anomaly.py", (), _FOUR_THREADS),
    Evaluator("mls_calibration", "eval_mls_calibration.py", (), _FOUR_THREADS),
    Evaluator("sl_lrbsz", "eval_scaling_law.py", (), {"SLTASK": "lr_bsz_scaling_law"}),
    Evaluator("sl_ushape", "eval_scaling_law.py", (), {"SLTASK": "easy_question_scaling_law"}),
    Evaluator("astro_cassini", "eval_astro.py", (), {"ASTROTASK": "cassini"}),
    Evaluator("astro_voyager2", "eval_astro.py", (), {"ASTROTASK": "voyager_2"}),
    Evaluator("zap_h16", "eval_zapbench.py", (), {"ZAPTASK": "zap_h16"}, gpu=True),
    Evaluator("zap_h32", "eval_zapbench.py", (), {"ZAPTASK": "zap_h32"}, gpu=True),
)}
assert list(EVALUATORS) == TASK_NAMES


def _get(task: str) -> Evaluator:
    try:
        return EVALUATORS[task]
    except KeyError:
        raise KeyError(f"no evaluator for task {task!r}; choose from {sorted(EVALUATORS)}") from None


def shard_path(task: str) -> str:
    return os.path.join(HERE, _get(task).script)


def shard_command(task: str) -> list[str]:
    return [sys.executable, shard_path(task), *_get(task).args]


_slot = threading.local()


@contextlib.contextmanager
def gpu_slot():
    devices = [d.strip() for d in os.environ.get("BVSR_GPUS", "").split(",") if d.strip()]
    if not devices:
        yield None
        return
    import fcntl
    lock_dir = os.environ.get("BVSR_GPU_LOCK_DIR") or os.path.join(tempfile.gettempdir(),
                                                                   "bvsr_gpu_slots")
    os.makedirs(lock_dir, exist_ok=True)
    fh = None
    while fh is None:
        for dev in devices:
            f = open(os.path.join(lock_dir, f"gpu_{dev}.lock"), "a+")
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                f.close()
                continue
            fh = f
            break
        else:
            time.sleep(2)
    prev = getattr(_slot, "device", None)
    _slot.device = dev
    try:
        yield dev
    finally:
        _slot.device = prev
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()


_SECRETS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "BVSR_SR_KEY")


def _is_secret(name: str) -> bool:
    return name in _SECRETS or name.endswith("_API_KEY")


def shard_env(task: str, cand_dir: str, res_dir: str, shard: int = 0, nshard: int = 1) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not _is_secret(k)}
    env.update({"CAND_DIR": os.path.abspath(cand_dir), "RES_DIR": os.path.abspath(res_dir),
                "SHARD": str(shard), "NSHARD": str(nshard)})
    env.update(_get(task).env)
    dev = getattr(_slot, "device", None)
    if dev is not None:
        env["CUDA_VISIBLE_DEVICES"] = dev
    return env


def run_shard(task: str, cand_dir: str, res_dir: str, *, shard: int = 0, nshard: int = 1,
              timeout: float | None = None, capture: bool = True) -> subprocess.CompletedProcess:
    os.makedirs(res_dir, exist_ok=True)
    return subprocess.run(shard_command(task), env=shard_env(task, cand_dir, res_dir, shard, nshard),
                          capture_output=capture, text=True, timeout=timeout)


__all__ = ["EVALUATORS", "Evaluator", "FULL_CONTEXT", "N_CONTEXTS", "TASKS", "TASK_NAMES", "Task",
           "get_task", "gpu_slot", "parse_code", "parse_search", "run_shard",
           "shard_command", "shard_env", "shard_path", "task_spec"]
