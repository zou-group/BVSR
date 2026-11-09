from __future__ import annotations

import os
import re

from ..models import PROPOSER_ORDER
from ..tasks import TASK_NAMES

BV_ARM = "bv"
BASELINE_ARMS = ("ar", "cross_domain", "no_domain")
CONTEXT_DIR = "context_{}"
_CONTEXT_RE = re.compile(r"context_(\d+)")


def _alt(names) -> str:
    return "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))


_RUN_RE = re.compile(r"(?P<bench>%s)_(?P<arm>%s)(?:_c(?P<ctx>\d+))?_(?P<model>%s)_(?P<job>[A-Za-z0-9]+)"
                     % (_alt(TASK_NAMES), _alt((BV_ARM,) + BASELINE_ARMS), _alt(PROPOSER_ORDER)))


def context_dir(run_dir: str, k: int) -> str:
    return os.path.join(run_dir, CONTEXT_DIR.format(k))


def is_run(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json"))


def context_dirs(path: str) -> list[str]:
    if is_run(path):
        return [path]
    if not os.path.isdir(path):
        return []
    found = []
    for name in os.listdir(path):
        m = _CONTEXT_RE.fullmatch(name)
        if m and is_run(os.path.join(path, name)):
            found.append((int(m.group(1)), os.path.join(path, name)))
    return [d for _, d in sorted(found)]


def run_name(path: str) -> str:
    p = os.path.normpath(os.path.abspath(path))
    base = os.path.basename(p)
    if _CONTEXT_RE.fullmatch(base):
        return os.path.basename(os.path.dirname(p)) + "/" + base
    return base


def parse_run_id(run: str) -> tuple[str, str, int | None, str]:
    m = _RUN_RE.fullmatch(run)
    if m is None or (m["arm"] == BV_ARM) != (m["ctx"] is not None):
        raise ValueError(f"not a released run id: {run!r}")
    return m["bench"], m["arm"], None if m["ctx"] is None else int(m["ctx"]), m["model"]


def run_labels(path: str) -> set[str]:
    name = run_name(path)
    return {name, name.split("/", 1)[0]}


__all__ = ["BASELINE_ARMS", "BV_ARM", "CONTEXT_DIR", "context_dir", "context_dirs", "is_run", "parse_run_id",
           "run_labels", "run_name"]
