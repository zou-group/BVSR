from __future__ import annotations

import functools
import glob
import json
import operator
import os
from abc import ABC, abstractmethod

_HERE = os.path.dirname(os.path.abspath(__file__))
N_CONTEXTS = 4
FULL_CONTEXT = N_CONTEXTS - 1


@functools.lru_cache(maxsize=1)
def _contexts() -> dict[str, list[str]]:
    with open(os.path.join(_HERE, "contexts.json"), encoding="utf-8") as f:
        return json.load(f)


@functools.lru_cache(maxsize=None)
def read_upstream(var: str, *parts: str) -> str:
    root = os.environ.get(var)
    if not root:
        raise RuntimeError(f"environment variable {var} is not set; it must point to the upstream "
                           f"checkout that provides {'/'.join(parts)} (bash reproduce.sh --rerun creates it)")
    with open(os.path.join(root, *parts), encoding="utf-8") as f:
        return f.read()


class Task(ABC):
    name: str
    marker: str
    title: str

    @abstractmethod
    def interface(self) -> str: ...

    @abstractmethod
    def baseline_pool(self) -> str: ...

    @abstractmethod
    def baselines(self) -> dict[str, float]:
        pass

    def guidance(self) -> str:
        return ""

    def context(self, i: int) -> str:
        try:
            k = operator.index(i)
        except TypeError:
            k = None
        if k is None or not 0 <= k < N_CONTEXTS:
            raise ValueError(f"context must be an integer in [0, {N_CONTEXTS - 1}], got {i!r}")
        return _contexts()[self.name][k]

    def agentic_search_prompt(self, field: str, context: int = 0, exclude=None) -> str:
        ex = ""
        if exclude:
            ex = ("\nPapers already used for this field in earlier rounds — your answer MUST be a "
                  "DIFFERENT paper, and should not merely restate their mechanism:\n"
                  + "\n".join(f"  - {e}" for e in exclude) + "\n")
        return (self.context(context)
                + f"Find ONE real paper from the field **{field}** whose single core mechanism is most "
                "promising to transplant into a different algorithm.\n"
                + ex
                + "\nUse the search_literature tool to find real papers in this field. Search as many times "
                "as you find useful, refining your queries as you learn what exists.\n\n"
                "Then reply EXACTLY in this format:\n"
                "PAPER: <authors, title, year>\n"
                "MECHANISM: <its ONE core mechanism, in one sentence>\n\n"
                "ONE paper, ONE mechanism — do NOT combine several. Do NOT write any code.")

    job_dash = "—"
    transplant = ("Implement a REAL transplant of THAT ONE mechanism into the search dynamics (selection "
                  "pressure / diversity / fitness shaping / operators) — not a generic method wearing its "
                  "name. ONE mechanism only; do NOT add other tricks.")
    code_format = "<the editable functions this task asks for; no import lines>"

    def write_prompt(self, field: str, paper: str, mechanism: str) -> str:
        return (self.interface() + "\n\n" + self.baseline_pool() + "\n\n"
                + f"# YOUR JOB {self.job_dash} transplant ONE mechanism (borrowed from the DISTANT field "
                  f"**{field}**) into the ACTUAL computation.\n"
                  f"PAPER: {paper}\n"
                  f"MECHANISM: {mechanism}\n\n"
                + self.transplant
                + self.guidance()
                + "\n\nOUTPUT FORMAT (exactly):\n```python\n" + self.code_format + "\n```")

    def collect(self, res_dir: str) -> list[dict]:
        rows = []
        for f in sorted(glob.glob(os.path.join(res_dir, "*.json"))):
            try:
                with open(f) as fh:
                    d = json.load(fh)
            except (ValueError, OSError):
                continue
            if d.get("reward") is not None:
                rows.append({"file": d["file"], "reward": d["reward"], "metrics": d.get("metrics", {})})
        return rows
