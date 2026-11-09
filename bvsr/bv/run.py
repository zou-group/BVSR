from __future__ import annotations

import argparse
import json
import os

from .. import __version__, models
from ..cli import HelpFormatter
from ..providers import make_provider
from ..retrieval import AgenticSearcher, FIELDS, Retriever
from ..tasks import N_CONTEXTS, TASK_NAMES, get_task
from .generate import generate
from .layout import BV_ARM, context_dir
from .verify import print_summary, score_summary


def resolve_effort(model: str, effort: str | None) -> str | None:
    if effort is None or effort == "auto":
        return models.resolve(model)[1]
    if effort.lower() == "none":
        return None
    return effort


def write_config(run_dir: str, cfg: dict) -> None:
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "config.json")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        for k in ("task", "arm", "context", "model_id"):
            if old.get(k) != cfg.get(k):
                raise SystemExit(f"{path} belongs to a different run ({k}={old.get(k)!r}, "
                                 f"requested {cfg.get(k)!r}); use a new --run_dir")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=1)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bvsr.bv.run",
                                 description="Run blind variation for one task and proposer: 112 two-stage "
                                             "proposals under each of the task's four contexts.",
                                 formatter_class=HelpFormatter)
    ap.add_argument("--task", required=True, choices=TASK_NAMES)
    ap.add_argument("--model", required=True,
                    help=f"proposer: a short name ({', '.join(models.PROPOSER_ORDER)}) or a Together model id")
    ap.add_argument("--run_dir", required=True, help="output directory (one subdirectory per context)")
    ap.add_argument("--n", type=int, default=len(FIELDS),
                    help="number of fields (the first n of the sorted 112-field pool; 112 = all)")
    ap.add_argument("--workers", type=int, default=3, help="proposals generated in parallel")
    ap.add_argument("--max_tokens", type=int, default=120000, help="output-token budget per call")
    ap.add_argument("--timeout", type=int, default=600, help="request timeout in seconds")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--reasoning_effort", default="auto",
                    help="auto (paper setting per model), none, low, medium or high")
    a = ap.parse_args(argv)

    model_id = models.resolve(a.model)[0]
    effort = resolve_effort(a.model, a.reasoning_effort)
    fields = list(FIELDS)[: a.n]
    task = get_task(a.task)
    provider = make_provider(model_id, timeout=a.timeout, reasoning_effort=effort)
    dirs = [context_dir(a.run_dir, k) for k in range(N_CONTEXTS)]
    for k, d in enumerate(dirs):
        write_config(d, {
            "task": a.task, "arm": BV_ARM, "context": k,
            "model": models.short_name(model_id), "model_id": model_id, "reasoning_effort": effort,
            "n": len(fields),
            "settings": {"workers": a.workers, "max_tokens": a.max_tokens, "timeout": a.timeout,
                         "temperature": a.temperature},
            "bvsr_version": __version__})

    for k, d in enumerate(dirs):
        retriever = Retriever(os.path.join(d, "retrieval_cache.jsonl"),
                              failure_flag=os.path.join(d, "RETRIEVAL_FAILURE"))
        searcher = AgenticSearcher(retriever, model_id, timeout=a.timeout, reasoning_effort=effort)
        print(f"[bv] task={a.task} context={k}/{N_CONTEXTS - 1} model={model_id} effort={effort} "
              f"fields={len(fields)} (agentic literature search) -> {d}", flush=True)
        generate(task, provider, searcher, fields, d, context=k,
                 temperature=a.temperature, max_tokens=a.max_tokens, workers=a.workers)
        print_summary(score_summary(a.task, d))


if __name__ == "__main__":
    main()
