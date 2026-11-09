from __future__ import annotations

import asyncio
import json
import os
import queue
import random
import threading
import time
import uuid

from bvsr.retrieval import FIELDS, AgenticSearcher, Retriever
from bvsr.tasks import get_task, parse_search

from . import patch

CONTEXT = 0
THREADS = 4
QUEUE_SIZE = 80
TEMPERATURE = 0.9
MAX_TRIES = 3
EXCLUDE = 12
SEARCH_TIMEOUT_S = 1000
BLOCK_WAIT_S = 1800

SENTINEL = "<<<BV_BLOCK>>>"
TRANSPLANT = ("# YOUR JOB — transplant ONE mechanism (borrowed from the DISTANT field **{field}**) "
              "into the ACTUAL computation.\n"
              "PAPER: {paper}\n"
              "MECHANISM: {mechanism}\n\n"
              "Implement a REAL transplant of THAT ONE mechanism into the program — not a generic "
              "method wearing its name. ONE mechanism only; do NOT add other tricks.")


def make_block(token: str, field: str, paper: str, mechanism: str) -> str:
    return f"[BV-DRAW {token}]\n" + TRANSPLANT.format(field=field, paper=paper, mechanism=mechanism)


def field_for(seed: int, i: int, attempt: int, fields: list[str] = FIELDS) -> str:
    return random.Random(f"bv:{seed}:{i}:{attempt}").choice(fields)


class BVSearch:
    def __init__(self, task, searcher, *, seed: int = 1, retriever=None, drawlog: str | None = None,
                 state_path: str | None = None, start: bool = True):
        self.task, self.searcher, self.retriever = task, searcher, retriever
        self.seed, self.drawlog, self.state_path = seed, drawlog, state_path
        self.lock = threading.Lock()
        self.n_started = 0
        self.n_consumed = 0
        self.used_by_field: dict[str, list[str]] = {}
        self.q: queue.Queue = queue.Queue(maxsize=QUEUE_SIZE)
        self._load_state()
        if start:
            for i in range(THREADS):
                threading.Thread(target=self._worker, daemon=True, name=f"bv-search-{i}").start()

    def _load_state(self) -> None:
        if not (self.state_path and os.path.exists(self.state_path)):
            return
        try:
            with open(self.state_path, encoding="utf-8") as f:
                st = json.load(f)
            self.n_started = int(st.get("n_started", 0))
            self.n_consumed = int(st.get("n_consumed", 0))
            self.used_by_field = {k: list(v) for k, v in st.get("used_by_field", {}).items()}
        except (OSError, ValueError) as e:
            print(f"[bv] cannot read {self.state_path} ({e}); starting a fresh draw sequence", flush=True)

    def _save_state(self) -> None:
        if not self.state_path:
            return
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"n_started": self.n_started, "n_consumed": self.n_consumed,
                       "used_by_field": self.used_by_field}, f)
        os.replace(tmp, self.state_path)

    def _log(self, rec: dict) -> None:
        if not self.drawlog:
            return
        rec["ts"] = time.time()
        try:
            with open(self.drawlog, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def episode(self, i: int, attempt: int) -> dict:
        field = field_for(self.seed, i, attempt)
        with self.lock:
            ex = list(self.used_by_field.get(field, []))[-EXCLUDE:]
        prompt = self.task.agentic_search_prompt(field, CONTEXT, exclude=ex or None)
        t0 = time.time()
        text, queries = self.searcher.run(prompt, temperature=TEMPERATURE)
        paper, mechanism = parse_search(text)
        rec = {"draw": i, "attempt": attempt, "field": field, "queries": queries, "paper": paper,
               "mechanism": mechanism, "search_s": round(time.time() - t0, 1)}
        if self.retriever is not None:
            rec["openalex_hits"] = sum(1 for q in queries if self.retriever.cached(q))
        if not paper or not mechanism:
            rec.update(event="fail", reason="no PAPER/MECHANISM parsed")
        return rec

    def _worker(self) -> None:
        while True:
            with self.lock:
                i = self.n_started
                self.n_started += 1
            chosen = None
            for attempt in range(MAX_TRIES):
                try:
                    rec = self.episode(i, attempt)
                except Exception as e:
                    rec = {"draw": i, "attempt": attempt, "event": "fail",
                           "reason": f"{type(e).__name__}: {str(e)[:200]}"}
                    time.sleep(5)
                if rec.get("event") == "fail":
                    self._log(rec)
                    continue
                chosen = rec
                break
            if chosen is None:
                time.sleep(2)
                continue
            chosen["tries"] = chosen["attempt"] + 1
            with self.lock:
                self.used_by_field.setdefault(chosen["field"], []).append(chosen["paper"])
            self.q.put(chosen)

    def block(self, gen_id=None, chain_idx=None) -> str:
        t0 = time.time()
        while True:
            try:
                rec = self.q.get(timeout=BLOCK_WAIT_S)
                break
            except queue.Empty:
                print(f"[bv] no search episode finished in {BLOCK_WAIT_S} s; still waiting", flush=True)
        token = uuid.uuid4().hex[:12]
        rec.update(event="draw", token=token, gen_id=gen_id, chain_idx=chain_idx,
                   wait_s=round(time.time() - t0, 1))
        with self.lock:
            self.n_consumed += 1
            self._save_state()
        self._log(rec)
        return make_block(token, rec["field"], rec["paper"], rec["mechanism"])


def make_hooks(search: BVSearch, orig_build_prompt, orig_generate_batch):
    def build_prompt(self, inspirations, failure_patterns=None, policy_context="",
                     shared_construction_summary=None):
        ctx = (SENTINEL + "\n" + policy_context) if policy_context else SENTINEL
        return orig_build_prompt(self, inspirations, failure_patterns, ctx, shared_construction_summary)

    async def _generate_batch(self, task):
        if SENTINEL in task.prompt:
            while True:
                try:
                    blk = await asyncio.to_thread(search.block, task.gen_id, task.chain_idx)
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    print(f"[bv] block() failed: {type(e).__name__}: {str(e)[:200]}; retrying in 30 s",
                          flush=True)
                    await asyncio.sleep(30)
            task.prompt = task.prompt.replace(SENTINEL, blk, 1)
        return await orig_generate_batch(self, task)

    return build_prompt, _generate_batch


def install(cfg, searcher=None, retriever=None, start: bool = True) -> BVSearch:
    from simpletes.engine.core import SimpleTESEngine
    from simpletes.generator import Generator

    if retriever is None:
        retriever = Retriever(os.path.join(cfg.out, "retrieval_cache.jsonl"), fail_fast=False)
    if searcher is None:
        searcher = AgenticSearcher(retriever, model=cfg.search_model, timeout=SEARCH_TIMEOUT_S,
                                   reasoning_effort=cfg.reasoning_effort,
                                   usage_log=os.path.join(cfg.out, "search_usage.jsonl"))
    search = BVSearch(get_task(cfg.task), searcher, seed=cfg.seed, retriever=retriever,
                      drawlog=os.path.join(cfg.out, "bv_draws.jsonl"),
                      state_path=os.path.join(cfg.out, "bv_state.json"), start=start)
    build_prompt, generate_batch = make_hooks(search, vars(Generator)["build_prompt"],
                                              vars(SimpleTESEngine)["_generate_batch"])
    patch(Generator, "build_prompt", build_prompt)
    patch(SimpleTESEngine, "_generate_batch", generate_batch)
    print(f"[bv] blind search per candidate: task={cfg.task} context={CONTEXT} model={cfg.search_model} "
          f"fields={len(FIELDS)} seed={cfg.seed} resume: draws={search.n_started}", flush=True)
    return search


__all__ = ["BVSearch", "SENTINEL", "TRANSPLANT", "field_for", "install", "make_block", "make_hooks"]
