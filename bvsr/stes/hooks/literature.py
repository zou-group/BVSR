from __future__ import annotations

import json
import os
import queue
import threading

from bvsr.retrieval import AgenticSearcher, Retriever, abstract_text
from bvsr.tasks import FULL_CONTEXT, get_task

from . import patch

HEADER = ("Retrieved literature (real published papers, full title and abstract). These come from a "
          "search of the scientific literature and are offered as a source of mechanisms to "
          "transplant; use one of them only if it plausibly improves the program, and ignore them "
          "otherwise.\n\n")

EXCLUDE = 12
TEMPERATURE = 0.9
PREFETCH_THREADS = 2
QUEUE_SIZE = 2
GET_TIMEOUT_S = 900
SEARCH_TIMEOUT_S = 1000


def targeted_prompt(info: str, used: list[str]) -> str:
    ex = list(used)[-EXCLUDE:]
    exs = ("\nPapers already used in this run -- find DIFFERENT papers:\n"
           + "\n".join(f"  - {e}" for e in ex) + "\n") if ex else ""
    return (info
            + "Find real papers whose core mechanisms are most promising to transplant "
              "into a solution to the problem described above.\n" + exs
            + "\nUse the search_literature tool to find real papers. Search as many times "
              "as you find useful, refining your queries as you learn what exists.\n\n"
              "Then reply EXACTLY in this format:\n"
              "PAPER: <authors, title, year -- copied from the chosen entry>\n"
              "MECHANISM: <its ONE core mechanism, in one sentence>\n\n"
              "ONE paper, ONE mechanism -- do NOT combine several. Do NOT write any code.")


def papers_from_queries(retriever, queries: list[str], exclude_ids: set[str], n: int) -> list[dict]:
    papers, seen = [], set()
    for q in queries:
        for w in retriever.cached(q) or []:
            wid, title, ab = w.get("id"), (w.get("title") or "").strip(), abstract_text(w)
            if not wid or wid in seen or not title or not ab or wid in exclude_ids:
                continue
            seen.add(wid)
            authors = ", ".join((a.get("author") or {}).get("display_name", "")
                                for a in (w.get("authorships") or [])[:3])
            papers.append({"id": wid, "title": title, "year": w.get("publication_year"),
                           "authors": authors, "cited_by": w.get("cited_by_count") or 0, "abstract": ab})
    papers.sort(key=lambda p: -p["cited_by"])
    return papers[:n]


def format_block(papers: list[dict]) -> str:
    if not papers:
        return ""
    body = "\n\n".join(f'[{i}] {p["authors"]}, "{p["title"]}", {p["year"]}\n{p["abstract"]}'
                       for i, p in enumerate(papers, 1))
    return HEADER + body + "\n"


def paper_ref(p: dict) -> str:
    return f'{p["authors"]}, "{p["title"]}", {p["year"]}'


class LiteratureSearch:
    def __init__(self, task, retriever, searcher, *, npapers: int = 6, drawlog: str | None = None,
                 start: bool = True):
        self.task, self.retriever, self.searcher = task, retriever, searcher
        self.npapers, self.drawlog = npapers, drawlog
        self.info = task.context(FULL_CONTEXT)
        self.lock = threading.Lock()
        self.used_ids: set[str] = set()
        self.used: list[str] = []
        self.n_draw = 0
        self.q: queue.Queue = queue.Queue(maxsize=QUEUE_SIZE)
        if start:
            for i in range(PREFETCH_THREADS):
                threading.Thread(target=self._worker, daemon=True, name=f"literature-{i}").start()

    def episode(self) -> tuple[list[str], list[dict]]:
        with self.lock:
            prompt = targeted_prompt(self.info, self.used)
        _, queries = self.searcher.run(prompt, temperature=TEMPERATURE)
        with self.lock:
            exclude = set(self.used_ids)
        return queries, papers_from_queries(self.retriever, queries, exclude, self.npapers)

    def _worker(self) -> None:
        while True:
            try:
                ep = (*self.episode(), None)
            except Exception as e:
                ep = ([], [], f"{type(e).__name__}: {str(e)[:200]}")
            self.q.put(ep)

    def block(self) -> str:
        try:
            queries, papers, error = self.q.get(timeout=GET_TIMEOUT_S)
        except queue.Empty:
            queries, papers, error = [], [], f"no search finished within {GET_TIMEOUT_S} s"
        with self.lock:
            self.n_draw += 1
            for p in papers:
                self.used_ids.add(p["id"])
                self.used.append(paper_ref(p))
            rec = {"draw": self.n_draw, "queries": queries,
                   "papers": [{"id": p["id"], "title": p["title"]} for p in papers]}
        if error:
            rec["error"] = error
        if self.drawlog:
            try:
                with open(self.drawlog, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except OSError:
                pass
        return format_block(papers)


def make_build_prompt(search: LiteratureSearch, orig):
    def build_prompt(self, inspirations, failure_patterns=None, policy_context="",
                     shared_construction_summary=None):
        lit = search.block()
        return orig(self, inspirations, failure_patterns,
                    (lit + policy_context) if lit else policy_context, shared_construction_summary)

    return build_prompt


def install(cfg, searcher=None, retriever=None, start: bool = True) -> LiteratureSearch:
    from simpletes.generator import Generator

    if retriever is None:
        retriever = Retriever(os.path.join(cfg.out, "retrieval_cache.jsonl"), fail_fast=False)
    if searcher is None:
        searcher = AgenticSearcher(retriever, model=cfg.search_model, timeout=SEARCH_TIMEOUT_S,
                                   reasoning_effort=cfg.reasoning_effort,
                                   usage_log=os.path.join(cfg.out, "search_usage.jsonl"))
    search = LiteratureSearch(get_task(cfg.task), retriever, searcher, npapers=cfg.npapers,
                              drawlog=os.path.join(cfg.out, "literature_draws.jsonl"), start=start)
    orig = vars(Generator)["build_prompt"]
    patch(Generator, "build_prompt", make_build_prompt(search, orig))
    print(f"[literature] targeted search per step: task={cfg.task} model={cfg.search_model} "
          f"papers/step={cfg.npapers}", flush=True)
    return search


__all__ = ["HEADER", "LiteratureSearch", "format_block", "install", "make_build_prompt", "papers_from_queries",
           "targeted_prompt"]
