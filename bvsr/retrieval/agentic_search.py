from __future__ import annotations

import hashlib
import json
import random
import time

from .. import models
from ..providers import pace, together_client

MAX_USES = 5
MAX_TOKENS = 200000

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_literature",
        "description": ("Search the scientific literature (all disciplines) for real, published papers. "
                        "Returns title, authors, year, citation count and abstract for each hit. "
                        "Call it as many times as useful, refining your query between calls."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "search query (keywords or a phrase)"},
                "n": {"type": "integer", "description": "how many papers to return (default 6, max 15)"},
            },
            "required": ["query"],
        },
    },
}]

BUDGET_EXHAUSTED = "Search budget exhausted. Now give your final answer in the required format."


def query_seed(query: str) -> int:
    return int(hashlib.md5(query.encode()).hexdigest(), 16) % 10000


class AgenticSearcher:
    def __init__(self, retriever, model: str, *, timeout: float = 1000, max_uses: int = MAX_USES,
                 reasoning_effort: str | None = None, usage_log: str | None = None):
        self.retriever = retriever
        self.model = models.resolve(model)[0]
        self.max_uses = max_uses
        self.reasoning_effort = reasoning_effort
        self.usage_log = usage_log
        self.cli = together_client(timeout)

    def _exec_tool(self, args) -> str:
        try:
            a = json.loads(args) if isinstance(args, str) else (args or {})
        except Exception:
            a = {}
        q = (a.get("query") or "").strip()
        n = min(int(a.get("n", 6) or 6), 15)
        if not q:
            return "ERROR: empty query"
        papers = self.retriever.search(q, n=n, seed=query_seed(q))
        if not papers:
            return f"No results for '{q}'. Try different keywords."
        return "\n\n".join(f"[{i + 1}] {self.retriever.format(p)}" for i, p in enumerate(papers))

    def _call(self, **kw):
        if self.reasoning_effort:
            kw = dict(kw, reasoning_effort=self.reasoning_effort)
        for attempt in range(6):
            try:
                pace()
                return self.cli.chat.completions.create(**kw)
            except Exception as e:
                msg = str(e).lower()
                if attempt == 5:
                    raise
                if "429" in msg or "rate" in msg:
                    time.sleep(min(60, 8 * 2 ** attempt) + random.uniform(0, 3))
                elif any(c in msg for c in ("500", "502", "503", "504", "overload", "connection",
                                            "timeout", "timed out")):
                    time.sleep(min(30, 3 * 2 ** attempt) + random.uniform(0, 2))
                else:
                    time.sleep(2)

    def run(self, prompt: str, temperature: float = 0.8, max_tokens: int = MAX_TOKENS) -> tuple[str, list[str]]:
        msgs = [{"role": "user", "content": prompt}]
        trace, uses = [], 0
        for _ in range(self.max_uses + 2):
            kw = dict(model=self.model, messages=msgs, temperature=temperature, max_tokens=max_tokens)
            if uses < self.max_uses:
                kw.update(tools=TOOLS, tool_choice="auto")
            r = self._call(**kw)
            m = r.choices[0].message
            self._log(getattr(r, "usage", None))
            tcs = getattr(m, "tool_calls", None)
            if not tcs:
                return (m.content or ""), trace
            msgs.append({"role": "assistant", "content": m.content or "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name,
                                                      "arguments": c.function.arguments}} for c in tcs]})
            for c in tcs:
                out = self._exec_tool(c.function.arguments)
                try:
                    q = json.loads(c.function.arguments).get("query", "")
                except Exception:
                    q = c.function.arguments
                trace.append(q)
                uses += 1
                msgs.append({"role": "tool", "tool_call_id": c.id, "name": c.function.name,
                             "content": out})
            if uses >= self.max_uses:
                msgs.append({"role": "user", "content": BUDGET_EXHAUSTED})
        r = self._call(model=self.model, messages=msgs, temperature=temperature, max_tokens=max_tokens)
        self._log(getattr(r, "usage", None))
        return (r.choices[0].message.content or ""), trace

    def _log(self, u) -> None:
        if not (self.usage_log and u):
            return
        try:
            with open(self.usage_log, "a") as f:
                f.write(json.dumps({"model": self.model, "prompt": u.prompt_tokens,
                                    "completion": u.completion_tokens}) + "\n")
        except OSError:
            pass
