from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import threading
import time

import requests

from .. import __version__, env

OPENALEX_URL = "https://api.openalex.org/works"
_SELECT = ("id,title,publication_year,cited_by_count,abstract_inverted_index,"
           "primary_location,authorships,fwci")
FAIL_LIMIT = 8


def abstract_text(work: dict) -> str:
    inv = work.get("abstract_inverted_index") or {}
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


class Retriever:
    def __init__(self, cache_path: str, *, api_key: str | None = None, mailto: str | None = None,
                 max_retries: int = 4, fail_limit: int = FAIL_LIMIT, fail_fast: bool = True,
                 failure_flag: str | None = None):
        self.cache_path = cache_path
        self.api_key = api_key if api_key is not None else env.openalex_api_key()
        self.mailto = mailto if mailto is not None else env.openalex_mailto()
        self.max_retries = max_retries
        self.fail_limit = int(fail_limit)
        self.fail_fast = fail_fast
        self.failure_flag = failure_flag
        ua = f"bvsr/{__version__}" + (f" (mailto:{self.mailto})" if self.mailto else "")
        self.headers = {"User-Agent": ua}
        d = os.path.dirname(cache_path)
        if d:
            os.makedirs(d, exist_ok=True)
        self.cache: dict[str, list] = {}
        if os.path.exists(cache_path):
            for line in open(cache_path, errors="ignore"):
                try:
                    rec = json.loads(line)
                    self.cache[rec["k"]] = rec["v"]
                except (ValueError, KeyError, TypeError):
                    pass
        self.used: set[str] = set()
        self._lock = threading.Lock()
        self._consec_fail = 0
        self._budget_warned = False

    def _signal(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} pid={os.getpid()} cache={self.cache_path} {msg}"
        banner = f"\n{'!' * 100}\n!!! RETRIEVAL FAILURE: {line}\n{'!' * 100}\n"
        print(banner, flush=True)
        print(banner, file=sys.stderr, flush=True)
        if self.failure_flag:
            try:
                d = os.path.dirname(self.failure_flag)
                if d:
                    os.makedirs(d, exist_ok=True)
                with open(self.failure_flag, "a") as f:
                    f.write(line + "\n")
            except OSError:
                pass

    @staticmethod
    def _key(query: str, per_page: int) -> str:
        return hashlib.md5(f"{query}|{per_page}".encode()).hexdigest()

    def cached(self, query: str, per_page: int = 25) -> list[dict] | None:
        return self.cache.get(self._key(query, per_page))

    def raw(self, query: str, per_page: int = 25) -> list[dict]:
        k = self._key(query, per_page)
        if k in self.cache:
            return self.cache[k]
        params = {"search": query, "per-page": per_page, "select": _SELECT}
        if self.api_key:
            params["api_key"] = self.api_key
        if self.mailto:
            params["mailto"] = self.mailto
        out, succeeded = [], False
        for a in range(self.max_retries):
            try:
                r = requests.get(OPENALEX_URL, params=params, headers=self.headers, timeout=30)
                if r.status_code == 200:
                    out = r.json().get("results", [])
                    succeeded = True
                    break
                if r.status_code == 429 and "budget" in r.text.lower() and "api_key" in params:
                    if not self._budget_warned:
                        self._budget_warned = True
                        self._signal("OpenAlex key budget exhausted -> falling back to the free pool")
                    del params["api_key"]
                    continue
                if r.status_code in (429, 503):
                    time.sleep(min(30, 3 * 2 ** a) + random.uniform(0, 2))
                    continue
                time.sleep(2)
            except Exception:
                time.sleep(min(20, 2 * 2 ** a))
        if not succeeded:
            with self._lock:
                self._consec_fail += 1
                n = self._consec_fail
            self._signal(f"hard failure {n}/{self.fail_limit} (no HTTP 200 after "
                         f"{self.max_retries} retries) query={query[:80]!r}")
            if self.fail_fast and n >= self.fail_limit:
                self._signal(f"{n} consecutive hard failures: retrieval is down; exiting the "
                             f"process (status 17) rather than generating without literature")
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(17)
            return []
        with self._lock:
            self._consec_fail = 0
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"k": k, "v": out}) + "\n")
            self.cache[k] = out
        return out

    def search(self, query: str, n: int = 1, window: int = 25, seed: int | None = None) -> list[dict]:
        res = self.raw(query, per_page=window)
        cand = []
        for w in res:
            wid = w.get("id")
            if not wid:
                continue
            ab = abstract_text(w)
            if len(ab) < 60:
                continue
            au = [a.get("author", {}).get("display_name", "") for a in (w.get("authorships") or [])[:3]]
            cand.append({"id": wid, "title": w.get("title") or "", "year": w.get("publication_year"),
                         "authors": ", ".join([a for a in au if a]), "abstract": ab,
                         "cites": w.get("cited_by_count", 0), "fwci": w.get("fwci")})
        random.Random(seed).shuffle(cand)
        out = []
        with self._lock:
            for c in cand:
                if c["id"] in self.used:
                    continue
                self.used.add(c["id"])
                out.append(c)
                if len(out) >= n:
                    break
        if not out and cand:
            out = cand[:n]
        return out

    @staticmethod
    def format(p: dict) -> str:
        return (f"PAPER: {p['authors']}, \"{p['title']}\", {p['year']} "
                f"(cited {p['cites']}x)\nABSTRACT: {p['abstract']}")
