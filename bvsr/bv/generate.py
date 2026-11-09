from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import threading
import time

from ..tasks import parse_code, parse_search

STAGE_B_ATTEMPTS = 3


def slug(field: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(field).lower())[:16].strip("_") or "x"


def stem(i: int, field: str) -> str:
    return f"{i:03d}_{slug(field)}"


def write_candidate(rec: dict, cand_dir: str,
                    meta_keys=("stage_a", "search_queries", "thinking")) -> str:
    os.makedirs(cand_dir, exist_ok=True)
    st = stem(rec["i"], rec["area"])
    with open(os.path.join(cand_dir, st + ".py"), "w") as f:
        f.write(rec["code"] + "\n")
    meta = {"area": rec["area"], "paper": rec["paper"], "mechanism": rec["mechanism"]}
    for k in meta_keys:
        if rec.get(k):
            meta[k] = rec[k]
    with open(os.path.join(cand_dir, st + ".meta.json"), "w") as f:
        json.dump(meta, f)
    return st


def load_partial(path: str) -> dict[int, dict]:
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done[d["i"]] = d
                except (ValueError, KeyError, TypeError):
                    continue
    return done


def generate(task, provider, searcher, fields, run_dir, *, context: int,
             temperature: float = 0.8, max_tokens: int = 120000, workers: int = 3) -> list[dict]:
    cand_dir = os.path.join(run_dir, "cand")
    os.makedirs(cand_dir, exist_ok=True)
    partial = os.path.join(run_dir, "partial.jsonl")
    done = load_partial(partial)
    lock = threading.Lock()

    def one(i: int):
        if i in done:
            write_candidate(done[i], cand_dir)
            return done[i]
        field = fields[i]
        try:
            stage_a, queries = searcher.run(
                task.agentic_search_prompt(field, context),
                temperature=temperature, max_tokens=max_tokens)
        except Exception as e:
            print(f"  [{i}] {field} (A agentic search): {str(e)[:64]}", flush=True)
            return None
        paper, mech = parse_search(stage_a)
        if not paper:
            print(f"  [{i}] {field} (A): no PAPER parsed", flush=True)
            return None
        answer, code = None, ""
        for attempt in range(STAGE_B_ATTEMPTS):
            try:
                answer = provider.chat(task.write_prompt(field, paper, mech),
                                       temperature=temperature, max_tokens=max_tokens)
            except Exception as e:
                print(f"  [{i}] {field} (B write, try {attempt + 1}): {str(e)[:64]}", flush=True)
                if attempt + 1 < STAGE_B_ATTEMPTS:
                    time.sleep(5 * (attempt + 1))
                continue
            code = parse_code(answer, task.marker)
            if task.marker in code:
                break
            print(f"  [{i}] {field} (B): marker missing (try {attempt + 1}/{STAGE_B_ATTEMPTS})",
                  flush=True)
            code = ""
        if not code or task.marker not in code:
            print(f"  [{i}] {field} (B): no valid program after {STAGE_B_ATTEMPTS} tries", flush=True)
            return None
        rec = {"i": i, "area": field, "paper": paper, "mechanism": mech, "code": code,
               "thinking": answer, "stage_a": stage_a}
        if queries:
            rec["search_queries"] = queries
        with lock, open(partial, "a") as f:
            f.write(json.dumps(rec) + "\n")
        write_candidate(rec, cand_dir)
        print(f"  ok {i:3d} {field[:30]:30s} <- {(paper or '?')[:44]}", flush=True)
        return rec

    out = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(one, range(len(fields))):
            if r:
                out.append(r)
    out.sort(key=lambda r: r["i"])
    print(f"[bv] {len(out)}/{len(fields)} valid proposals (2 calls each) -> {cand_dir}", flush=True)
    return out


__all__ = ["generate", "load_partial", "slug", "stem", "write_candidate"]
