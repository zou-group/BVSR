from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import glob
import gzip
import hashlib
import json
import math
import os
import re
import threading
import time

from ..cli import HelpFormatter
from ..metrics import add_metrics, load_anchors, population_anchors
from ..models import LABEL_MODELS
from ..tasks import get_task
from .layout import BV_ARM, context_dirs, parse_run_id, run_labels, run_name

STAGES = ("collect", "judge", "group", "assemble")
KEY_COLUMNS = ["bench", "arm", "context", "proposer"]
LABEL_COLUMNS = ["score", "p", "v"]
METRIC_COLUMNS = ["op", "u", "nv", "c"]
CANDIDATE_COLUMNS = ["candidate_id"] + KEY_COLUMNS + LABEL_COLUMNS + METRIC_COLUMNS

JUDGE_PROMPT = """You are grading how much PRIOR EVIDENCE OF UTILITY a research proposal
already had at the moment it was made.

TASK: {task_desc}

THE PAPER THE PROPOSER WAS SHOWN:
Title: {title}
Abstract: {abstract}

THE PROPOSAL (the proposer's stated mechanism):
{mechanism}

First, state in ONE sentence the PROBLEM CLASS of the task above — what kind of problem
must be solved, in your own words. Then judge: to what degree does the shown paper ALREADY
establish that this mechanism would be effective for THAT PROBLEM CLASS? Evidence counts
only if the paper demonstrates the mechanism on the task's problem class or a closely
related one. Success demonstrated ONLY in the paper's own, unrelated domain is NOT evidence
for this task and CAPS the score at 0.4. Score 0.0-1.0:
- 1.0 = the paper directly demonstrates this exact method succeeding on this
  problem class; the proposer essentially knew it would work (direct transfer).
- 0.7 = the paper shows this method working on a closely related problem;
  strong prior evidence, modest adaptation needed.
- 0.4 = the paper suggests the core idea but in a distant domain or only as a
  component; real uncertainty remained about whether it transfers.
- 0.1 = the paper at most inspired an ingredient or an analogy; the proposer
  could not know from it whether the mechanism works here.
- 0.0 = the mechanism's effectiveness cannot be inferred from this paper.
Judge ONLY what the paper itself establishes. Do NOT use your own knowledge of
whether the method actually works, and do NOT grade how good the proposal is.
Reply with one JSON object only:
{{"problem_class": "<one sentence>", "v_lit": <float>, "reason": "<one sentence>"}}"""

GROUP_PROMPT = """Below are {n} one-sentence descriptions of the core mechanism of research proposals,
each identified by its index. Group together the indices whose descriptions express the SAME
core method or idea -- paraphrases of one technique, the same named algorithm, or the same
mathematical mechanism -- even when the wording differs. Different applications of the same core
method belong in one group. Genuinely different methods stay apart. Do not force groups: leave
unique ideas ungrouped.

{items}

Think through the grouping carefully first: compare the descriptions against each other and
reason about which ones share one core method (watch for the same named algorithm appearing
under different applications). Before the fence, state one short line per group naming the
shared method. Then give your FINAL ANSWER as one JSON object wrapped in a ```json code
fence, listing every group of size >= 2:
```json
{{"groups": [[i, j, ...], ...]}}
```"""


_TASK_DESC: dict[str, str] = {}


def task_desc(bench: str) -> str:
    if bench not in _TASK_DESC:
        _TASK_DESC[bench] = " ".join(get_task(bench).interface().split())
    return _TASK_DESC[bench]


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def title_of(citation: str) -> str:
    m = re.search(r'"([^"]+)"', citation or "")
    return m.group(1) if m else (citation or "")


def de_invert(aii) -> str:
    if not isinstance(aii, dict) or not aii:
        return ""
    pos = []
    for w, ps in aii.items():
        for p in ps:
            pos.append((p, w))
    return " ".join(w for _, w in sorted(pos))


def work_abstract(work: dict) -> str:
    return work.get("abstract") or de_invert(work.get("abstract_inverted_index"))


def build_abstract_index(run_dirs) -> dict[str, str]:
    paths = [os.path.join(d, "retrieval_cache.jsonl") for d in run_dirs]

    def read_one(path):
        out = {}
        if not os.path.exists(path):
            return out
        with open(path, errors="ignore") as f:
            for line in f:
                try:
                    works = json.loads(line).get("v")
                except (ValueError, AttributeError):
                    continue
                for w in works if isinstance(works, list) else []:
                    if not isinstance(w, dict):
                        continue
                    ti, ab = w.get("title"), work_abstract(w)
                    if ti and ab:
                        out.setdefault(norm_title(ti), ab)
        return out

    idx: dict[str, str] = {}
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        for part in ex.map(read_one, paths):
            for k, v in part.items():
                idx.setdefault(k, v)
    return idx


class CallCache:
    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        self.data: dict[str, str] = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        self.data[d["key"]] = d["text"]
                    except (ValueError, KeyError):
                        continue

    @staticmethod
    def key(model: str, effort: str, max_tokens: int, prompt: str) -> str:
        return hashlib.sha256(f"{model}|{effort}|{max_tokens}|{prompt}".encode()).hexdigest()

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def put(self, key: str, model: str, text: str) -> None:
        with self.lock:
            self.data[key] = text
            with open(self.path, "a") as f:
                f.write(json.dumps({"key": key, "model": model, "text": text}) + "\n")


def make_caller(kind: str, cache: CallCache):
    spec = LABEL_MODELS[kind]
    holder = [None]
    lock = threading.Lock()

    def get_provider():
        with lock:
            if holder[0] is None:
                from ..providers import OpenAIProvider
                holder[0] = OpenAIProvider(spec["id"], reasoning_effort=spec["reasoning_effort"], max_retries=3)
            return holder[0]

    def call(prompt: str) -> str:
        k = CallCache.key(spec["id"], spec["reasoning_effort"], spec["max_tokens"], prompt)
        hit = cache.get(k)
        if hit is not None:
            return hit
        prov = get_provider()
        for attempt in range(5):
            try:
                r = prov.complete(prompt, max_tokens=spec["max_tokens"])
                if r.finish_reason == "length":
                    raise RuntimeError("finish_reason=length")
                text = r.text or ""
                if text:
                    cache.put(k, spec["id"], text)
                return text
            except Exception:
                if attempt == 4:
                    return ""
                time.sleep(6 * (attempt + 1))
        return ""

    return call


def _read_json(path: str):
    with open(path) as f:
        return json.load(f)


def read_exclude(path: str) -> set[str]:
    if not path:
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip() and not line.startswith("#")}


def cell_order(key: tuple) -> tuple:
    bench, arm, context, model = key
    return bench, arm, -1 if context is None else context, model


def collect(run_dirs, exclude: set[str] = frozenset()) -> list[dict]:
    by_cell: dict[tuple, list[str]] = {}
    for d in sorted(set(run_dirs)):
        for cd in context_dirs(d):
            if run_labels(cd) & exclude:
                continue
            cfg = _read_json(os.path.join(cd, "config.json"))
            by_cell.setdefault((cfg["task"], cfg["arm"], cfg.get("context"), cfg["model"]), []).append(cd)
    rows, used_dirs = [], []
    for (bench, arm, context, model), dirs in sorted(by_cell.items(), key=lambda kv: cell_order(kv[0])):
        task = get_task(bench)
        cell: list[dict] = []
        for d in dirs:
            run = run_name(d)
            res_dir = os.path.join(d, "res")
            rewards = {}
            if os.path.isdir(res_dir):
                rewards = {os.path.splitext(os.path.basename(r["file"]))[0]: r["reward"]
                           for r in task.collect(res_dir)}
            used_dirs.append(d)
            for c in sorted(glob.glob(os.path.join(d, "cand", "*.py"))):
                st = os.path.basename(c)[:-3]
                mp = c[:-3] + ".meta.json"
                rew = rewards.get(st)
                if f"{run}/{st}" in exclude:
                    continue
                if not os.path.exists(mp) or rew is None or not math.isfinite(float(rew)):
                    continue
                meta = _read_json(mp)
                cell.append({"bench": bench, "arm": arm, "context": context, "model": model,
                             "run": run, "id": f"{run}/{st}", "area": meta.get("area") or "",
                             "mechanism": meta.get("mechanism") or "", "paper": meta.get("paper") or "",
                             "score": float(rew)})
        rows += cell
    idx = build_abstract_index(used_dirs)
    hit = 0
    for r in rows:
        r["abstract"] = idx.get(norm_title(title_of(r["paper"]))) or ""
        hit += bool(r["abstract"])
    print(f"[collect] {len(rows)} candidates in {len(by_cell)} cells; abstracts found for {hit}", flush=True)
    return rows


def judge_prompt(r: dict) -> str:
    return JUDGE_PROMPT.format(
        task_desc=task_desc(r["bench"]), title=title_of(r["paper"]) or "(unknown)",
        abstract=(r["abstract"] or "(abstract not available; judge from the title alone)"),
        mechanism=r["mechanism"])


def parse_judgement(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        o = json.loads(m.group(0))
        return {"v": float(o["v_lit"]), "problem_class": str(o.get("problem_class", "")),
                "reason": str(o.get("reason", ""))}
    except (ValueError, KeyError, TypeError):
        return None


def stage_judge(rows: list[dict], call, workers: int = 64) -> dict[str, dict]:
    out: dict[str, dict] = {}
    lock = threading.Lock()

    def one(r):
        j = parse_judgement(call(judge_prompt(r)))
        if j is not None:
            with lock:
                out[r["id"]] = j

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, rows))
    print(f"[judge] {len(out)}/{len(rows)} judged", flush=True)
    return out


def group_prompt(cell_rows: list[dict]) -> str:
    items = "\n".join(f"[{i}] {r['mechanism']}" for i, r in enumerate(cell_rows))
    return GROUP_PROMPT.format(n=len(cell_rows), items=items)


def parse_groups(text: str, n: int):
    fences = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text or "", re.S)
    m = re.search(r"\{.*\}", text or "", re.S)
    raw = fences[-1] if fences else (m.group(0) if m else None)
    if raw is None:
        return None
    try:
        groups = json.loads(raw)["groups"]
    except (ValueError, KeyError, TypeError):
        return None
    p, grp = [1.0 / n] * n, [-1] * n
    for gi, g in enumerate(groups):
        g = [i for i in g if isinstance(i, int) and 0 <= i < n]
        for i in set(g):
            p[i] = len(set(g)) / n
            grp[i] = gi
    return p, grp


def cells_of(rows: list[dict]) -> dict[tuple, list[dict]]:
    cells: dict[tuple, list[dict]] = {}
    for r in rows:
        cells.setdefault((r["bench"], r["arm"], r.get("context"), r["model"]), []).append(r)
    return dict(sorted(cells.items(), key=lambda kv: cell_order(kv[0])))


def stage_group(rows: list[dict], call, workers: int = 8) -> dict[str, dict]:
    out: dict[str, dict] = {}
    lock = threading.Lock()

    def one(item):
        cell, rs = item
        parsed = parse_groups(call(group_prompt(rs)), len(rs))
        if parsed is None:
            print(f"[group] FAILED {cell}", flush=True)
            return
        p, grp = parsed
        with lock:
            for r, pi, gi in zip(rs, p, grp):
                out[r["id"]] = {"p": pi, "group": gi}
        print(f"[group] {cell}: {len(set(g for g in grp if g >= 0))} groups / {len(rs)}", flush=True)

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, list(cells_of(rows).items())))
    return out


def _read_table(path: str) -> tuple[list[str], list[dict]]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        missing = [c for c in ["candidate_id"] + LABEL_COLUMNS if c not in fields]
        if missing:
            raise SystemExit(f"{path}: not a candidates table (missing columns {missing})")
        return fields, list(reader)


def _key_of(r: dict, own_keys: bool) -> dict:
    if own_keys:
        ctx = r["context"]
        bench, arm, context, model = r["bench"], r["arm"], int(ctx) if ctx not in (None, "") else None, r["proposer"]
    else:
        bench, arm, context, model = parse_run_id(r["candidate_id"].split("/", 1)[0])
    return {"id": r["candidate_id"], "bench": bench, "arm": arm, "context": context, "model": model}


def read_baselines(path: str) -> list[dict]:
    fields, table = _read_table(path)
    own_keys = all(c in fields for c in KEY_COLUMNS)
    out = []
    for r in table:
        row = _key_of(r, own_keys)
        if row["arm"] == BV_ARM:
            continue
        row.update({k: float(r[k]) for k in LABEL_COLUMNS})
        out.append(row)
    return out


def assemble(rows: list[dict], judged: dict, grouped: dict, path: str, anchors: str = "population",
             baselines: list[dict] = ()) -> int:
    out = []
    for r in rows:
        j, g = judged.get(r["id"]), grouped.get(r["id"])
        if j is None or g is None or not math.isfinite(r["score"]):
            continue
        out.append(dict(r, p=g["p"], v=j["v"]))
    n_own = len(out)
    out += [dict(b) for b in baselines if math.isfinite(b["score"])]
    add_metrics(out, anchors=population_anchors(out) if anchors == "population" else load_anchors())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CANDIDATE_COLUMNS)
        for r in out:
            w.writerow([r["id"], r["bench"], r["arm"], "" if r.get("context") is None else r["context"], r["model"],
                        r["score"], r["p"], r["v"], r["op"], r["u"], r["nv"], r["c"]])
    print(f"[assemble] {path}: {len(out)} rows ({n_own} labelled here, {len(out) - n_own} from --baselines)",
          flush=True)
    return len(out)


def read_candidates(path: str) -> list[dict]:
    fields, table = _read_table(path)
    own_keys = all(c in fields for c in KEY_COLUMNS)
    own_metrics = all(c in fields for c in METRIC_COLUMNS)
    rows = []
    for r in table:
        row = _key_of(r, own_keys)
        row.update({k: float(r[k]) for k in LABEL_COLUMNS + (METRIC_COLUMNS if own_metrics else [])})
        rows.append(row)
    if not own_metrics:
        add_metrics(rows, anchors=population_anchors(rows))
    return rows


def _write_jsonl(path: str, items) -> None:
    with open(path + ".tmp", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    os.replace(path + ".tmp", path)


def _read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bvsr.bv.label_creativity",
                                 description="Creativity labels of every verified proposal "
                                             "(paper Sec. 2, App. B.1).",
                                 formatter_class=HelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[], help="run directories (shell globs are fine)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--stage", default=",".join(STAGES), help=f"comma list of {STAGES}")
    ap.add_argument("--exclude", default="",
                    help="text file of run names or run/candidate ids to leave out")
    ap.add_argument("--anchors", choices=["population", "paper"], default="population",
                    help="utility s_best: the best score of the task among the labelled candidates "
                         "(population) or the packaged paper values")
    ap.add_argument("--baselines", default="",
                    help="released candidates table (bv_results/candidates.csv.gz): its prompting-"
                         "baseline rows are added to candidates.csv with their released labels, so "
                         "Table 1 and Fig. 2 compare your BV runs with the baselines in one population")
    ap.add_argument("--judge_workers", type=int, default=64)
    ap.add_argument("--group_workers", type=int, default=8)
    a = ap.parse_args(argv)
    stages = [s.strip() for s in a.stage.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise SystemExit(f"unknown stage(s) {bad}; choose from {STAGES}")
    os.makedirs(a.out, exist_ok=True)
    cand_path = os.path.join(a.out, "collected.jsonl")
    judge_path = os.path.join(a.out, "judgements.jsonl")
    group_path = os.path.join(a.out, "groups.jsonl")

    if "collect" in stages and a.runs:
        rows = collect(a.runs, read_exclude(a.exclude))
        _write_jsonl(cand_path, rows)
    elif os.path.exists(cand_path):
        rows = _read_jsonl(cand_path)
    else:
        raise SystemExit("no candidates: pass --runs (collect stage) first")

    cache = CallCache(os.path.join(a.out, "calls.jsonl"))
    if "judge" in stages:
        judged = stage_judge(rows, make_caller("nonobviousness", cache), a.judge_workers)
        _write_jsonl(judge_path, [dict(id=k, **v) for k, v in sorted(judged.items())])
    if "group" in stages:
        grouped = stage_group(rows, make_caller("grouping", cache), a.group_workers)
        _write_jsonl(group_path, [dict(id=k, **v) for k, v in sorted(grouped.items())])
    if "assemble" in stages:
        judged = {d["id"]: d for d in _read_jsonl(judge_path)} if os.path.exists(judge_path) else {}
        grouped = {d["id"]: d for d in _read_jsonl(group_path)} if os.path.exists(group_path) else {}
        assemble(rows, judged, grouped, os.path.join(a.out, "candidates.csv"), a.anchors,
                 read_baselines(a.baselines) if a.baselines else [])


if __name__ == "__main__":
    main()
