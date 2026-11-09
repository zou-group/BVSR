from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import time
import uuid

import numpy as np

from bvsr.sr.prompt import MODEL_TAG
from bvsr.tasks import task_spec

from . import patch


def _print(msg: str) -> None:
    print(msg, flush=True)


def _sha(text: str | None) -> str | None:
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text is not None else None


def _append_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def keep_and_discard(scores, cut: float) -> tuple[list[int], list[int]]:
    order = [int(j) for j in np.argsort(-np.asarray(scores, dtype=float), kind="stable")]
    d = int(round(cut * len(order)))
    return order[:len(order) - d], order[len(order) - d:]


class SRState:
    KEYS = ("batches", "submitted", "delivered", "failed", "kept", "discarded", "max_tokens")

    def __init__(self, path: str, meta: dict, log=_print):
        self.path = path
        self.d: dict = {}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    self.d = json.load(fh)
            except (OSError, ValueError) as e:
                log(f"[sr] cannot read {path} ({e}); starting fresh counters")
                self.d = {}
        for k in self.KEYS:
            self.d.setdefault(k, 0)
        self.d.setdefault("starts", 0)
        self.d["starts"] += 1
        self.d.setdefault("checkpoints", [])
        self.d.update(meta)

    def record(self, *, submitted, delivered, failed, kept, discarded, max_tokens, fingerprint) -> None:
        d = self.d
        d["batches"] += 1
        for k, v in (("submitted", submitted), ("delivered", delivered), ("failed", failed),
                     ("kept", kept), ("discarded", discarded)):
            d[k] += int(v)
        d["max_tokens"] = max(int(d["max_tokens"]), int(max_tokens))
        if fingerprint and fingerprint not in d["checkpoints"]:
            d["checkpoints"].append(fingerprint)

    def save(self) -> None:
        self.d["updated_at"] = time.time()
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.d, fh, indent=1)
        os.replace(tmp, self.path)


class _HoldQueue:
    __slots__ = ("gate", "gen_id", "chain_idx")

    def __init__(self, gate, gen_id, chain_idx):
        self.gate, self.gen_id, self.chain_idx = gate, gen_id, chain_idx

    async def put(self, node_id):
        self.gate._take(self.gen_id, self.chain_idx, node_id)


class _EngineView:
    __slots__ = ("_real", "eval_queue")

    def __init__(self, real, hold_queue):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "eval_queue", hold_queue)

    def __getattr__(self, name):
        return getattr(self._real, name)


class Gate:
    def __init__(self, client, spec: str, task: str, out_dir: str, cut: float = 0.5, log=_print):
        if not 0.0 <= float(cut) <= 1.0:
            raise ValueError(f"cut must be in [0, 1], got {cut}")
        self.client, self.spec, self.task, self.out_dir = client, spec, task, out_dir
        self.spec_sha256 = _sha(spec)
        self.cut = float(cut)
        self.log = log
        self.start_id = uuid.uuid4().hex[:8]
        self.scores_path = os.path.join(out_dir, "sr_scores.jsonl")
        self.discards_path = os.path.join(out_dir, "sr_discards.jsonl")
        self.state = SRState(os.path.join(out_dir, "sr_state.json"),
                             {"task": task, "cut": self.cut, "spec_sha256": self.spec_sha256},
                             log=log)
        self.lock = threading.Lock()
        self.held: dict = {}
        self.chain_of: dict = {}
        self.first_held_at: dict = {}
        self.flushed: set = set()
        self.tasks: dict = {}
        self.fatal: str | None = None
        self.engine = None
        self.loop = None
        self._orig_hgo = None
        self._orig_ogf = None

    def bind_originals(self, orig_hgo, orig_ogf) -> None:
        self._orig_hgo, self._orig_ogf = orig_hgo, orig_ogf

    def _bind_engine(self, engine) -> None:
        if self.engine is None:
            self.engine = engine
            self.loop = asyncio.get_running_loop()
        elif engine is not self.engine:
            raise RuntimeError("the SR gate serves one engine per process")

    async def handle_generation_output(self, runtime, engine, task, result, track_io):
        try:
            self._bind_engine(engine)
            view = _EngineView(engine, _HoldQueue(self, task.gen_id, task.chain_idx))
            await self._orig_hgo(runtime, view, task, result, track_io)
            self.maybe_flush(task.gen_id)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._stop(f"handle_generation_output gen_id={getattr(task, 'gen_id', None)}: {e!r}")

    def on_generation_failed(self, selector, gen_id):
        completion = self._orig_ogf(selector, gen_id)
        try:
            self.maybe_flush(gen_id, selector=selector)
        except Exception as e:
            self._stop(f"on_generation_failed gen_id={gen_id}: {e!r}")
        return completion

    def _take(self, gen_id, chain_idx, node_id) -> None:
        with self.lock:
            late = gen_id in self.flushed
            self.held.setdefault(gen_id, []).append(node_id)
            self.chain_of[gen_id] = chain_idx
            self.first_held_at.setdefault(gen_id, time.time())
        if late:
            self.log(f"[sr] WARNING gen_id={gen_id}: candidate {node_id} arrived after its step was flushed")

    def maybe_flush(self, gen_id, selector=None) -> bool:
        if selector is None:
            if self.engine is None:
                return False
            selector = self.engine.selector
        with selector._lock:
            b = selector._batches.get(gen_id)
            if b is None:
                return False
            done, submitted = int(b["done"]), int(b["submitted"])
        with self.lock:
            n_held = len(self.held.get(gen_id, ()))
            if gen_id in self.flushed or n_held == 0 or n_held + done < submitted:
                return False
            self.flushed.add(gen_id)
            task = self._create_task(gen_id) if self._on_loop() else None
        if task is None:
            self.loop.call_soon_threadsafe(self._schedule, gen_id)
        else:
            task.add_done_callback(lambda t, g=gen_id: self._task_done(g))
        return True

    def _on_loop(self) -> bool:
        try:
            return asyncio.get_running_loop() is self.loop
        except RuntimeError:
            return False

    def _create_task(self, gen_id):
        task = self.loop.create_task(self.flush(gen_id), name=f"sr_flush_{gen_id}")
        self.tasks[gen_id] = task
        return task

    def _schedule(self, gen_id) -> None:
        with self.lock:
            task = self._create_task(gen_id)
        task.add_done_callback(lambda t, g=gen_id: self._task_done(g))

    def _task_done(self, gen_id) -> None:
        with self.lock:
            self.tasks.pop(gen_id, None)

    async def flush(self, gen_id) -> None:
        try:
            await self._flush(gen_id)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if self.engine._stop_event.is_set():
                self.log(f"[sr] gen_id={gen_id}: run stopping, held batch not scored ({type(e).__name__})")
                return
            self._stop(f"flush gen_id={gen_id}: {type(e).__name__}: {e}")

    async def _flush(self, gen_id) -> None:
        engine, selector = self.engine, self.engine.selector
        with self.lock:
            ids = list(self.held[gen_id])
            chain_idx = self.chain_of[gen_id]
            first_at = self.first_held_at[gen_id]
        with selector._lock:
            b = selector._batches.get(gen_id)
            if b is None:
                raise RuntimeError("batch settled before its flush")
            n_failed, submitted = int(b["done"]), int(b["submitted"])
        n = len(ids)
        async with engine._db_lock:
            nodes = [engine._pending_nodes[i] for i in ids]
        items = [{"id": nd.id, "code": nd.code} for nd in nodes]
        t0 = time.time()
        res = await asyncio.to_thread(self.client.score_batch, self.task, self.spec, items, gen_id=gen_id,
                                      should_stop=engine._stop_event.is_set)
        score_s = time.time() - t0
        scores = np.asarray(res["scores"], dtype=float)
        n_tokens = [int(x) for x in res["n_tokens"]]
        if scores.shape != (n,) or len(n_tokens) != n:
            raise RuntimeError(f"selector returned {scores.shape} scores / {len(n_tokens)} token counts for {n}")
        if not np.all(np.isfinite(scores)):
            raise RuntimeError("selector returned a non-finite score")
        fingerprint = res.get("ckpt_fingerprint")
        model_tag = str(res.get("model_tag") or MODEL_TAG)

        keep, disc = keep_and_discard(scores, self.cut)
        rank = {j: r + 1 for r, j in enumerate(keep + disc)}
        d, n_keep = len(disc), len(keep)
        common = {"start_id": self.start_id, "gen_id": gen_id, "chain_idx": chain_idx,
                  "n_submitted": submitted, "n_delivered": n, "n_failed": n_failed, "n_kept": n_keep}

        for j in reversed(disc):
            nd, nid = nodes[j], ids[j]
            async with engine._db_lock:
                engine._pending_nodes.pop(nid)
            light = {"type": "sr_discard", "reason": f"sr_rank {rank[j]}/{n}", "gen_id": gen_id,
                     "chain_idx": chain_idx, "node_id": nid, "code_sha256": _sha(nd.code),
                     "n_tokens": n_tokens[j], "sr_score": float(scores[j]), "sr_rank": rank[j],
                     "n_delivered": n, "n_failed": n_failed, "n_kept": n_keep, "created_at": nd.created_at}
            async with engine._counter_lock:
                engine._failure_records.append(light)
            full = dict(light, start_id=self.start_id, code=nd.code, llm_input=getattr(nd, "llm_input", None),
                        llm_output=getattr(nd, "llm_output", None), ckpt_fingerprint=fingerprint)
            _append_jsonl(self.discards_path, [full])
            completion = self._orig_ogf(selector, gen_id)
            if completion is not None:
                await engine._handle_pending_finalize(completion)

        now = time.time()
        hold_s = now - first_at
        lines = [dict(common, type="cand", ts=now, node_id=ids[j], code_sha256=_sha(nodes[j].code),
                      n_tokens=n_tokens[j], score=float(scores[j]), rank=rank[j], kept=rank[j] <= n_keep)
                 for j in keep + disc]
        lines.append(dict(common, type="batch", ts=now, n_discarded=d, ties=int(n - len(set(scores.tolist()))),
                          hold_s=round(hold_s, 1), score_s=round(score_s, 1), cut=self.cut,
                          spec_sha256=self.spec_sha256, model_tag=model_tag, ckpt_fingerprint=fingerprint,
                          kept_ids=[ids[j] for j in keep], discarded_ids=[ids[j] for j in reversed(disc)]))
        _append_jsonl(self.scores_path, lines)
        self.state.record(submitted=submitted, delivered=n, failed=n_failed, kept=n_keep, discarded=d,
                          max_tokens=max(n_tokens), fingerprint=fingerprint)
        self.state.save()

        for j in keep:
            await engine.eval_queue.put(ids[j])
        with self.lock:
            self.held[gen_id] = []
        engine._progress_event.set()
        self.log(f"[sr] gen_id={gen_id} chain={chain_idx} delivered={n} failed={n_failed} kept={n_keep} "
                 f"discarded={d} hold_s={hold_s:.0f} score_s={score_s:.1f}")

    def _stop(self, reason: str) -> None:
        self.log(f"[sr] FATAL {reason}; stopping the run (fix the cause, then resume with --resume)")
        with self.lock:
            self.fatal = self.fatal or reason
        try:
            self.state.d["stopped"] = reason
            self.state.save()
        except OSError:
            pass
        engine = self.engine
        if engine is not None:
            engine._stop_event.set()
            engine._progress_event.set()


ACTIVE: Gate | None = None


def attach(gate: Gate, runtime_cls=None, policy_cls=None) -> Gate:
    global ACTIVE
    if ACTIVE is not None:
        raise RuntimeError("an SR gate is already attached")
    if runtime_cls is None:
        from simpletes.engine.runtime import RuntimeBase as runtime_cls
    if policy_cls is None:
        from simpletes.policies.base import TrajectoryPolicyBase as policy_cls
    gate.bind_originals(vars(runtime_cls)["handle_generation_output"], vars(policy_cls)["on_generation_failed"])

    async def handle_generation_output(self, engine, task, result, track_io):
        return await gate.handle_generation_output(self, engine, task, result, track_io)

    def on_generation_failed(self, gen_id):
        return gate.on_generation_failed(self, gen_id)

    patch(runtime_cls, "handle_generation_output", handle_generation_output)
    patch(policy_cls, "on_generation_failed", on_generation_failed)
    ACTIVE = gate
    return gate


def detach() -> None:
    global ACTIVE
    ACTIVE = None


def install(cfg, client=None) -> Gate:
    if client is None:
        from bvsr.stes.sr_client import SRClient
        client = SRClient(cfg.sr_url, cfg.sr_key, timeout=cfg.sr_timeout_s, retry_s=cfg.sr_retry_s)
    spec = task_spec(cfg.task)
    gate = attach(Gate(client, spec, cfg.task, cfg.out, cut=cfg.cut))
    _print(f"[sr] gate installed: task={cfg.task} cut={cfg.cut} spec sha256={gate.spec_sha256} "
           f"({len(spec)} chars) out={cfg.out}")
    return gate


__all__ = ["ACTIVE", "Gate", "SRState", "attach", "detach", "install", "keep_and_discard"]
