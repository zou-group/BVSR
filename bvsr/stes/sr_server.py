from __future__ import annotations

import argparse
import collections
import hashlib
import hmac
import json
import os
import secrets
import signal
import statistics
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bvsr.cli import HelpFormatter
from bvsr.sr.prompt import DEFAULT_CKPT, MAX_LEN, MODEL_TAG, TOKENIZER_ID, build_prompt
from bvsr.sr.scorer import Scorer, SRInputRejected

LRU_SIZE = 512
INFLIGHT_MAX_S = 2 * 3600
WARMUP_SPEC = "Warm-up task: return zero."
WARMUP_CODE = "# EVOLVE-BLOCK-START\ndef solve():\n    return 0\n# EVOLVE-BLOCK-END\n"


def _log(msg: str) -> None:
    print(f"[sr-server] {time.strftime('%F %T')} {msg}", flush=True)


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def ckpt_fingerprint(ckpt: str) -> str:
    if os.path.isdir(ckpt):
        with open(os.path.join(ckpt, "config.json"), "rb") as fh:
            csha = hashlib.sha256(fh.read()).hexdigest()[:12]
        nbytes = sum(os.path.getsize(os.path.join(ckpt, f)) for f in sorted(os.listdir(ckpt))
                     if f.endswith(".safetensors"))
        return f"config:{csha};safetensors:{nbytes}"
    try:
        from huggingface_hub import HfApi
        return f"hf:{ckpt}@{HfApi().model_info(ckpt).sha}"
    except Exception:
        return f"hf:{ckpt}"


class _Inflight:
    def __init__(self):
        self.event = threading.Event()
        self.result = None
        self.error = None


class ScoringState:
    def __init__(self, scorer, api_key: str, ckpt: str, fingerprint: str, max_len: int):
        self.scorer, self.api_key = scorer, api_key
        self.ckpt, self.fingerprint, self.max_len = ckpt, fingerprint, max_len
        self.gpu_lock = threading.Lock()
        self.meta_lock = threading.Lock()
        self.cache: collections.OrderedDict = collections.OrderedDict()
        self.inflight: dict = {}
        self.inflight_started: dict = {}
        self.task_spec: dict[str, str] = {}
        self.ready = False
        self.broken = None
        self.t_start = time.time()
        self.n_requests = self.n_items = self.n_cached = 0

    def _compute(self, prompts, ids):
        tok_ids, n_tokens = [], []
        for item_id, p in zip(ids, prompts):
            x = self.scorer.encode(p)
            try:
                n_tokens.append(self.scorer.check(x))
            except SRInputRejected as e:
                raise SRInputRejected(f"item {item_id}: {e}") from None
            tok_ids.append(x)
        with self.gpu_lock:
            scores = [float(v) for v in self.scorer.score_ids(tok_ids)]
        if len(scores) != len(prompts):
            raise RuntimeError(f"{len(scores)} scores for {len(prompts)} prompts")
        return scores, n_tokens

    def score(self, body: dict) -> dict:
        spec, task = body["spec"], body["task"]
        if _sha(spec) != body["spec_sha256"]:
            raise SRInputRejected("spec_sha256 does not match sha256(spec)")
        with self.meta_lock:
            seen = self.task_spec.setdefault(task, body["spec_sha256"])
        if seen != body["spec_sha256"]:
            raise SRInputRejected(f"the spec of task {task} changed while this server was running")
        ids = [it["id"] for it in body["items"]]
        codes = [it["code"] for it in body["items"]]
        prompts = [build_prompt(spec, c) for c in codes]
        key = (str(body["run_id"]), str(body["gen_id"]), body["spec_sha256"], tuple(sorted(_sha(c) for c in codes)))
        t0 = time.time()
        with self.meta_lock:
            self.n_requests += 1
            self.n_items += len(ids)
            hit = self.cache.get(key)
            if hit is not None:
                self.cache.move_to_end(key)
                self.n_cached += 1
                entry, joined = None, False
            else:
                entry = self.inflight.get(key)
                joined = entry is not None
                if not joined:
                    entry = self.inflight[key] = _Inflight()
                    self.inflight_started[key] = t0
        if hit is not None:
            return dict(hit, cached=True, elapsed_s=time.time() - t0)
        if joined:
            entry.event.wait()
            if entry.error is not None:
                raise entry.error
            return dict(entry.result, cached=True, elapsed_s=time.time() - t0)
        try:
            scores, n_tokens = self._compute(prompts, ids)
            result = {"scores": scores, "n_tokens": n_tokens, "model_tag": MODEL_TAG,
                      "ckpt_fingerprint": self.fingerprint}
            entry.result = result
            with self.meta_lock:
                self.cache[key] = result
                while len(self.cache) > LRU_SIZE:
                    self.cache.popitem(last=False)
        except BaseException as e:
            entry.error = e
            raise
        finally:
            with self.meta_lock:
                self.inflight.pop(key, None)
                self.inflight_started.pop(key, None)
            entry.event.set()
        res = dict(result, cached=False, elapsed_s=time.time() - t0)
        _log(f"/score task={task} run={body['run_id']} gen_id={body['gen_id']} n={len(ids)} "
             f"tokens max={max(n_tokens)} median={int(statistics.median(n_tokens))} "
             f"elapsed={res['elapsed_s']:.2f}s")
        return res

    def probe_after_failure(self, exc: BaseException) -> None:
        try:
            with self.gpu_lock:
                self.scorer.score_ids([self.scorer.encode(build_prompt(WARMUP_SPEC, WARMUP_CODE))])
        except Exception as e:
            self.broken = f"after {exc!r}: probe raised {e!r}"
            _log(f"SCORER BROKEN: {self.broken}")

    def health(self) -> dict:
        now = time.time()
        with self.meta_lock:
            counters = dict(n_requests=self.n_requests, n_items=self.n_items, n_cached=self.n_cached,
                            n_inflight=len(self.inflight))
            oldest = round(now - min(self.inflight_started.values()), 1) if self.inflight_started else 0.0
        return {"ok": bool(self.ready and self.broken is None and oldest < INFLIGHT_MAX_S),
                "ready": self.ready, "broken": self.broken, "oldest_inflight_s": oldest,
                "ckpt": self.ckpt, "ckpt_fingerprint": self.fingerprint,
                "max_len": self.max_len, "model_tag": MODEL_TAG,
                "uptime_s": round(now - self.t_start, 1), **counters}

    def warm_up(self) -> None:
        t0 = time.time()
        body = {"task": "_warmup", "spec": WARMUP_SPEC, "spec_sha256": _sha(WARMUP_SPEC),
                "run_id": "warmup", "gen_id": -1, "items": [{"id": "warmup", "code": WARMUP_CODE}]}
        res = self.score(body)
        with self.meta_lock:
            self.cache.clear()
            self.task_spec.pop("_warmup", None)
            self.n_requests = self.n_items = self.n_cached = 0
        _log(f"warm-up done in {time.time() - t0:.1f}s (score {res['scores'][0]:.4f})")


def validate(body) -> dict:
    if not isinstance(body, dict):
        raise ValueError("body is not a JSON object")
    for k in ("task", "spec", "spec_sha256", "run_id"):
        if not isinstance(body.get(k), str) or not body[k]:
            raise ValueError(f"field {k!r} missing or not a non-empty string")
    if "gen_id" not in body:
        raise ValueError("field 'gen_id' missing")
    items = body.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("field 'items' missing or empty")
    for i, it in enumerate(items):
        if not isinstance(it, dict) or not isinstance(it.get("id"), str) or not isinstance(it.get("code"), str):
            raise ValueError(f"items[{i}] must be {{id: str, code: str}}")
    if len({it["id"] for it in items}) != len(items):
        raise ValueError("duplicate item ids")
    return body


class Handler(BaseHTTPRequestHandler):
    server_version = "BVSR-SR/1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/health":
            return self._send(200, self.server.state.health())
        self._send(404, {"error": "not found"})

    def do_POST(self):
        st = self.server.state
        if self.path.split("?", 1)[0] != "/score":
            return self._send(404, {"error": "not found"})
        auth = self.headers.get("Authorization", "")
        if not (auth.startswith("Bearer ") and hmac.compare_digest(auth[7:].strip(), st.api_key)):
            return self._send(401, {"error": "bad api key"})
        if not st.ready:
            return self._send(503, {"error": "not ready (loading or warming up)"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = validate(json.loads(self.rfile.read(n).decode()))
        except (ValueError, UnicodeDecodeError) as e:
            return self._send(400, {"error": f"bad request: {e}"})
        try:
            res = st.score(body)
        except SRInputRejected as e:
            _log(f"422 task={body['task']} gen_id={body['gen_id']}: {e}")
            return self._send(422, {"error": str(e)})
        except Exception as e:
            _log(f"500 task={body['task']} gen_id={body['gen_id']}: {e!r}\n{traceback.format_exc()}")
            self._send(500, {"error": repr(e)})
            if st.broken is None:
                st.probe_after_failure(e)
            return
        self._send(200, res)


class SRServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, state: ScoringState):
        super().__init__(addr, Handler)
        self.state = state


def read_or_create_key(path: str | None) -> str:
    if path is None:
        key = os.environ.get("BVSR_SR_KEY", "").strip()
        if not key:
            sys.exit("pass --api_key_file or set BVSR_SR_KEY")
        return key
    if not os.path.exists(path):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(secrets.token_hex(24) + "\n")
        _log(f"created a new API key in {path}")
    with open(path) as fh:
        key = fh.read().strip()
    if not key:
        sys.exit(f"empty API key file {path}")
    return key


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bvsr.stes.sr_server", formatter_class=HelpFormatter,
                                 description="Serve the SR selector to the SimpleTES SR gate over HTTP.")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="selector checkpoint: local directory or Hugging Face id")
    ap.add_argument("--tokenizer", default=TOKENIZER_ID)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="bind address")
    ap.add_argument("--max_len", type=int, default=MAX_LEN, help="longest accepted input in tokens")
    ap.add_argument("--mb", type=int, default=8, help="sequences per micro-batch")
    ap.add_argument("--mb_tokens", type=int, default=65536, help="padded tokens per micro-batch")
    ap.add_argument("--api_key_file", default=None,
                    help="file holding the bearer key (created with a random key if missing); "
                         "default: $BVSR_SR_KEY")
    a = ap.parse_args(argv)
    api_key = read_or_create_key(a.api_key_file)
    ckpt, fp = a.ckpt, ckpt_fingerprint(a.ckpt)
    _log(f"loading {ckpt} ({fp}) max_len={a.max_len} mb={a.mb} mb_tokens={a.mb_tokens}")
    scorer = Scorer(ckpt, tokenizer_id=a.tokenizer, max_len=a.max_len, mb=a.mb, mb_tokens=a.mb_tokens,
                    overlength="reject", log=_log)
    state = ScoringState(scorer, api_key, ckpt, fp, a.max_len)
    srv = SRServer((a.host, a.port), state)
    th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.5}, daemon=True)
    th.start()
    _log(f"listening on {a.host}:{a.port} (warming up)")
    state.warm_up()
    state.ready = True
    _log(f"READY ckpt={ckpt} fingerprint={fp}")

    def _term(signum, frame):
        _log(f"signal {signum}: shutting down")
        threading.Thread(target=srv.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    th.join()
    srv.server_close()


if __name__ == "__main__":
    main()
