from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid

import requests

CONNECT_TIMEOUT_S = 30
HEALTH_TIMEOUT_S = 15

_FLIGHT = threading.Lock()


class SRClientError(Exception):
    pass


class AuthError(SRClientError):
    pass


class InputRejected(SRClientError):
    pass


class Stopped(SRClientError):
    pass


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


class SRClient:
    def __init__(self, url: str | None = None, key: str | None = None, timeout: float = 3600.0,
                 retry_s: float = 15.0, log=print, session=None):
        url = url or os.environ.get("BVSR_SR_URL")
        key = key if key is not None else os.environ.get("BVSR_SR_KEY")
        if not url:
            raise SRClientError("no SR server URL: pass --sr_url or set BVSR_SR_URL")
        if not key:
            raise SRClientError("no SR server key: set BVSR_SR_KEY")
        self.url = url.rstrip("/")
        self.key = key
        self.timeout, self.retry_s = float(timeout), float(retry_s)
        self.log = log
        self.session = session or requests.Session()
        self.run_id = uuid.uuid4().hex[:12]

    @staticmethod
    def health(url: str, timeout: float = HEALTH_TIMEOUT_S) -> dict | None:
        try:
            r = requests.get(url.rstrip("/") + "/health", timeout=timeout)
            d = r.json() if r.status_code == 200 else None
        except (requests.RequestException, ValueError):
            return None
        return d if isinstance(d, dict) and d.get("ok") else None

    def wait_ready(self, max_wait_s: float, poll_s: float = 30.0) -> dict:
        t0 = time.time()
        while True:
            h = self.health(self.url)
            if h is not None:
                return h
            if time.time() - t0 >= max_wait_s:
                raise SRClientError(f"SR server {self.url} not ready after {max_wait_s / 60:.0f} min")
            self.log(f"[sr-client] waiting for {self.url}/health ...")
            time.sleep(poll_s)

    def score_batch(self, task: str, spec: str, items: list[dict], *, gen_id, should_stop=None) -> dict:
        body = {"task": task, "spec": spec, "spec_sha256": _sha(spec),
                "run_id": self.run_id, "gen_id": gen_id,
                "items": [{"id": str(it["id"]), "code": it["code"]} for it in items]}
        data = json.dumps(body).encode()
        with _FLIGHT:
            return self._post(data, len(items), f"task={task} gen_id={gen_id} n={len(items)}", should_stop)

    def _post(self, data: bytes, n: int, tag: str, should_stop=None) -> dict:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"}
        attempt = 0
        while True:
            attempt += 1
            if attempt > 1 and should_stop is not None and should_stop():
                raise Stopped(f"{tag}: abandoned after {attempt - 1} attempts (shutting down)")
            t0 = time.time()
            try:
                r = self.session.post(self.url + "/score", data=data, headers=headers,
                                      timeout=(CONNECT_TIMEOUT_S, self.timeout))
            except requests.RequestException as e:
                self._retry(attempt, tag, f"{type(e).__name__}: {str(e)[:200]}")
                continue
            if r.status_code == 200:
                try:
                    res = r.json()
                except ValueError as e:
                    self._retry(attempt, tag, f"unparsable response: {e}")
                    continue
                if (not isinstance(res.get("scores"), list) or len(res["scores"]) != n
                        or len(res.get("n_tokens", [])) != n):
                    raise SRClientError(f"malformed /score response for {tag}: keys={sorted(res)}")
                self.log(f"[sr-client] {tag} scored in {time.time() - t0:.1f}s (attempt {attempt}, "
                         f"cached={res.get('cached')}, max tokens {max(res['n_tokens'])})")
                return res
            detail = r.text[:500]
            if r.status_code >= 500:
                self._retry(attempt, tag, f"HTTP {r.status_code}: {detail}")
                continue
            if r.status_code == 401:
                raise AuthError(f"401 from {self.url}: {detail}")
            if r.status_code == 422:
                raise InputRejected(f"422 from {self.url} for {tag}: {detail}")
            raise SRClientError(f"HTTP {r.status_code} from {self.url} for {tag}: {detail}")

    def _retry(self, attempt: int, tag: str, what: str) -> None:
        if attempt == 1 or attempt % 20 == 0:
            self.log(f"[sr-client] {tag} attempt {attempt}: {what}; retrying every {self.retry_s:g}s "
                     f"(candidates stay held until the selector answers)")
        time.sleep(self.retry_s)


__all__ = ["AuthError", "InputRejected", "SRClient", "SRClientError", "Stopped"]
