#!/usr/bin/env python
"""Untrusted-side runner of the MLS-Anomaly evaluator: the only process that executes candidate
code, deliberately a different module from the scorer (``eval_mls_anomaly.py``).

Contract with the parent (files in this process's working directory, which is a private
temporary directory):

  ``in.pkl``   {"code": str, "datasets": [{"id": "d0", "Xtr": ndarray, "Xte": ndarray}, ...],
                "per_dataset_timeout": int, "want_torch": bool, "allow_extra": [dir, ...]}
  ``out.pkl``  {"scores": {id: ndarray|None}, "errors": {id: str}, "timings": {id: float},
                "param_count": int, "grad_tensor_numel": int, "torch_imported": bool, "note": str}

By construction this process has no labels, no dataset name or path, no split seed or split code,
no metric function the parent uses, and no way to align an externally reconstructed label vector
with the rows it scores (the parent permutes the test rows with a per-candidate secret and
un-permutes the returned scores itself). On top of that a non-removable ``sys.addaudithook``
enforces a read allowlist and blocks process spawning, sockets and the libc handle via ctypes.
"""
import os
import pickle
import sys
import time

# ---------------------------------------------------------------------------------------------
# 0. read the payload before any hook is installed
# ---------------------------------------------------------------------------------------------
with open("in.pkl", "rb") as _fh:
    _PAYLOAD = pickle.loads(_fh.read())


# ---------------------------------------------------------------------------------------------
# 1. torch parameter accounting (installed BEFORE candidate code runs when torch is expected)
# ---------------------------------------------------------------------------------------------
_PEAK = [0]                                    # peak sum(p.numel()) ever allocated as a Parameter
_LAST_SWEEP = [0.0]                            # wall-clock throttle for the autograd-time sweep


def _sweep_grad_leaves():
    """Sum numel over every live LEAF tensor with requires_grad, i.e. exactly the learnable
    weights of a hand-rolled model whether or not they are wrapped in nn.Parameter, and update the
    peak. Such weights are invisible to nn.Parameter / nn.Module walks but must exist as
    requires_grad leaves whenever gradients are computed."""
    torch = sys.modules.get("torch")
    if torch is None:
        return
    import gc
    total = 0
    try:
        for o in gc.get_objects():
            try:
                if isinstance(o, torch.Tensor) and o.requires_grad and o.is_leaf:
                    total += int(o.numel())
            except Exception:
                continue
    except Exception:
        return
    if total > _PEAK[0]:
        _PEAK[0] = total


def _maybe_sweep(force=False):
    now = time.time()
    if force or (now - _LAST_SWEEP[0]) > 1.0:          # at most about once per second
        _LAST_SWEEP[0] = now
        _sweep_grad_leaves()


def _install_torch_peak_tracker():
    """Install every torch parameter counter BEFORE any candidate statement runs. Three hooks, so
    the peak survives even if the candidate frees or hides its weights before the final count:
      (a) nn.Parameter.__new__      -- classic parameters;
      (b) Tensor.requires_grad_(..) -- raw leaves made trainable after creation;
      (c) autograd backward / grad  -- a throttled sweep of all requires_grad leaves at the moment
          they are provably alive (training needs gradients).
    The import cost is only paid when the candidate source mentions torch."""
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return
    orig_new = nn.Parameter.__new__
    running = [0]

    def _new(cls, data=None, requires_grad=True):
        p = orig_new(cls, data, requires_grad)
        try:
            running[0] += int(p.numel())
            if running[0] > _PEAK[0]:
                _PEAK[0] = running[0]
        except Exception:
            pass
        return p

    nn.Parameter.__new__ = staticmethod(_new)

    orig_rg = torch.Tensor.requires_grad_

    def _rg(self, requires_grad=True):
        out = orig_rg(self, requires_grad)
        try:
            _maybe_sweep(force=True)
        except Exception:
            pass
        return out

    torch.Tensor.requires_grad_ = _rg

    def _wrap_autograd(mod, name):
        try:
            orig = getattr(mod, name)
        except Exception:
            return

        def wrapper(*a, **k):
            try:
                _maybe_sweep()
            except Exception:
                pass
            return orig(*a, **k)
        try:
            setattr(mod, name, wrapper)
        except Exception:
            pass

    _wrap_autograd(torch.Tensor, "backward")
    _wrap_autograd(torch.autograd, "backward")
    _wrap_autograd(torch.autograd, "grad")


def _count_torch(detector):
    """Max of three independent counts in the units of the benchmark's budget_check.py
    (sum of p.numel() over torch parameters): (a) the peak tracker, (b) a gc sweep over every live
    nn.Parameter, (c) budget_check.py's own attribute walk of the fitted detector.
    Returns (param_count, grad_tensor_numel, torch_imported)."""
    torch = sys.modules.get("torch")
    if torch is None:
        return 0, 0, False
    import gc
    try:
        _maybe_sweep(force=True)                    # final grad-leaf sweep (raw-tensor models)
    except Exception:
        pass
    try:
        import torch.nn as nn
    except Exception:
        return _PEAK[0], 0, True

    live = 0
    grad_numel = 0
    try:
        for o in gc.get_objects():
            try:
                if isinstance(o, nn.Parameter):
                    live += int(o.numel())
                elif isinstance(o, torch.Tensor) and getattr(o, "requires_grad", False):
                    grad_numel += int(o.numel())        # recorded only, never enforced
            except Exception:
                continue
    except Exception:
        pass

    walk = 0
    try:
        visited, queue = set(), [detector]
        while queue:
            obj = queue.pop()
            if id(obj) in visited:
                continue
            visited.add(id(obj))
            if isinstance(obj, nn.Module):
                walk += sum(p.numel() for p in obj.parameters())
                continue
            for attr_name in dir(obj):
                if attr_name.startswith("__"):
                    continue
                try:
                    attr = getattr(obj, attr_name)
                except Exception:
                    continue
                if isinstance(attr, nn.Module):
                    walk += sum(p.numel() for p in attr.parameters())
                elif hasattr(attr, "__dict__") and not callable(attr):
                    if id(attr) not in visited:
                        queue.append(attr)
    except Exception:
        pass
    return int(max(_PEAK[0], live, walk)), int(grad_numel), True


# ---------------------------------------------------------------------------------------------
# 2. the audit hook: read allowlist + no process spawning + no sockets + no libc ctypes handle
# ---------------------------------------------------------------------------------------------
class SandboxViolation(BaseException):
    """Deliberately NOT an Exception: a bare `except Exception` in candidate code cannot swallow it."""


_SELF_DIR = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
# Drop this script's own directory from the import path, so the import machinery never lists it
# and the read allowlist below needs no exception for it.
sys.path = [p for p in sys.path if os.path.realpath(p or os.getcwd()) != _SELF_DIR]


def _allowed_prefixes():
    pref = []
    for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, os.path.dirname(os.__file__)):
        if p:
            pref.append(os.path.realpath(p))
    for p in sys.path:                        # site-packages overlays the parent put on PYTHONPATH
        if p and os.path.isdir(p):
            rp = os.path.realpath(p)
            if rp != "/":
                pref.append(rp)
    pref += ["/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/proc", "/sys", "/dev",
             "/run", "/opt", "/var/tmp", "/tmp", os.path.realpath(os.getcwd())]
    # Extra writable caches are named explicitly by the parent rather than derived from
    # environment variables such as XDG_CACHE_HOME, which may point next to the datasets.
    for d in _PAYLOAD.get("allow_extra", []):
        pref.append(os.path.realpath(d))
    out, seen = [], set()
    for p in pref:
        if p and p not in seen:
            seen.add(p)
            out.append(p if p.endswith(os.sep) else p + os.sep)
    return tuple(out)


def _prewarm():
    """Some legitimate libraries shell out once, at first use, for host discovery: matplotlib's
    font manager runs `fc-list` (pyod imports matplotlib), joblib runs `lscpu`, threadpoolctl runs
    `ldconfig -p`. Doing that here, in trusted code before the hook exists, lets the hook refuse
    process spawning outright without breaking pyod or scikit-learn."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.font_manager     # noqa: F401
    except Exception:
        pass
    try:
        import joblib
        joblib.cpu_count(only_physical_cores=True)
    except Exception:
        pass
    try:
        from sklearn.utils.parallel import _get_threadpool_controller
        _get_threadpool_controller()
    except Exception:
        try:
            import threadpoolctl
            threadpoolctl.ThreadpoolController()
        except Exception:
            pass
    if _PAYLOAD.get("want_torch"):
        # torch's start-up touches caches outside the library tree, so it is imported here, and
        # the parameter tracker is installed before any candidate statement runs
        _install_torch_peak_tracker()


_prewarm()
_ALLOW = _allowed_prefixes()
# A fresh interpreter started by the candidate would not carry this audit hook, so every route to
# one is refused; sockets are refused so labels cannot be fetched over the network.
_DENY_EVENTS = (
    "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.fork", "os.forkpty",
    "subprocess.Popen", "socket.connect", "socket.bind", "socket.getaddrinfo",
    "socket.gethostbyname", "socket.sendto", "urllib.Request", "ftplib.connect",
)


def _hook(event, args):
    if event in ("open", "os.listdir", "os.scandir"):
        path = args[0] if args else None
        if path is None or isinstance(path, int):
            return                                   # an already-open descriptor
        try:
            p = os.path.realpath(os.fspath(path))
        except Exception:
            raise SandboxViolation(f"sandbox: unreadable path argument to {event}")
        # every allowlist entry ends in a separator, so append one to `p` too (an allowed
        # directory must match its own prefix)
        if (p + os.sep).startswith(_ALLOW) or p == os.devnull:
            return
        raise SandboxViolation(
            "sandbox: this process may not read files or directories outside the interpreter's "
            "own library directories. The candidate is given every array it is entitled to as an "
            "argument; there is nothing else for it to read.")
    if event == "sys.addaudithook":
        raise SandboxViolation("sandbox: audit hooks may not be added by a candidate program")
    if event.startswith(_DENY_EVENTS):
        raise SandboxViolation(f"sandbox: '{event}' is not available inside a candidate program")
    if event == "ctypes.dlopen":
        name = args[0] if args else None
        base = "" if name is None else os.path.basename(str(name))
        # numba/llvmlite open concrete .so files, which stays allowed; the libc handle
        # (CDLL(None) / CDLL("libc.so.6")) would reach open(2) below the audit hook
        if name is None or base.startswith("libc.so") or base in ("libc", "libc.so"):
            raise SandboxViolation("sandbox: opening the libc handle through ctypes is not allowed")


sys.addaudithook(_hook)                             # append-only: CPython cannot remove this


# ---------------------------------------------------------------------------------------------
# 3. run the candidate
# ---------------------------------------------------------------------------------------------
class _Deadline(BaseException):
    """Per-dataset wall-clock stop. BaseException so candidate try/except Exception cannot eat it."""


def _emit(obj):
    # written into the working directory, the one directory the audit hook allows
    data = pickle.dumps(obj, protocol=4)
    with open("out.pkl", "wb") as fh:
        fh.write(data)


def main():
    import signal
    import numpy as np

    res = {"scores": {}, "errors": {}, "timings": {}, "param_count": 0,
           "grad_tensor_numel": 0, "torch_imported": False, "note": "ok"}

    ns = {"__name__": "candidate_program", "__builtins__": __builtins__,
          "np": np, "numpy": np, "RANDOM_SEED": 42, "SEED": 42}
    try:
        exec(compile(_PAYLOAD["code"], "<candidate>", "exec"), ns)
    except SandboxViolation as e:
        res["note"] = f"SANDBOX {e}"
        _emit(res)
        return
    except BaseException as e:                       # noqa: BLE001 - report, never propagate
        res["note"] = f"EXC-at-import {type(e).__name__}: {str(e)[:300]}"
        _emit(res)
        return

    cls = ns.get("CustomAnomalyDetector")
    if cls is None or not callable(cls):
        res["note"] = "no callable CustomAnomalyDetector defined"
        _emit(res)
        return

    tmo = int(_PAYLOAD.get("per_dataset_timeout", 120))

    def _alarm(signum, frame):
        raise _Deadline("per-dataset time limit exceeded")

    signal.signal(signal.SIGALRM, _alarm)
    last_det = None
    for d in _PAYLOAD["datasets"]:
        did, t0 = d["id"], time.time()
        try:
            signal.alarm(tmo)
            det = cls()                              # a fresh detector per dataset, as the task does
            last_det = det
            det.fit(np.array(d["Xtr"], dtype=np.float64, copy=True))
            s = det.decision_function(np.array(d["Xte"], dtype=np.float64, copy=True))
            signal.alarm(0)
            s = np.asarray(s, dtype=np.float64).ravel()
            res["scores"][did] = s
        except SandboxViolation as e:
            signal.alarm(0)
            res["errors"][did] = f"SANDBOX {e}"
        except _Deadline as e:
            signal.alarm(0)
            res["errors"][did] = f"TIMEOUT {e}"
        except BaseException as e:                   # noqa: BLE001
            signal.alarm(0)
            res["errors"][did] = f"{type(e).__name__}: {str(e)[:300]}"
        finally:
            signal.alarm(0)
            res["timings"][did] = round(time.time() - t0, 3)

    res["param_count"], res["grad_tensor_numel"], res["torch_imported"] = _count_torch(last_det)
    _emit(res)


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:                       # noqa: BLE001
        try:
            _emit({"scores": {}, "errors": {}, "timings": {}, "param_count": 0,
                   "grad_tensor_numel": 0, "torch_imported": "torch" in sys.modules,
                   "note": f"CHILD-FATAL {type(e).__name__}: {str(e)[:300]}"})
        except Exception:
            pass
        sys.exit(1)
