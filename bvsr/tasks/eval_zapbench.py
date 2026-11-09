import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid

import numpy as np

_DATA_ROOT = os.environ.get("BVSR_DATA", "data")
CACHE = os.path.abspath(os.environ.get("ZAP_CACHE") or os.path.join(_DATA_ROOT, "zapbench", "cache"))
WORK = os.path.abspath(os.environ.get("ZAP_WORK") or os.path.join(_DATA_ROOT, "zapbench", "work"))
SANDBOX_PY = sys.executable
BWRAP = "/usr/bin/bwrap"

BUDGET = float(os.environ.get("ZAP_TIME_BUDGET", "600"))
TIMEOUT = BUDGET * 1.5 + 300
SCORE_CHUNK = 16
FAILURE = -10.0

HORIZONS = {"zap_h16": 16, "zap_h32": 32}
REF = {16: {"copy_oracle": 0.005318, "persistence": 0.031578, "mean_context": 0.028914},
       32: {"copy_oracle": 0.011612, "persistence": 0.036258, "mean_context": 0.033450}}

RUNNER = r'''
import json, os, resource, sys, time, traceback

_lim = int(os.environ["ZAP_MEM_BYTES"])
for _r in (resource.RLIMIT_DATA,):
    try:
        resource.setrlimit(_r, (_lim, _lim))
    except Exception as _e:
        print(f"[runner] setrlimit {_r} failed: {_e}", file=sys.stderr, flush=True)

import numpy as np
import importlib.util

D, R = "/zapdata", "/zaprun"
H = int(os.environ["ZAP_HORIZON"])
BUD = float(os.environ["ZAP_BUDGET"])

try:
    X_train = np.load(D + "/X_train.npy")
    lengths = np.load(D + "/X_train_condition_lengths.npy")
    predict_inputs = np.load(D + "/val_predict_inputs.npy")

    spec = importlib.util.spec_from_file_location("user_prog", R + "/candidate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_prog"] = mod
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "forecast", None)):
        raise RuntimeError("program must define a callable forecast()")

    t0 = time.time()
    Y = mod.forecast(
        X_train, predict_inputs,
        X_train_condition_lengths=lengths,
        random_state=42,
        time_budget_s=BUD,
        num_train_conditions=int(lengths.shape[0]),
        horizon=H,
        context_len=int(predict_inputs.shape[1]),
    )
    dt = time.time() - t0

    Y = np.ascontiguousarray(np.asarray(Y, dtype=np.float32))
    want = (predict_inputs.shape[0], H, predict_inputs.shape[2])
    if Y.shape != want:
        raise ValueError(f"Y_pred shape {Y.shape} != expected {want}")
    if not np.isfinite(Y).all():
        raise ValueError("Y_pred contains non-finite values")

    np.save(R + "/Y_pred.npy", Y, allow_pickle=False)
    with open(R + "/meta.json", "w") as f:
        json.dump({"forecast_seconds": dt}, f)
except BaseException as e:
    traceback.print_exc()
    try:
        with open(R + "/error.txt", "w") as f:
            f.write(f"{type(e).__name__}: {str(e)[:400]}")
    except Exception:
        pass
    sys.exit(1)
'''

PROBE = r'''
import os, sys, glob
bad = []
for p in [q for q in os.environ.get("ZAP_PROBE_PATHS", "").split(os.pathsep) if q]:
    if os.path.exists(p):
        bad.append("VISIBLE:" + p)
seen = sorted(os.path.basename(p) for p in glob.glob("/zapdata/*"))
if seen != ["X_train.npy", "X_train_condition_lengths.npy", "val_predict_inputs.npy"]:
    bad.append("ZAPDATA:" + ",".join(seen))
skip = ("/proc", "/sys", "/dev", "/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/opt",
        os.environ.get("ZAP_ENV_ROOT", "/nonexistent"))
hits, nvisited = [], 0
for root, dirs, files in os.walk("/"):
    if root.startswith(skip):
        dirs[:] = []
        continue
    nvisited += 1
    if nvisited > 20000:
        break
    for fn in files:
        if fn.endswith(".npy") and fn not in (
                "X_train.npy", "X_train_condition_lengths.npy", "val_predict_inputs.npy"):
            try:
                if os.path.getsize(os.path.join(root, fn)) > 64 * 1024 * 1024:
                    hits.append(os.path.join(root, fn))
            except OSError:
                pass
if hits:
    bad.append("BIG_NPY:" + ",".join(hits[:3]))
try:
    import socket
    socket.create_connection(("8.8.8.8", 53), timeout=2)
    bad.append("NETWORK_UP")
except Exception:
    pass
import numpy as np
sys.stdout.write("PROBE:" + (",".join(bad) if bad else "CLEAN") + "\n")
'''


def _mem_bytes():
    return int(float(os.environ.get("ZAP_MEM_GB") or "43.2") * 1024 ** 3)


def _bwrap_cmd(workdir, script_name, horizon):
    env_root = os.path.dirname(os.path.dirname(os.path.abspath(SANDBOX_PY)))
    cmd = [BWRAP, "--die-with-parent", "--new-session",
           "--unshare-net", "--unshare-ipc", "--unshare-uts", "--unshare-pid", "--unshare-cgroup-try"]
    for p in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/sys", "/opt"):
        if os.path.exists(p):
            cmd += ["--ro-bind", p, p]
    cmd += ["--proc", "/proc", "--dev-bind", "/dev", "/dev", "--tmpfs", "/dev/shm",
            "--tmpfs", "/tmp", "--tmpfs", "/var", "--tmpfs", "/run"]
    if os.path.exists("/run/nvidia-persistenced"):
        cmd += ["--ro-bind", "/run/nvidia-persistenced", "/run/nvidia-persistenced"]
    cmd += ["--ro-bind", env_root, env_root]
    for f in ("X_train.npy", "X_train_condition_lengths.npy", "val_predict_inputs.npy"):
        cmd += ["--ro-bind", os.path.join(CACHE, f), "/zapdata/" + f]
    cmd += ["--bind", workdir, "/zaprun", "--chdir", "/zaprun"]
    for k, v in (("HOME", "/zaprun"), ("TMPDIR", "/tmp"), ("PYTHONDONTWRITEBYTECODE", "1"),
                 ("PYTHONHASHSEED", "0"), ("MPLCONFIGDIR", "/tmp")):
        cmd += ["--setenv", k, v]
    threads = os.environ.get("ZAP_THREADS", "4")
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        cmd += ["--setenv", k, threads]
    for k in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
        if os.environ.get(k) is not None:
            cmd += ["--setenv", k, os.environ[k]]
    cmd += ["--setenv", "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"]
    cmd += ["--setenv", "ZAP_MEM_BYTES", str(_mem_bytes())]
    cmd += ["--setenv", "ZAP_ENV_ROOT", env_root]
    cmd += ["--setenv", "ZAP_PROBE_PATHS",
            os.pathsep.join(os.path.realpath(p) for p in (CACHE, WORK))]
    cmd += ["--setenv", "ZAP_HORIZON", str(horizon)]
    cmd += ["--setenv", "ZAP_BUDGET", str(BUDGET)]
    cmd += ["--", SANDBOX_PY, "-u", "/zaprun/" + script_name]
    return cmd


def _selftest(horizon):
    if not os.path.exists(BWRAP):
        return False, f"bwrap not found at {BWRAP}"
    d = os.path.join(WORK, f"_probe_{os.getpid()}_{uuid.uuid4().hex[:8]}")
    os.makedirs(d, exist_ok=True)
    try:
        with open(os.path.join(d, "probe.py"), "w") as f:
            f.write(PROBE)
        r = subprocess.run(_bwrap_cmd(d, "probe.py", horizon), capture_output=True, text=True,
                           timeout=300)
        out = (r.stdout or "") + (r.stderr or "")
        if "PROBE:CLEAN" in out:
            return True, "clean"
        return False, f"rc={r.returncode} out={out[-600:]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _run_candidate(cand_path, workdir, horizon):
    os.makedirs(workdir, exist_ok=True)
    shutil.copyfile(cand_path, os.path.join(workdir, "candidate.py"))
    with open(os.path.join(workdir, "runner.py"), "w") as f:
        f.write(RUNNER)
    cmd = _bwrap_cmd(workdir, "runner.py", horizon)
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         start_new_session=True)
    try:
        _out, err = p.communicate(timeout=TIMEOUT)
        wall = time.time() - t0
        err = (err or b"").decode(errors="replace")
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
        p.communicate()
        return None, f"Timeout: candidate exceeded {TIMEOUT:.0f}s wall clock", 0.0, time.time() - t0, ""
    pred = os.path.join(workdir, "Y_pred.npy")
    ep = os.path.join(workdir, "error.txt")
    fs = 0.0
    mp = os.path.join(workdir, "meta.json")
    if os.path.exists(mp):
        try:
            fs = float(json.load(open(mp)).get("forecast_seconds", 0.0))
        except Exception:
            pass
    if os.path.exists(ep):
        return None, open(ep).read()[:400], fs, wall, err[-1200:]
    if not os.path.exists(pred):
        return None, f"no Y_pred written (rc={p.returncode})", fs, wall, err[-1200:]
    return pred, None, fs, wall, err[-1200:]


_COND_OFFSETS = (0, 649, 2422, 3078, 3735, 5047, 5638, 6623, 7279, 7879)
_COND_PADDING = 1
_CONDITIONS_TRAIN = (0, 1, 2, 4, 5, 6, 7, 8)
_TEST_FRACTION = 0.2
_VAL_FRACTION = 0.1
_PRED_WIN_LEN = 32


def _window_counts_from_constants(context_len):
    counts = []
    for c in _CONDITIONS_TRAIN:
        inc0 = _COND_OFFSETS[c] + _COND_PADDING
        exc0 = _COND_OFFSETS[c + 1] - _COND_PADDING
        total = exc0 - inc0
        n_test = int(total * _TEST_FRACTION)
        n_val = int(total * _VAL_FRACTION)
        n_train = total - n_test - n_val
        inc = inc0 + n_train - context_len
        exc = inc0 + n_train + n_val
        n = (exc - inc) - _PRED_WIN_LEN - context_len + 1
        if n > 0:
            counts.append(int(n))
    return counts


def _leak_src(meta, N, H):
    wc = meta.get("window_counts_per_condition")
    if not wc:
        wc = _window_counts_from_constants(int(meta.get("context_len", 4)))
    wc = np.asarray(wc, dtype=np.int64)
    if int(wc.sum()) != N:
        raise ValueError(f"window counts per condition sum to {int(wc.sum())}, but the cache holds "
                         f"{N} validation windows")
    off = np.concatenate([[0], np.cumsum(wc)])
    cond = np.zeros(N, np.int64)
    local = np.zeros(N, np.int64)
    ncond = np.zeros(N, np.int64)
    for c in range(len(wc)):
        s, e = int(off[c]), int(off[c + 1])
        cond[s:e] = c
        local[s:e] = np.arange(e - s)
        ncond[s:e] = wc[c]
    h = np.arange(1, H + 1, dtype=np.int64)[None, :]
    ok = (local[:, None] + h) < ncond[:, None]
    return np.where(ok, off[cond][:, None] + local[:, None] + h, -1)


def score(pred_path, H, meta):
    T = np.load(os.path.join(CACHE, "val_targets.npy"), mmap_mode="r")
    X = np.load(os.path.join(CACHE, "val_predict_inputs.npy"), mmap_mode="r")
    N, F = T.shape[0], T.shape[2]
    want = (N, H, F)

    nbytes = os.path.getsize(pred_path)
    if nbytes > N * H * F * 4 + 4096:
        raise ValueError(f"Y_pred file too large: {nbytes} bytes")
    P = np.load(pred_path, mmap_mode="r", allow_pickle=False)
    if P.shape != want:
        raise ValueError(f"Y_pred shape {P.shape} != expected {want}")
    if P.dtype != np.float32:
        raise ValueError(f"Y_pred dtype {P.dtype} != float32")

    last = np.asarray(X[:, -1, :], dtype=np.float32)
    src = _leak_src(meta, N, H)

    s_all = 0.0
    s_av = 0.0
    s_un = 0.0
    n_av = 0
    n_un = 0
    step = np.zeros(H, np.float64)
    m4 = 0
    m2 = 0
    pm4 = 0
    sp = sc = spp = scc = spc = 0.0
    sq = sr = sqq = srr = sqr = 0.0
    n_pers = 0
    for i in range(0, N, SCORE_CHUNK):
        j = min(i + SCORE_CHUNK, N)
        yp = np.asarray(P[i:j], dtype=np.float64)
        if not np.isfinite(yp).all():
            raise ValueError("Y_pred contains non-finite values")
        yt = np.asarray(T[i:j, :H], dtype=np.float64)
        d = np.abs(yp - yt)
        s_all += float(d.sum())
        step += d.sum(axis=(0, 2))

        av = src[i:j] >= 0
        idx = src[i:j][av]
        cp = np.asarray(last[idx], dtype=np.float64)
        pv = yp[av]
        s_av += float(d[av].sum())
        s_un += float(d[~av].sum())
        n_av += int(av.sum())
        n_un += int((~av).sum())
        dc = np.abs(pv - cp)
        m4 += int((dc <= 1e-4).sum())
        m2 += int((dc <= 1e-2).sum())
        sp += float(pv.sum())
        sc += float(cp.sum())
        spp += float((pv * pv).sum())
        scc += float((cp * cp).sum())
        spc += float((pv * cp).sum())

        pers = np.broadcast_to(np.asarray(last[i:j], np.float64)[:, None, :], yp.shape)
        pm4 += int((np.abs(yp - pers) <= 1e-4).sum())
        sq += float(yp.sum())
        sr += float(pers.sum())
        sqq += float((yp * yp).sum())
        srr += float((pers * pers).sum())
        sqr += float((yp * pers).sum())
        n_pers += yp.size

    n_all = N * H * F

    def _corr(n, sx, sy, sxx, syy, sxy):
        if n < 2:
            return 0.0
        cov = sxy / n - (sx / n) * (sy / n)
        vx = max(sxx / n - (sx / n) ** 2, 0.0)
        vy = max(syy / n - (sy / n) ** 2, 0.0)
        return float(cov / ((vx * vy) ** 0.5)) if vx > 0 and vy > 0 else 0.0

    mae = s_all / n_all
    return {
        "mae": mae,
        "score": -mae,
        "mae_by_step": [float(x / (N * F)) for x in step],
        "mae_avail": (s_av / (n_av * F)) if n_av else None,
        "mae_unavail": (s_un / (n_un * F)) if n_un else None,
        "avail_frac": float(n_av) / float(n_av + n_un),
        "leak_match_frac_1e-4": float(m4) / float(max(n_av * F, 1)),
        "leak_match_frac_1e-2": float(m2) / float(max(n_av * F, 1)),
        "leak_copy_corr": _corr(n_av * F, sp, sc, spp, scc, spc),
        "persist_match_frac_1e-4": float(pm4) / float(n_all),
        "persist_corr": _corr(n_pers, sq, sr, sqq, srr, sqr),
        "mae_copy_oracle_ref": REF[H]["copy_oracle"],
        "mae_persistence_ref": REF[H]["persistence"],
        "mae_mean_context_ref": REF[H]["mean_context"],
        "n_windows": int(N), "n_neurons": int(F), "horizon": int(H),
    }


def main():
    task = os.environ.get("ZAPTASK", "zap_h32")
    if task not in HORIZONS:
        sys.exit(f"ZAPTASK must be one of {sorted(HORIZONS)}, got {task!r}")
    H = HORIZONS[task]
    cand_dir = os.environ["CAND_DIR"]
    res_dir = os.environ["RES_DIR"]
    shard = int(os.environ.get("SHARD", "0"))
    nshard = int(os.environ.get("NSHARD", "1"))
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    meta = json.load(open(os.path.join(CACHE, "mean_baseline_mae_val.json")))
    _leak_src(meta, np.load(os.path.join(CACHE, "val_targets.npy"), mmap_mode="r").shape[0], H)

    ok, why = _selftest(H)
    if not ok:
        sys.exit(f"[zap] SANDBOX SELF-TEST FAILED ({why}). Refusing to evaluate candidates.")
    print(f"[zap] sandbox ok; task={task} H={H} budget={BUDGET:.0f}s timeout={TIMEOUT:.0f}s "
          f"shard {shard}/{nshard}", flush=True)

    cands = sorted(glob.glob(os.path.join(cand_dir, "*.py")))
    for i, c in enumerate(cands):
        if i % nshard != shard:
            continue
        stem = os.path.basename(c)[:-3]
        out = os.path.join(res_dir, stem + ".json")
        if os.path.exists(out):
            continue
        wd = os.path.join(WORK, f"{task}_{stem[:48]}_{os.getpid()}_{uuid.uuid4().hex[:8]}")
        shutil.rmtree(wd, ignore_errors=True)
        rec = {"file": os.path.basename(c), "reward": FAILURE,
               "metrics": {"validity": 0.0, "horizon": H, "task": task}}
        t0 = time.time()
        try:
            pred, err, fsec, wall, tail = _run_candidate(c, wd, H)
            rec["metrics"]["forecast_seconds"] = fsec
            rec["metrics"]["child_seconds"] = wall
            if err is not None:
                rec["metrics"]["error"] = err
                if tail:
                    rec["metrics"]["stderr_tail"] = tail[-600:]
            else:
                m = score(pred, H, meta)
                rec["reward"] = float(m["score"])
                m["validity"] = 1.0
                m["forecast_seconds"] = fsec
                m["child_seconds"] = wall
                m["task"] = task
                rec["metrics"] = m
        except Exception as e:
            rec["metrics"]["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        finally:
            shutil.rmtree(wd, ignore_errors=True)
        rec["metrics"]["eval_seconds"] = time.time() - t0
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f)
        os.replace(tmp, out)
        mm = rec["metrics"]
        print(f"[{shard}/{nshard}] {stem}: reward={rec['reward']:.5f} "
              f"mae={mm.get('mae', float('nan')):.6f} "
              f"avail={mm.get('mae_avail', float('nan'))} unavail={mm.get('mae_unavail', float('nan'))} "
              f"leak1e-4={mm.get('leak_match_frac_1e-4', float('nan'))} "
              f"t={mm['eval_seconds']:.0f}s {mm.get('error', '')}", flush=True)


if __name__ == "__main__":
    main()
