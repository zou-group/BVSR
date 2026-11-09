import argparse
import glob
import json
import math
import os
import pickle
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
CHILD = os.path.join(HERE, "assets", "mls_anomaly_child.py")
_DATA_ROOT = os.environ.get("BVSR_DATA", "data")

DATA_DIR = os.path.abspath(os.environ.get("MLS_ANOMALY_DATA") or os.path.join(_DATA_ROOT, "adbench"))
PYLIBS = os.path.abspath(os.environ.get("MLS_ANOMALY_PYLIBS")
                         or os.path.join(_DATA_ROOT, "mls_anomaly_pylibs"))
DATASET_FILES = {"cardio": "6_cardio.npz", "thyroid": "38_thyroid.npz",
                 "satellite": "30_satellite.npz", "shuttle": "32_shuttle.npz"}
ORDER = ["cardio", "thyroid", "satellite", "shuttle"]
SEED = 42
TRAIN_RATIO = 0.6

PER_DATASET_TIMEOUT = 120
PER_CAND_TIMEOUT = 300
CHILD_MEM_GB = 16
BWRAP = "/usr/bin/bwrap"
ENV_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
TORCH_PARAM_CAP = 10000
FAIL = 0.0

_REFUSE = [
    (r"\.npz\b", "reads a dataset archive"),
    (r"\bnp\.load\s*\(|\bnumpy\.load\s*\(", "reads a dataset archive"),
    (r"\bloadmat\b", "reads a dataset archive"),
    (r"\badbench\b", "reaches for the dataset directory"),
    (r"\bDATASET_FILES\b|\bDATA_ROOT\b|\bDATA_DIR\b", "reaches for the dataset directory"),
    (r"\bimport\s+subprocess\b|\bsubprocess\.", "spawns a process"),
    (r"\bos\.system\s*\(|\bos\.popen\s*\(|\bos\.exec", "spawns a process"),
    (r"\bimport\s+socket\b|\bsocket\.socket\b|\burllib\b|\brequests\.get\b", "opens the network"),
    (r"ctypes\.(?:CDLL|cdll|pythonapi)", "reaches below the interpreter"),
    (r"sklearn\.metrics\.\w+\s*=[^=]", "rebinds a scoring function"),
    (r"\bbuiltins\.\w+\s*=[^=]", "rebinds a scoring function"),
    (r"\b(?:roc_auc_score|f1_score)\s*=\s*(?:lambda|def\b)", "rebinds a scoring function"),
    (r"\bdef\s+(?:roc_auc_score|f1_score)\s*\(", "rebinds a scoring function"),
]


def refuse_reason(code):
    for pat, why in _REFUSE:
        m = re.search(pat, code)
        if m:
            return f"REFUSED: candidate {why} ({m.group(0)!r}); a scoring rule needs nothing but " \
                   f"the arrays it is handed"
    return None


_CAL = {
    "auroc_cardio":    {"floor": 0.620793, "gamma": 4.392775881909392},
    "f1_cardio":       {"floor": 0.157143, "gamma": 1.0248488417416606},
    "auroc_thyroid":   {"floor": 0.869822, "gamma": 4.324524457111717},
    "f1_thyroid":      {"floor": 0.173684, "gamma": 1.0275608362565405},
    "auroc_satellite": {"floor": 0.516096, "gamma": 0.7730721001320184},
    "f1_satellite":    {"floor": 0.357494, "gamma": 0.7025753222812635},
    "auroc_shuttle":   {"floor": 0.548255, "scale": 0.4080138230962531},
    "f1_shuttle":      {"floor": 0.143162, "scale": 0.7488082997845678},
}
_GMEAN_EPS = 0.01


def _term_score(metric, x):
    c = _CAL[metric]
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    floor = c["floor"]
    if "scale" in c:
        if x <= floor:
            return 0.0
        z = (x - floor) / c["scale"]
        return 1.0 if z > 30 else 2.0 / (1.0 + math.exp(-z)) - 1.0
    r = max(0.0, min(1.0, (x - floor) / (1.0 - floor)))
    return r ** c["gamma"]


def official_score(metrics, datasets=ORDER):
    per_setting = {}
    for d in datasets:
        terms = []
        for m in (f"auroc_{d}", f"f1_{d}"):
            s = _term_score(m, metrics.get(m))
            if s is None:
                return 0.0, {}
            terms.append(s)
        per_setting[d] = sum(terms) / len(terms)
    vals = list(per_setting.values())
    if all(v <= 0.0 for v in vals):
        return 0.0, per_setting
    g = math.exp(sum(math.log(max(v, _GMEAN_EPS)) for v in vals) / len(vals))
    return float(g), per_setting


_SPLITS = {}


def splits():
    if _SPLITS:
        return _SPLITS
    for name in ORDER:
        d = np.load(os.path.join(DATA_DIR, DATASET_FILES[name]), allow_pickle=True)
        X = d["X"].astype(np.float64)
        y = (d["y"].astype(np.int32).ravel() > 0).astype(np.int32)
        Xtr, Xte, _ytr, yte = train_test_split(
            X, y, test_size=1.0 - TRAIN_RATIO, stratify=y, random_state=SEED)
        sc = StandardScaler()
        _SPLITS[name] = (sc.fit_transform(Xtr), sc.transform(Xte), yte)
    return _SPLITS


def score_one_dataset(scores, y_test):
    try:
        auroc = float(roc_auc_score(y_test, scores))
    except ValueError:
        auroc = 0.5
    contamination = float(y_test.mean())
    if 0.0 < contamination < 1.0:
        threshold = np.percentile(scores, 100 * (1 - contamination))
        y_pred = (scores >= threshold).astype(int)
    else:
        y_pred = np.zeros_like(y_test)
    return auroc, float(f1_score(y_test, y_pred, zero_division=0.0))


def _child_env():
    keep = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL", "TZ") if k in os.environ}
    keep.update({
        "HOME": "/nonexistent", "PYTHONPATH": PYLIBS, "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4", "JOBLIB_MULTIPROCESSING": "0",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": os.path.join(PYLIBS, "_cache", "mpl"),
        "NUMBA_CACHE_DIR": os.path.join(PYLIBS, "_cache", "numba"),
    })
    return keep


def _limits():
    import resource
    lim = CHILD_MEM_GB * 1024 ** 3
    for what, val in ((resource.RLIMIT_AS, (lim, lim)), (resource.RLIMIT_CORE, (0, 0))):
        try:
            resource.setrlimit(what, val)
        except (ValueError, OSError):
            pass


def _bwrap_prefix(work):
    cmd = [BWRAP, "--die-with-parent", "--new-session",
           "--unshare-net", "--unshare-ipc", "--unshare-uts", "--unshare-pid",
           "--unshare-cgroup-try"]
    for pth in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/sys", "/opt"):
        if os.path.exists(pth):
            cmd += ["--ro-bind", pth, pth]
    cmd += ["--proc", "/proc", "--dev-bind", "/dev", "/dev",
            "--tmpfs", "/dev/shm", "--tmpfs", "/tmp", "--tmpfs", "/var/tmp", "--tmpfs", "/run"]
    cmd += ["--ro-bind", ENV_ROOT, ENV_ROOT]
    if os.path.isdir(PYLIBS):
        cmd += ["--ro-bind", PYLIBS, PYLIBS]
    cmd += ["--bind", work, "/work", "--chdir", "/work"]
    return cmd


def run_candidate(code):
    reason = refuse_reason(code)
    if reason:
        return FAIL, {"refused": 1}, reason

    S = splits()
    rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
    perms, payload_ds = {}, []
    for i, name in enumerate(ORDER):
        Xtr, Xte, _ = S[name]
        pte = rng.permutation(Xte.shape[0])
        perms[f"d{i}"] = (name, pte)
        payload_ds.append({"id": f"d{i}",
                           "Xtr": np.ascontiguousarray(Xtr),
                           "Xte": np.ascontiguousarray(Xte[pte])})
    work = tempfile.mkdtemp(prefix="mlsanom_")
    child_path = os.path.join(work, "child.py")
    shutil.copyfile(CHILD, child_path)
    inp = os.path.join(work, "in.pkl")
    outp = os.path.join(work, "out.pkl")
    payload = {"code": code, "datasets": payload_ds,
               "per_dataset_timeout": PER_DATASET_TIMEOUT,
               "want_torch": "torch" in code,
               "allow_extra": [os.path.join(PYLIBS, "_cache", "mpl"),
                               os.path.join(PYLIBS, "_cache", "numba")]}
    with open(inp, "wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    logf = open(os.path.join(work, "log.txt"), "w+")
    cmd = _bwrap_prefix(work) + ["--", sys.executable, "/work/child.py"]
    t0 = time.time()
    rc, note_extra = None, ""
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=logf, stderr=logf,
                         cwd=work, env=_child_env(), preexec_fn=_limits, start_new_session=True)
    try:
        rc = p.wait(timeout=PER_CAND_TIMEOUT)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
        p.wait(timeout=30)
        note_extra = f"TIMEOUT after {PER_CAND_TIMEOUT}s"
    elapsed = time.time() - t0

    try:
        with open(outp, "rb") as fh:
            blob = fh.read()
    except OSError:
        blob = b""
    logf.seek(0)
    tail = " | ".join(logf.read().strip().splitlines()[-3:])[-300:]
    logf.close()
    shutil.rmtree(work, ignore_errors=True)

    if not blob:
        n = note_extra or f"child produced no result (rc={rc})"
        return FAIL, {"elapsed_s": round(elapsed, 2)}, f"{n} [{tail}]"
    try:
        res = pickle.loads(blob)
    except Exception as e:
        return FAIL, {"elapsed_s": round(elapsed, 2)}, f"unreadable child result: {e} [{tail}]"

    metrics = {"elapsed_s": round(elapsed, 2),
               "torch_params": int(res.get("param_count", 0)),
               "torch_grad_tensor_numel": int(res.get("grad_tensor_numel", 0)),
               "torch_imported": bool(res.get("torch_imported", False))}
    notes = []
    if note_extra:
        notes.append(note_extra)
    if res.get("note") and res["note"] != "ok":
        notes.append(res["note"])

    if metrics["torch_params"] > TORCH_PARAM_CAP:
        notes.append(f"PARAM-CAP {metrics['torch_params']} > {TORCH_PARAM_CAP}")
        return FAIL, metrics, "; ".join(notes)

    n_ok = 0
    for did, (name, pte) in perms.items():
        _Xtr, _Xte, yte = S[name]
        s = res.get("scores", {}).get(did)
        err = res.get("errors", {}).get(did)
        metrics[f"elapsed_{name}"] = res.get("timings", {}).get(did)
        if s is None:
            notes.append(f"{name}: {err or 'no scores'}")
            continue
        s = np.asarray(s, dtype=np.float64).ravel()
        if s.shape[0] != yte.shape[0]:
            notes.append(f"{name}: wrong shape {s.shape} != ({yte.shape[0]},)")
            continue
        if not np.all(np.isfinite(s)):
            notes.append(f"{name}: non-finite scores")
            continue
        aligned = np.empty_like(s)
        aligned[pte] = s
        auroc, f1 = score_one_dataset(aligned, yte)
        metrics[f"auroc_{name}"], metrics[f"f1_{name}"] = round(auroc, 6), round(f1, 6)
        n_ok += 1

    official, per_setting = official_score(metrics)
    metrics["official_score"] = round(official, 6)
    for k, v in per_setting.items():
        metrics[f"setting_{k}"] = round(v, 6)
    if n_ok:
        metrics["raw_mean"] = round(float(np.mean(
            [metrics[f"{m}_{d}"] for d in ORDER for m in ("auroc", "f1")
             if f"{m}_{d}" in metrics])), 6)
        metrics["auroc_mean_ok"] = round(float(np.mean(
            [metrics[f"auroc_{d}"] for d in ORDER if f"auroc_{d}" in metrics])), 6)
    metrics["n_datasets_ok"] = n_ok
    metrics["suspect_near_perfect"] = int(
        sum(1 for d in ORDER if metrics.get(f"auroc_{d}", 0.0) >= 0.995) >= 3
        or metrics.get("auroc_satellite", 0.0) >= 0.95)
    return float(official), metrics, ("; ".join(notes) if notes else "ok")


def _atomic_write(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def _score_file(cand, out):
    code = open(cand, errors="ignore").read()
    try:
        reward, metrics, note = run_candidate(code)
    except Exception as e:
        reward, metrics, note = FAIL, {}, f"SCORER-EXC {type(e).__name__}: {str(e)[:300]}"
    rec = {"file": os.path.basename(cand), "reward": reward,
           "official_score": metrics.get("official_score", reward),
           "metrics": metrics, "note": note}
    _atomic_write(out, rec)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one")
    ap.add_argument("--out")
    a = ap.parse_args()
    if not os.path.exists(BWRAP):
        sys.exit(f"[mls_anomaly] bwrap not found at {BWRAP}. Refusing to evaluate candidates.")
    if a.one:
        rec = _score_file(a.one, a.out or (a.one[:-3] + ".json"))
        print(f"[mls_anomaly] {rec['file']} reward={rec['reward']:.4f} | {rec['note']}", flush=True)
        return
    SH = int(os.environ.get("SHARD", "0"))
    NS = int(os.environ.get("NSHARD", "1"))
    CAND, RES = os.environ["CAND_DIR"], os.environ["RES_DIR"]
    os.makedirs(RES, exist_ok=True)
    mine = sorted(glob.glob(os.path.join(CAND, "*.py")))[SH::NS]
    print(f"[mls_anomaly {SH}/{NS}] {len(mine)} cands", flush=True)
    for i, f in enumerate(mine):
        out = os.path.join(RES, os.path.basename(f)[:-3] + ".json")
        if os.path.exists(out):
            continue
        rec = _score_file(f, out)
        print(f"[mls_anomaly {SH}] {i+1}/{len(mine)} {rec['file']} "
              f"reward={rec['reward']:.4f} | {rec['note']}", flush=True)
    print(f"[mls_anomaly {SH}] DONE", flush=True)


if __name__ == "__main__":
    main()
