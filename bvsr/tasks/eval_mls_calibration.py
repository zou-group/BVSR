import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import warnings

import numpy as np

warnings.filterwarnings("ignore")

CACHE_DIR = os.environ.get("MLSCAL_CACHE") or os.path.join(
    os.environ.get("BVSR_DATA", "data"), "mls_calibration_cache")
SKLEARN_DATA_HOME = os.path.join(CACHE_DIR, "sklearn_data")
SEED = 42
BWRAP = "/usr/bin/bwrap"
ENV_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
MARKER = "class CalibrationMethod"
FAIL = 0.0

SETTINGS = [
    ("rf-mnist", "rf", "mnist"),
    ("mlp-fashion_mnist", "mlp", "fashion_mnist"),
    ("svm-breast_cancer", "svm", "breast_cancer"),
    ("gbm-madelon", "gbm", "madelon"),
]

LEAK_PATTERNS = (
    "sklearn.datasets", "from sklearn import datasets", "fetch_openml", "load_breast_cancer",
    "load_dataset", "openml", "mls_calibration_cache", "mlscal_cache", "y_test", "test_labels",
    "eval_shard", "leaderboard", "score_spec", "custom_calibration", "_scaffold",
)


def expected_calibration_error(probs, labels, n_bins=15):
    if probs.ndim == 2:
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
    else:
        confidences = np.where(probs >= 0.5, probs, 1 - probs)
        predictions = (probs >= 0.5).astype(int)
        accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        prop = in_bin.mean()
        if prop > 0:
            avg_conf = confidences[in_bin].mean()
            avg_acc = accuracies[in_bin].mean()
            ece += prop * abs(avg_acc - avg_conf)
    return ece


def compute_brier_score(probs, labels, n_classes):
    from sklearn.metrics import brier_score_loss
    from sklearn.preprocessing import LabelBinarizer
    if n_classes == 2 and probs.ndim == 1:
        return brier_score_loss(labels, probs)
    lb = LabelBinarizer()
    lb.classes_ = np.arange(n_classes)
    labels_onehot = lb.transform(labels)
    if n_classes == 2:
        labels_onehot = np.column_stack([1 - labels_onehot, labels_onehot])
    if probs.ndim == 1:
        probs = np.column_stack([1 - probs, probs])
    return np.mean(np.sum((probs - labels_onehot) ** 2, axis=1))


def compute_nll(probs, labels, n_classes):
    from sklearn.metrics import log_loss
    if probs.ndim == 1:
        probs_2d = np.column_stack([1 - probs, probs])
    else:
        probs_2d = probs
    probs_2d = np.clip(probs_2d, 1e-15, 1 - 1e-15)
    return log_loss(labels, probs_2d, labels=np.arange(n_classes))


def adaptive_ece(probs, labels, n_bins=15):
    probs = np.asarray(probs)
    if probs.ndim == 2:
        conf = np.max(probs, axis=1)
        acc = (np.argmax(probs, axis=1) == labels).astype(float)
    else:
        conf = np.where(probs >= 0.5, probs, 1 - probs)
        acc = ((probs >= 0.5).astype(int) == labels).astype(float)
    n = len(conf)
    if n == 0:
        return float("nan")
    order = np.argsort(conf, kind="mergesort")
    ece = 0.0
    for idx in np.array_split(order, min(n_bins, n)):
        if len(idx) == 0:
            continue
        ece += (len(idx) / n) * abs(acc[idx].mean() - conf[idx].mean())
    return float(ece)


def top1_accuracy(probs, labels):
    probs = np.asarray(probs)
    pred = np.argmax(probs, axis=1) if probs.ndim == 2 else (probs >= 0.5).astype(int)
    return float((pred == labels).mean())


def _official_load_dataset(name, seed=SEED):
    from sklearn.datasets import fetch_openml, load_breast_cancer
    from sklearn.model_selection import train_test_split
    dh = os.environ.get("SKLEARN_DATA_HOME", SKLEARN_DATA_HOME)
    if name == "mnist":
        data = fetch_openml("mnist_784", version=1, data_home=dh, parser="auto", as_frame=False)
        X, y = data["data"].astype(np.float32), data["target"].astype(int)
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), 20000, replace=False)
        X, y = X[idx], y[idx]
        X = X / 255.0
        n_classes = 10
    elif name == "fashion_mnist":
        data = fetch_openml("Fashion-MNIST", version=1, data_home=dh, parser="auto", as_frame=False)
        X, y = data["data"].astype(np.float32), data["target"].astype(int)
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), 20000, replace=False)
        X, y = X[idx], y[idx]
        X = X / 255.0
        n_classes = 10
    elif name == "breast_cancer":
        data = load_breast_cancer()
        X, y = data["data"].astype(np.float32), data["target"].astype(int)
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        X = (X - mean) / std
        n_classes = 2
    elif name == "madelon":
        data = fetch_openml("madelon", version=1, data_home=dh, parser="auto", as_frame=False)
        X = data["data"].astype(np.float32)
        y_raw = data["target"]
        unique_labels = np.unique(y_raw)
        label_map = {lab: i for i, lab in enumerate(sorted(unique_labels))}
        y = np.array([label_map[lab] for lab in y_raw])
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        X = (X - mean) / std
        n_classes = 2
    else:
        raise ValueError(f"Unknown dataset: {name}")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=seed, stratify=y_trainval)
    return X_train, y_train, X_cal, y_cal, X_test, y_test, n_classes


def _official_build_classifier(name, seed=SEED):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=2,
                                      random_state=seed, n_jobs=-1)
    if name == "svm":
        return SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=seed)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu", max_iter=200,
                             early_stopping=True, validation_fraction=0.1, random_state=seed)
    if name == "gbm":
        return GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                          random_state=seed)
    raise ValueError(f"Unknown classifier: {name}")


def cache_path(label):
    return os.path.join(CACHE_DIR, f"{label}_seed{SEED}.npz")


def build_cache(force=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    for label, clf_name, ds_name in SETTINGS:
        out = cache_path(label)
        if os.path.exists(out) and not force:
            print(f"[cache] {label}: already present", flush=True)
            continue
        t0 = time.time()
        np.random.seed(SEED)
        X_tr, y_tr, X_cal, y_cal, X_te, y_te, n_classes = _official_load_dataset(ds_name, SEED)
        clf = _official_build_classifier(clf_name, SEED)
        clf.fit(X_tr, y_tr)
        train_acc = float(clf.score(X_tr, y_tr))
        test_acc = float(clf.score(X_te, y_te))
        if n_classes == 2:
            cal_probs = clf.predict_proba(X_cal)[:, 1]
            test_probs = clf.predict_proba(X_te)[:, 1]
        else:
            cal_probs = clf.predict_proba(X_cal)
            test_probs = clf.predict_proba(X_te)
        np.savez_compressed(out + ".tmp.npz",
                            cal_probs=cal_probs, cal_labels=y_cal.astype(np.int64),
                            test_probs=test_probs, test_labels=y_te.astype(np.int64),
                            n_classes=np.int64(n_classes),
                            train_acc=np.float64(train_acc), test_acc=np.float64(test_acc))
        os.replace(out + ".tmp.npz", out)
        print(f"[cache] {label}: train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
              f"cal={cal_probs.shape} test={test_probs.shape} n_classes={n_classes} "
              f"({time.time() - t0:.1f}s) -> {out}", flush=True)


_CACHE = {}


def load_cache(label):
    if label not in _CACHE:
        p = cache_path(label)
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing cache {p}; run `--build-cache` first")
        d = np.load(p)
        _CACHE[label] = {k: d[k] for k in d.files}
    return _CACHE[label]


def _install_child_sandbox(sandbox):
    import builtins
    import io
    import importlib.abc

    sandbox = os.path.realpath(sandbox)
    prefixes = tuple(os.path.realpath(p) for p in
                     {sys.prefix, sys.base_prefix, "/usr", "/lib", "/lib64", "/etc",
                      "/proc", "/sys"})

    def _allowed(path):
        try:
            p = os.path.realpath(os.fspath(path))
        except Exception:
            return False
        low = p.lower()
        if "datasets" in low or "mls_calibration_cache" in low:
            return False
        if p == sandbox or p.startswith(sandbox + os.sep):
            return True
        return any(p == q or p.startswith(q + os.sep) for q in prefixes)

    def _guard(fn, name):
        def wrapper(file, *a, **kw):
            if isinstance(file, int) or not _allowed(file):
                raise PermissionError(
                    f"sandbox: {name}({file!r}) denied — the candidate may only touch its own "
                    "working directory")
            return fn(file, *a, **kw)
        return wrapper

    builtins.open = _guard(builtins.open, "open")
    io.open = _guard(io.open, "io.open")
    os.open = _guard(os.open, "os.open")
    for nm in ("listdir", "scandir", "walk"):
        if hasattr(os, nm):
            setattr(os, nm, _guard(getattr(os, nm), f"os.{nm}"))

    def _denied(name):
        def f(*a, **kw):
            raise PermissionError(f"sandbox: {name} is disabled inside a candidate")
        return f

    for nm in ("system", "popen", "fork", "forkpty", "execv", "execve", "execvp", "execvpe",
               "spawnv", "spawnve", "spawnvp", "spawnvpe", "posix_spawn", "posix_spawnp"):
        if hasattr(os, nm):
            setattr(os, nm, _denied(f"os.{nm}"))
    if "subprocess" in sys.modules:
        sp = sys.modules["subprocess"]
        for nm in ("run", "call", "check_call", "check_output", "getoutput", "getstatusoutput"):
            if hasattr(sp, nm):
                setattr(sp, nm, _denied(f"subprocess.{nm}"))
        if hasattr(sp, "Popen"):
            sp.Popen.__init__ = _denied("subprocess.Popen")
    if "socket" in sys.modules:
        sk = sys.modules["socket"]
        for nm in ("socket", "create_connection", "getaddrinfo", "socketpair"):
            if hasattr(sk, nm):
                setattr(sk, nm, _denied(f"socket.{nm}"))
    if "ctypes" in sys.modules:
        ct = sys.modules["ctypes"]
        for nm in ("CDLL", "PyDLL", "WinDLL", "cdll"):
            if hasattr(ct, nm):
                setattr(ct, nm, _denied(f"ctypes.{nm}"))

    BLOCKED_PREFIX = ("sklearn.datasets", "openml", "urllib", "http", "requests", "ftplib",
                      "telnetlib", "socketserver", "xmlrpc", "ctypes", "torchvision",
                      "tensorflow", "keras", "datasets", "kagglehub", "pooch", "aiohttp",
                      "httpx", "paramiko", "smtplib", "webbrowser")

    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            root = fullname.split(".")[0]
            if fullname.startswith(BLOCKED_PREFIX) or root in BLOCKED_PREFIX:
                raise ImportError(
                    f"sandbox: importing {fullname!r} is blocked — a calibration method may not "
                    "load data, reach the network, or bind foreign code")
            return None

    for mod in [m for m in list(sys.modules) if m.startswith("sklearn.datasets")]:
        sys.modules.pop(mod, None)
    sys.modules.pop("openml", None)
    keep = tuple(os.path.realpath(p) for p in {sys.prefix, sys.base_prefix})
    sys.path[:] = [p for p in sys.path if p and
                   any(os.path.realpath(p).startswith(q) for q in keep)]
    sys.meta_path.insert(0, _Blocker())


def _child_preimport():
    import warnings
    warnings.filterwarnings("ignore")
    import numpy, scipy
    from scipy import optimize, interpolate, special, stats, linalg
    import sklearn
    from sklearn import base, utils, metrics, preprocessing, isotonic, linear_model
    from sklearn import model_selection, calibration, ensemble, tree, neighbors
    from sklearn import svm, cluster, decomposition, mixture, neural_network
    from sklearn import gaussian_process, kernel_ridge, pipeline, multiclass
    return {
        "np": numpy, "numpy": numpy, "scipy": scipy, "optimize": optimize,
        "interpolate": interpolate, "special": special, "stats": stats, "linalg": linalg,
        "sklearn": sklearn, "BaseEstimator": base.BaseEstimator,
        "TransformerMixin": base.TransformerMixin, "RegressorMixin": base.RegressorMixin,
        "ClassifierMixin": base.ClassifierMixin, "math": math, "warnings": warnings,
    }


def child_main(sandbox):
    import resource
    try:
        resource.setrlimit(resource.RLIMIT_AS, (12 << 30, 12 << 30))
        resource.setrlimit(resource.RLIMIT_FSIZE, (512 << 20, 512 << 20))
    except (ValueError, OSError):
        pass

    inp = np.load(os.path.join(sandbox, "input.npz"), allow_pickle=False)
    labels_list = [str(x) for x in inp["settings"]]
    payload = {lab: (inp[f"cal_probs__{lab}"], inp[f"cal_labels__{lab}"], inp[f"test_probs__{lab}"])
               for lab in labels_list}
    with open(os.path.join(sandbox, "cand.py")) as f:
        code = f.read()

    env = _child_preimport()
    _install_child_sandbox(sandbox)

    status, preds = {}, {}
    g = dict(env)
    g["__name__"] = "__candidate__"
    g["__builtins__"] = __builtins__
    try:
        exec(compile(code, "<candidate>", "exec"), g)
    except BaseException as e:
        status["__global__"] = {"note": f"EXC {type(e).__name__}: {str(e)[:300]}",
                                "traceback": _clean_tb()}
        _write_child(sandbox, preds, status)
        return
    cls = g.get("CalibrationMethod")
    if not isinstance(cls, type):
        status["__global__"] = {"note": "candidate defines no CalibrationMethod class"}
        _write_child(sandbox, preds, status)
        return

    for lab in labels_list:
        cal_probs, cal_labels, test_probs = payload[lab]
        t0 = time.time()
        try:
            obj = cls()
            obj.fit(np.array(cal_probs, copy=True), np.array(cal_labels, copy=True))
            out = obj.predict_proba(np.array(test_probs, copy=True))
            out = np.asarray(out, dtype=np.float64)
            preds[lab] = out
            status[lab] = {"note": "ok", "elapsed": round(time.time() - t0, 3)}
        except BaseException as e:
            status[lab] = {"note": f"EXC {type(e).__name__}: {str(e)[:300]}",
                           "elapsed": round(time.time() - t0, 3), "traceback": _clean_tb()}
    _write_child(sandbox, preds, status)


def _clean_tb():
    return "".join(l for l in traceback.format_exc().splitlines(keepends=True)
                   if os.path.basename(__file__) not in l)[-3000:]


def _write_child(sandbox, preds, status):
    np.savez(os.path.join(sandbox, "out.npz"),
             **{f"pred__{k}": v for k, v in preds.items()})
    with open(os.path.join(sandbox, "status.json"), "w") as f:
        json.dump(status, f)


def _child_env(sandbox):
    keep = {}
    for k in ("PATH", "LANG", "LC_ALL", "TZ", "PYTHONHASHSEED"):
        if k in os.environ:
            keep[k] = os.environ[k]
    keep.update({
        "HOME": sandbox, "TMPDIR": sandbox, "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2", "JOBLIB_MULTIPROCESSING": "0",
        "SKLEARN_DATA_HOME": "/nonexistent",
    })
    return keep


def _validate(out, ref, n_classes):
    out = np.asarray(out, dtype=np.float64)
    if out.shape != ref.shape:
        return None, f"shape mismatch: {out.shape} vs {ref.shape}"
    if not np.all(np.isfinite(out)):
        return None, "non-finite calibrated probabilities"
    if out.ndim == 2:
        row_sums = out.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-3):
            return None, (f"rows do not sum to 1: min={row_sums.min():.4f}, "
                          f"max={row_sums.max():.4f}")
        if out.min() < -1e-9:
            return None, f"negative probability {out.min():.3e}"
        out = np.clip(out, 0.0, None)
    else:
        if out.min() < -1e-6 or out.max() > 1 + 1e-6:
            return None, f"probabilities outside [0,1]: [{out.min():.4f}, {out.max():.4f}]"
        out = np.clip(out, 0.0, 1.0)
    return out, "ok"


def _bwrap_prefix(sandbox):
    cmd = [BWRAP, "--die-with-parent", "--new-session",
           "--unshare-net", "--unshare-ipc", "--unshare-uts", "--unshare-pid",
           "--unshare-cgroup-try"]
    for pth in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/sys", "/opt"):
        if os.path.exists(pth):
            cmd += ["--ro-bind", pth, pth]
    cmd += ["--proc", "/proc", "--dev-bind", "/dev", "/dev",
            "--tmpfs", "/dev/shm", "--tmpfs", "/tmp", "--tmpfs", "/var/tmp", "--tmpfs", "/run"]
    cmd += ["--ro-bind", ENV_ROOT, ENV_ROOT]
    cmd += ["--bind", sandbox, "/work", "--chdir", "/work"]
    cmd += ["--setenv", "HOME", "/work", "--setenv", "TMPDIR", "/tmp"]
    return cmd


def run_candidate(code, timeout=180):
    import mls_calibration_score as S

    if MARKER not in code:
        return FAIL, {}, f"no `{MARKER}` in candidate", None
    low = code.lower()
    for p in LEAK_PATTERNS:
        if p in low:
            return FAIL, {}, f"LEAK: candidate references '{p}'", None

    sandbox = tempfile.mkdtemp(prefix="mlscal_")
    try:
        payload, perms, truth = {}, {}, {}
        rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
        for label, _, _ in SETTINGS:
            c = load_cache(label)
            test_perm = rng.permutation(len(c["test_labels"]))
            payload[f"cal_probs__{label}"] = c["cal_probs"]
            payload[f"cal_labels__{label}"] = c["cal_labels"]
            payload[f"test_probs__{label}"] = c["test_probs"][test_perm]
            perms[label] = test_perm
            truth[label] = (c["test_probs"], c["test_labels"], int(c["n_classes"]))
        np.savez(os.path.join(sandbox, "input.npz"),
                 settings=np.array([s[0] for s in SETTINGS]), **payload)
        with open(os.path.join(sandbox, "cand.py"), "w") as f:
            f.write(code)

        t0 = time.time()
        try:
            shutil.copyfile(os.path.abspath(__file__), os.path.join(sandbox, "shard.py"))
            cmd = _bwrap_prefix(sandbox) + [
                "--", sys.executable, "/work/shard.py", "--child", "--sandbox", "/work"]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, cwd=sandbox,
                env=_child_env(os.path.realpath(sandbox)))
        except subprocess.TimeoutExpired:
            return FAIL, {"wall_s": round(time.time() - t0, 1)}, f"timeout >{timeout}s", None
        wall = time.time() - t0

        out_p = os.path.join(sandbox, "out.npz")
        st_p = os.path.join(sandbox, "status.json")
        if not os.path.exists(out_p) or not os.path.exists(st_p):
            tail = (proc.stderr or proc.stdout or "")[-1500:]
            return FAIL, {"wall_s": round(wall, 1)}, f"child produced no output (rc={proc.returncode})", tail
        child_out = np.load(out_p, allow_pickle=False)
        status = json.load(open(st_p))
        if "__global__" in status:
            g = status["__global__"]
            return FAIL, {"wall_s": round(wall, 1)}, g.get("note", "candidate failed"), g.get("traceback")

        metrics, notes, tb = {"wall_s": round(wall, 1)}, [], None
        raw = {}
        for label, _, _ in SETTINGS:
            st = status.get(label, {"note": "missing"})
            metrics[f"elapsed_{label}"] = st.get("elapsed")
            if f"pred__{label}" not in child_out.files:
                notes.append(f"{label}: {st.get('note', 'missing')}")
                tb = tb or st.get("traceback")
                continue
            test_probs_orig, y_test, n_classes = truth[label]
            got = np.asarray(child_out[f"pred__{label}"], dtype=np.float64)
            if got.shape != test_probs_orig.shape:
                notes.append(f"{label}: shape mismatch {got.shape} vs {test_probs_orig.shape}")
                continue
            inv = np.empty_like(got)
            inv[perms[label]] = got
            ok, why = _validate(inv, test_probs_orig, n_classes)
            if ok is None:
                notes.append(f"{label}: {why}")
                continue
            raw[f"ECE_{label}"] = float(expected_calibration_error(ok, y_test))
            raw[f"Brier_{label}"] = float(compute_brier_score(ok, y_test, n_classes))
            raw[f"NLL_{label}"] = float(compute_nll(ok, y_test, n_classes))
            metrics[f"ECEadapt_{label}"] = round(adaptive_ece(ok, y_test), 6)
            metrics[f"ACC_{label}"] = round(top1_accuracy(ok, y_test), 6)

        for k, v in raw.items():
            metrics[k] = round(v, 6)
        score, per_setting, per_term = S.official_score(raw)
        for k, v in per_setting.items():
            metrics[f"set_{k}"] = round(v, 6)
        for k, v in per_term.items():
            metrics[f"term_{k}"] = round(v, 6)
        sl, per_setting_l, _ = S.local_score(raw)
        metrics["score_local"] = round(sl, 6)
        for k, v in per_setting_l.items():
            metrics[f"setlocal_{k}"] = round(v, 6)
        metrics["n_settings_scored"] = len(raw) // 3
        if len(raw) < 3 * len(SETTINGS):
            return FAIL, metrics, ("; ".join(notes) or "incomplete settings")[:600], tb
        return float(score), metrics, "ok", None
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def _atomic_write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--sandbox", help=argparse.SUPPRESS)
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--one", help="score a single candidate .py")
    ap.add_argument("--out", help="output json for --one")
    ap.add_argument("--timeout", type=int, default=180)
    a = ap.parse_args()

    if a.child:
        return child_main(a.sandbox)
    if a.build_cache:
        return build_cache(force=a.force)
    if not os.path.exists(BWRAP):
        sys.exit(f"[mlscal] bwrap not found at {BWRAP}. Refusing to evaluate candidates.")

    if a.one:
        r, m, note, tb = run_candidate(open(a.one).read(), timeout=a.timeout)
        _atomic_write(a.out, {"file": os.path.basename(a.one), "reward": r, "metrics": m,
                              "note": note, "traceback": tb})
        print(f"[one] {os.path.basename(a.one)} reward={r} | {note}", flush=True)
        return

    missing = [lab for lab, _, _ in SETTINGS if not os.path.exists(cache_path(lab))]
    if missing:
        sys.exit(f"[mlscal] no base-classifier cache for {missing} under {CACHE_DIR}. "
                 f"Run `{sys.executable} {os.path.abspath(__file__)} --build-cache` once.")

    SH, NSH = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
    cand_dir, res_dir = os.environ["CAND_DIR"], os.environ["RES_DIR"]
    os.makedirs(res_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(cand_dir) if f.endswith(".py"))
    mine = [f for i, f in enumerate(files) if i % NSH == SH]
    print(f"[mlscal {SH}] {len(mine)}/{len(files)} candidates", flush=True)
    for i, f in enumerate(mine):
        out = os.path.join(res_dir, f[:-3] + ".json")
        if os.path.exists(out):
            continue
        try:
            r, m, note, tb = run_candidate(open(os.path.join(cand_dir, f)).read(),
                                           timeout=a.timeout)
        except Exception as e:
            r, m, note, tb = FAIL, {}, f"HARNESS {type(e).__name__}: {str(e)[:300]}", None
        _atomic_write(out, {"file": f, "reward": r, "metrics": m, "note": note, "traceback": tb})
        print(f"[mlscal {SH}] {i + 1}/{len(mine)} {f} reward={r} | {note}", flush=True)


if __name__ == "__main__":
    main()
