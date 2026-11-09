import argparse
import json
import os
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("LABELPROJ_DATA") or os.path.join(os.environ.get("BVSR_DATA", "data"), "labelproj")
RESULTS_JSON = os.path.join(DATA, "lp_results.json")
DATASET_ID = "cellxgene_census/gtex_v9"
MARKER = "def project"
ALL_METRICS = ["accuracy", "f1_macro", "f1_micro", "f1_weighted"]
LEAK_PATTERNS = ("solution.h5ad", "labelproj_data/solution", "label_pred_true")


def anchors():
    R = json.load(open(RESULTS_JSON))
    ctrl = {"true_labels", "majority_vote", "random_labels"}
    rows = [r for r in R if r.get("dataset_id") == DATASET_ID and r["method_id"] in ctrl]
    assert len(rows) == 3, f"expected 3 control rows, got {len(rows)}"
    out = {}
    for met in ALL_METRICS:
        vs = [r["metric_values"][met] for r in rows if isinstance(r["metric_values"][met], (int, float))]
        out[met] = (min(vs), max(vs))
    return out


def _load():
    import anndata as ad
    train = ad.read_h5ad(os.path.join(DATA, "train.h5ad"))
    test = ad.read_h5ad(os.path.join(DATA, "test.h5ad"))
    sol = ad.read_h5ad(os.path.join(DATA, "solution.h5ad"))
    assert "label" not in test.obs.columns, "labels leaked into test object"
    assert (test.obs_names == sol.obs_names).all()
    return train, test, sol


def official_metrics(pred, sol):
    from sklearn.metrics import f1_score
    import sklearn.preprocessing
    pred = pd.Categorical(np.asarray(pred, dtype=object))
    true = sol.obs["label"]
    cats = list(true.dtype.categories) + list(pred.dtype.categories)
    enc = sklearn.preprocessing.LabelEncoder().fit(cats)
    t = enc.transform(np.asarray(true))
    p = enc.transform(np.asarray(pred))
    out = {"accuracy": float(np.mean(t == p))}
    for avg in ("macro", "micro", "weighted"):
        out[f"f1_{avg}"] = float(f1_score(t, p, average=avg))
    return out


def score(raw, anch):
    scaled, clipped = {}, []
    for m in ALL_METRICS:
        v = raw.get(m)
        lo, hi = anch[m]
        s = (v - lo) / (hi - lo) if hi > lo else float("nan")
        scaled[m] = None if not np.isfinite(s) else float(s)
        clipped.append(0.0 if not np.isfinite(s) else float(np.clip(s, 0.0, 1.0)))
    return scaled, float(np.mean(clipped))


def run_candidate(code, train, test, sol, anch):
    low = code.lower()
    for p in LEAK_PATTERNS:
        if p in low:
            return None, {}, f"LEAK: candidate references '{p}'"
    if MARKER not in code:
        return None, {}, f"no `{MARKER}` in candidate"

    import scanpy as sc
    import scipy.sparse as sp
    import sklearn
    sys.path.insert(0, HERE)
    import compat
    compat.install()
    g = {"np": np, "pd": pd, "sp": sp, "sc": sc, "sklearn": sklearn, "__name__": "__cand__"}
    exec(compile(code, "<candidate>", "exec"), g)
    fn = g.get("project")
    if not callable(fn):
        return None, {}, "project not callable"

    test_work = compat.csr_inplace(test.copy())
    pred = fn(compat.csr_inplace(train.copy()), test_work)
    if pred is not None and hasattr(pred, "obs") and "label_pred" in getattr(pred, "obs").columns:
        pred = pred.obs["label_pred"].values
    elif pred is None and "label_pred" in test_work.obs.columns:
        pred = test_work.obs["label_pred"].values
    if pred is None:
        return None, {}, "returned None"
    pred = np.asarray(pred).ravel()
    if pred.shape[0] != test.n_obs:
        return None, {}, f"prediction length {pred.shape[0]} != {test.n_obs} test cells"

    raw = official_metrics(pred, sol)
    scaled, mean_score = score(raw, anch)
    metrics = {f"raw_{k}": round(v, 6) for k, v in raw.items()}
    metrics.update({f"scaled_{k}": (None if v is None else round(v, 6)) for k, v in scaled.items()})
    metrics["mean_score"] = round(mean_score, 6)
    metrics["n_pred_classes"] = int(len(set(pred.tolist())))
    return mean_score, metrics, "ok"


def _atomic_write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one")
    ap.add_argument("--out")
    ap.add_argument("--timeout", type=int, default=1500)
    a = ap.parse_args()

    anch = anchors()
    train, test, sol = _load()

    if a.one:
        tb = None
        try:
            reward, metrics, note = run_candidate(open(a.one).read(), train, test, sol, anch)
        except Exception as e:
            reward, metrics, note = None, {}, f"EXC {type(e).__name__}: {str(e)[:300]}"
            tb = "".join(l for l in traceback.format_exc().splitlines(keepends=True)
                         if os.path.basename(__file__) not in l)[-4000:]
            traceback.print_exc()
        _atomic_write(a.out, {"file": os.path.basename(a.one), "reward": reward,
                              "metrics": metrics, "note": note, "traceback": tb})
        print(f"[one] {os.path.basename(a.one)} reward={reward} | {note}", flush=True)
        return

    SH, NSH = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
    cand_dir, res_dir = os.environ["CAND_DIR"], os.environ["RES_DIR"]
    os.makedirs(res_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(cand_dir) if f.endswith(".py"))
    mine = [f for i, f in enumerate(files) if i % NSH == SH]
    print(f"[labelproj {SH}] {len(mine)}/{len(files)} candidates", flush=True)
    for i, f in enumerate(mine):
        out = os.path.join(res_dir, f[:-3] + ".json")
        if os.path.exists(out):
            continue
        try:
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "--one",
                                os.path.join(cand_dir, f), "--out", out],
                               capture_output=True, text=True, timeout=a.timeout)
            if os.path.exists(out):
                d = json.load(open(out))
                reward, note = d.get("reward"), d.get("note", "ok")
            else:
                reward, note = None, f"no result (rc={r.returncode})"
                _atomic_write(out, {"file": f, "reward": None, "metrics": {}, "note": note})
        except subprocess.TimeoutExpired:
            reward, note = None, f"timeout >{a.timeout}s"
            _atomic_write(out, {"file": f, "reward": None, "metrics": {}, "note": note})
        except Exception as e:
            reward, note = None, f"EXC {type(e).__name__}: {str(e)[:300]}"
            _atomic_write(out, {"file": f, "reward": None, "metrics": {}, "note": note})
        print(f"[labelproj {SH}] {i+1}/{len(mine)} {f} reward={reward} | {note}", flush=True)


if __name__ == "__main__":
    main()
