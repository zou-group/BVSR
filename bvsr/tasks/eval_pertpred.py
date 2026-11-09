import argparse
import json
import os
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PERTPRED_DATA") or os.path.join(os.environ.get("BVSR_DATA", "data"), "pertpred")
LAYER = "clipped_sign_log10_pval"
MARKER = "def predict_de"
ALL_METRICS = ["mean_rowwise_cosine", "mean_rowwise_mae", "mean_rowwise_pearson",
               "mean_rowwise_rmse", "mean_rowwise_spearman"]
MAXIMIZE = {"mean_rowwise_cosine": True, "mean_rowwise_mae": False, "mean_rowwise_pearson": True,
            "mean_rowwise_rmse": False, "mean_rowwise_spearman": True}
LEAK_PATTERNS = ("de_test", "sc_test", "pertpred_data/de_test", "prediction.h5ad", "score.h5ad")


def anchors():
    R = [r for r in json.load(open(os.path.join(DATA, "pp_results.json"))) if r.get("dataset_id")]
    mi = json.load(open(os.path.join(DATA, "pp_method_info.json")))
    base = {m["method_id"] for m in mi if m.get("is_baseline")}
    out = {}
    for met in ALL_METRICS:
        vs = [r["metric_values"][met] for r in R if r["method_id"] in base
              and isinstance(r["metric_values"].get(met), (int, float))]
        out[met] = (min(vs), max(vs))
    return out


def _load():
    import anndata as ad
    tr = ad.read_h5ad(os.path.join(DATA, "de_train.h5ad"))
    te = ad.read_h5ad(os.path.join(DATA, "de_test.h5ad"))
    idm = pd.read_csv(os.path.join(DATA, "id_map.csv"))
    return tr, te, idm


def official_metrics(P, te):
    Y = te.layers[LAYER]
    Y = np.asarray(Y.todense()) if hasattr(Y, "todense") else np.asarray(Y)
    P = np.asarray(P, dtype=np.float64)
    P = np.where(np.isnan(P), 0.0, P)
    out = {}
    out["mean_rowwise_rmse"] = float(np.mean(np.sqrt(np.mean((Y - P) ** 2, axis=1))))
    out["mean_rowwise_mae"] = float(np.mean(np.mean(np.abs(Y - P), axis=1)))
    pear, spear, cosi = [], [], []
    from scipy.stats import spearmanr
    for i in range(Y.shape[0]):
        y, p = Y[i], P[i]
        sy, sp_ = y.std(), p.std()
        pear.append(np.corrcoef(y, p)[0, 1] if sy > 0 and sp_ > 0 else 0.0)
        s = spearmanr(y, p).statistic if sp_ > 0 else 0.0
        spear.append(0.0 if not np.isfinite(s) else s)
        den = np.linalg.norm(y) * np.linalg.norm(p)
        cosi.append(float(y @ p / den) if den > 0 else 0.0)
    out["mean_rowwise_pearson"] = float(np.mean(pear))
    out["mean_rowwise_spearman"] = float(np.mean(spear))
    out["mean_rowwise_cosine"] = float(np.mean(cosi))
    return out


def score(raw, anch):
    scaled, clipped = {}, []
    for m in ALL_METRICS:
        lo, hi = anch[m]
        v = raw[m]
        s = ((v - lo) / (hi - lo)) if MAXIMIZE[m] else ((hi - v) / (hi - lo))
        s = s if np.isfinite(s) else 0.0
        scaled[m] = float(s)
        clipped.append(float(np.clip(s, 0.0, 1.0)))
    return scaled, float(np.mean(clipped))


def run_candidate(code, tr, te, idm, anch):
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
    fn = g.get("predict_de")
    if not callable(fn):
        return None, {}, "predict_de not callable"
    P = fn(compat.csr_inplace(tr.copy()), idm.copy())
    if P is None:
        return None, {}, "returned None"
    P = np.asarray(P, dtype=float)
    if P.shape != (len(idm), tr.n_vars):
        return None, {}, f"prediction shape {P.shape} != ({len(idm)}, {tr.n_vars})"
    if not np.all(np.isfinite(P)):
        return None, {}, "prediction contains NaN/inf"
    raw = official_metrics(P, te)
    scaled, mean_score = score(raw, anch)
    metrics = {f"raw_{k}": round(v, 6) for k, v in raw.items()}
    metrics.update({f"scaled_{k}": round(v, 6) for k, v in scaled.items()})
    metrics["mean_score"] = round(mean_score, 6)
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
    tr, te, idm = _load()
    if a.one:
        tb = None
        try:
            reward, metrics, note = run_candidate(open(a.one).read(), tr, te, idm, anch)
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
    print(f"[pertpred {SH}] {len(mine)}/{len(files)} candidates", flush=True)
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
        print(f"[pertpred {SH}] {i+1}/{len(mine)} {f} reward={reward} | {note}", flush=True)


if __name__ == "__main__":
    main()
