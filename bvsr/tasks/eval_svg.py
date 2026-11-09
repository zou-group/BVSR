import argparse
import json
import os
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("SVG_DATA") or os.path.join(os.environ.get("BVSR_DATA", "data"), "svg")
DATASET_ID = "zenodo_spatial/dbitseq/mouse_e10_eye"
MARKER = "def rank_svg"
LEAK_PATTERNS = ("solution.h5ad", "true_spatial", "svg_data/solution", "orig_feature")
PERM_SEED = 20260815


def anchors():
    R = json.load(open(os.path.join(DATA, "svg_results.json")))
    rows = {r["method_id"]: r["metric_values"]["correlation"] for r in R
            if r.get("dataset_id") == DATASET_ID and r["method_id"] in ("true_ranking", "random_ranking")}
    assert set(rows) == {"true_ranking", "random_ranking"}, rows
    return rows["random_ranking"], rows["true_ranking"]


def _load():
    import anndata as ad
    a = ad.read_h5ad(os.path.join(DATA, "dataset.h5ad"))
    s = ad.read_h5ad(os.path.join(DATA, "solution.h5ad"))
    assert "true_spatial_var_score" not in a.var.columns
    return a, s


def official_metric(score, a, s):
    df = pd.DataFrame({"feature_id": a.var["feature_id"].values,
                       "pred_spatial_var_score": np.asarray(score, dtype=float)})
    mg = pd.merge(df, s.var, how="left", on="feature_id")
    corr = mg.groupby("orig_feature_name", observed=True).apply(
        lambda x: x["pred_spatial_var_score"].corr(x["true_spatial_var_score"], method="kendall"))
    return float(corr.mean())


def run_candidate(code, a, s, lo, hi):
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
    a = compat.csr_inplace(a)
    g = {"np": np, "pd": pd, "sp": sp, "sc": sc, "sklearn": sklearn, "__name__": "__cand__"}
    exec(compile(code, "<candidate>", "exec"), g)
    fn = g.get("rank_svg")
    if not callable(fn):
        return None, {}, "rank_svg not callable"

    perm = np.random.default_rng(PERM_SEED).permutation(a.n_vars)

    def _call(ad_):
        sc_ = fn(ad_)
        if sc_ is None:
            return None
        sc_ = np.asarray(sc_, dtype=float).ravel()
        return sc_ if sc_.shape[0] == a.n_vars else None

    sp_score = _call(a[:, perm].copy())
    if sp_score is None:
        return None, {}, "returned None or wrong length (permuted pass)"
    score = np.empty_like(sp_score)
    score[perm] = sp_score
    if not np.all(np.isfinite(score)):
        return None, {}, "score contains NaN/inf"
    corr = official_metric(score, a, s)
    if not np.isfinite(corr):
        return None, {}, "degenerate: correlation undefined (all-tied scores in every group)"
    scaled = float(np.clip((corr - lo) / (hi - lo), 0.0, 1.0))
    metrics = {"raw_correlation": round(corr, 6), "mean_score": round(scaled, 6),
               "n_distinct": int(len(np.unique(np.round(score, 9))))}
    try:
        plain = _call(a.copy())
        if plain is not None and np.all(np.isfinite(plain)):
            c0 = official_metric(plain, a, s)
            if np.isfinite(c0):
                metrics["raw_correlation_unpermuted"] = round(c0, 6)
                metrics["order_gap"] = round(c0 - corr, 6)
    except Exception:
        pass
    return scaled, metrics, "ok"


def _atomic_write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one")
    ap.add_argument("--out")
    ap.add_argument("--timeout", type=int, default=900)
    a_ = ap.parse_args()

    lo, hi = anchors()
    a, s = _load()

    if a_.one:
        tb = None
        try:
            reward, metrics, note = run_candidate(open(a_.one).read(), a, s, lo, hi)
        except Exception as e:
            reward, metrics, note = None, {}, f"EXC {type(e).__name__}: {str(e)[:300]}"
            tb = "".join(l for l in traceback.format_exc().splitlines(keepends=True)
                         if os.path.basename(__file__) not in l)[-4000:]
            traceback.print_exc()
        _atomic_write(a_.out, {"file": os.path.basename(a_.one), "reward": reward,
                               "metrics": metrics, "note": note, "traceback": tb})
        print(f"[one] {os.path.basename(a_.one)} reward={reward} | {note}", flush=True)
        return

    SH, NSH = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
    cand_dir, res_dir = os.environ["CAND_DIR"], os.environ["RES_DIR"]
    os.makedirs(res_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(cand_dir) if f.endswith(".py"))
    mine = [f for i, f in enumerate(files) if i % NSH == SH]
    print(f"[svg {SH}] {len(mine)}/{len(files)} candidates", flush=True)
    for i, f in enumerate(mine):
        out = os.path.join(res_dir, f[:-3] + ".json")
        if os.path.exists(out):
            continue
        try:
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "--one",
                                os.path.join(cand_dir, f), "--out", out],
                               capture_output=True, text=True, timeout=a_.timeout)
            if os.path.exists(out):
                d = json.load(open(out))
                reward, note = d.get("reward"), d.get("note", "ok")
            else:
                reward, note = None, f"no result (rc={r.returncode})"
                _atomic_write(out, {"file": f, "reward": None, "metrics": {}, "note": note})
        except subprocess.TimeoutExpired:
            reward, note = None, f"timeout >{a_.timeout}s"
            _atomic_write(out, {"file": f, "reward": None, "metrics": {}, "note": note})
        except Exception as e:
            reward, note = None, f"EXC {type(e).__name__}: {str(e)[:300]}"
            _atomic_write(out, {"file": f, "reward": None, "metrics": {}, "note": note})
        print(f"[svg {SH}] {i+1}/{len(mine)} {f} reward={reward} | {note}", flush=True)


if __name__ == "__main__":
    main()
