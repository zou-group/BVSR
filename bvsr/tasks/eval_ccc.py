import argparse
import json
import os
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("CCC_DATA") or os.path.join(os.environ.get("BVSR_DATA", "data"), "ccc")
INPUT_H5AD = os.path.join(DATA, "ccc_input.h5ad")
TRUTH_CSV = os.path.join(DATA, "ccc_truth.csv")
MERGE_KEYS = ["ligand", "target"]
TOP_PROP = 0.05
MARKER = "def infer_ccc"

LEAK_PATTERNS = ("ccc_truth", "ccc_target", "assumed_truth", "ccc_testdata", "tnbc_op")


def _join_truth_and_pred(truth, pred):
    gt = truth.merge(pred, on=MERGE_KEYS, how="left")
    gt.loc[gt["response"].isna(), "response"] = 0
    if gt["score"].isna().all():
        gt["score"] = 0.0
    else:
        gt.loc[gt["score"].isna(), "score"] = np.nanmin(gt["score"]) - np.finfo(float).eps
    return gt


def _auprc(gt):
    from sklearn.metrics import auc, precision_recall_curve
    if gt["response"].nunique() < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(gt["response"], gt["score"], pos_label=1)
    return float(auc(recall, precision))


def _ap(gt):
    from sklearn.metrics import average_precision_score
    if gt["response"].nunique() < 2:
        return float("nan")
    return float(average_precision_score(gt["response"], gt["score"]))


def _odds_ratio(gt, top_prop=TOP_PROP):
    gt = gt.sort_values("score", ascending=False)
    top_n = int(gt.shape[0] * top_prop)
    a = np.zeros(len(gt))
    a[:top_n] = 1
    gt = gt.assign(top_n=a)
    top, bot = gt[gt["top_n"] == 1], gt[gt["top_n"] == 0]
    tp = float(np.sum(top.response == 1))
    fp = float(np.sum(top.response == 0))
    fn = float(np.sum(bot.response == 1))
    tn = float(np.sum(bot.response == 0))
    num, den = tp * tn, fp * fn
    if den == 0:
        if num == 0:
            return float("nan")
        oddsratio = np.inf
    else:
        oddsratio = num / den
    return float(1 - 1 / (1 + oddsratio / 2))


def _score_pred(truth, pred):
    out = {}
    for fold in ("full", "dev", "test"):
        t = truth if fold == "full" else truth[truth.fold == fold]
        gt = _join_truth_and_pred(t[MERGE_KEYS + ["response"]], pred)
        out[f"auprc_{fold}"] = _auprc(gt)
        out[f"ap_{fold}"] = _ap(gt)
        out[f"odds_ratio_{fold}"] = _odds_ratio(gt)
    return out


def _true_events(adata, truth):
    return truth[MERGE_KEYS + ["response"]].rename(columns={"response": "score"})


def _random_events(adata, truth, n_events=1000):
    rng = np.random.default_rng(seed=1)
    res = adata.uns["ligand_receptor_resource"]
    cts = np.asarray(adata.uns["cell_types"], dtype=object)
    df = pd.DataFrame({
        "ligand": rng.choice(np.asarray(res["ligand_genesymbol"], dtype=object), n_events),
        "target": rng.choice(cts, n_events),
        "score": rng.uniform(0, 1, n_events),
    })
    return df.loc[~df[MERGE_KEYS].duplicated()]


def _anchors(adata, truth):
    lo = _score_pred(truth, _random_events(adata, truth))
    hi = _score_pred(truth, _true_events(adata, truth))
    return lo, hi


def _scale(metrics, lo, hi):
    out = {}
    for k, v in metrics.items():
        d = hi[k] - lo[k]
        out[k] = float("nan") if (not np.isfinite(d) or abs(d) < 1e-12) else (v - lo[k]) / d
    return out


def _check_pred(pred, adata):
    if not isinstance(pred, pd.DataFrame):
        raise TypeError(f"uns['ccc_pred'] must be a DataFrame, got {type(pred).__name__}")
    for c in MERGE_KEYS + ["score"]:
        if c not in pred.columns:
            raise ValueError(f"uns['ccc_pred'] missing column '{c}' (has {list(pred.columns)})")
    if "response" in pred.columns:
        raise ValueError("uns['ccc_pred'] must not contain a 'response' column")
    if not np.all(np.isreal(pred["score"])):
        raise ValueError("uns['ccc_pred']['score'] must be real-valued")
    if len(pred) == 0:
        raise ValueError("uns['ccc_pred'] is empty")
    if pred.groupby(MERGE_KEYS).size().max() > 1:
        raise ValueError("uns['ccc_pred'] must hold exactly one row per (ligand, target)")
    if not np.isfinite(pd.to_numeric(pred["score"], errors="coerce")).all():
        raise ValueError("uns['ccc_pred']['score'] contains NaN/inf")
    return pred[MERGE_KEYS + ["score"]].copy()


def run_candidate(code, adata, truth, lo, hi):
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
    fn = g.get("infer_ccc")
    if not callable(fn):
        return None, {}, "infer_ccc not callable"

    work = compat.csr_inplace(adata.copy())
    ret = fn(work)
    obj = ret if (ret is not None and hasattr(ret, "uns")) else work
    if "ccc_pred" in obj.uns:
        raw_pred = obj.uns["ccc_pred"]
    elif isinstance(ret, (pd.DataFrame, dict)) or (ret is not None and not hasattr(ret, "uns")):
        raw_pred = ret
    else:
        return None, {}, "did not set uns['ccc_pred']"
    if raw_pred is None:
        return None, {}, "did not set uns['ccc_pred']"
    pred = _check_pred(pd.DataFrame(raw_pred), adata)

    raw = _score_pred(truth, pred)
    sc_ = _scale(raw, lo, hi)
    metrics = {f"raw_{k}": (None if not np.isfinite(v) else round(float(v), 6)) for k, v in raw.items()}
    metrics.update({f"scaled_{k}": (None if not np.isfinite(v) else round(float(v), 6))
                    for k, v in sc_.items()})
    ov = [sc_["ap_full"], sc_["odds_ratio_full"]]
    op = [sc_["auprc_full"], sc_["odds_ratio_full"]]
    if not all(np.isfinite(x) for x in ov):
        return None, metrics, "non-finite metric"
    reward = float(np.mean(ov))
    metrics["overall_full"] = round(reward, 6)
    metrics["overall_op_full"] = (round(float(np.mean(op)), 6)
                                  if all(np.isfinite(x) for x in op) else None)
    metrics["robust_full"] = round(float(np.min(ov)), 6)
    for f in ("dev", "test"):
        p = [sc_[f"ap_{f}"], sc_[f"odds_ratio_{f}"]]
        metrics[f"overall_{f}"] = round(float(np.mean(p)), 6) if all(np.isfinite(x) for x in p) else None
    metrics["n_pred"] = int(len(pred))
    return reward, metrics, "ok"


def _load():
    import anndata as ad
    adata = ad.read_h5ad(INPUT_H5AD)
    truth = pd.read_csv(TRUTH_CSV)
    assert "ccc_target" not in adata.uns, "truth leaked into the model-facing object"
    return adata, truth


def _atomic_write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", help="score a single candidate .py")
    ap.add_argument("--out", help="output json for --one")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()

    adata, truth = _load()
    lo, hi = _anchors(adata, truth)

    if a.one:
        tb = None
        try:
            reward, metrics, note = run_candidate(open(a.one).read(), adata, truth, lo, hi)
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
    print(f"[ccc {SH}] {len(mine)}/{len(files)} candidates", flush=True)
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
                reward, metrics, note = d.get("reward"), d.get("metrics", {}), d.get("note", "ok")
            else:
                reward, metrics, note = None, {}, f"no result (rc={r.returncode})"
        except subprocess.TimeoutExpired:
            reward, metrics, note = None, {}, f"timeout >{a.timeout}s"
        except Exception as e:
            reward, metrics, note = None, {}, f"EXC {type(e).__name__}: {str(e)[:300]}"
        _atomic_write(out, {"file": f, "reward": reward, "metrics": metrics, "note": note})
        print(f"[ccc {SH}] {i+1}/{len(mine)} {f} reward={reward} | {note}", flush=True)


if __name__ == "__main__":
    main()
