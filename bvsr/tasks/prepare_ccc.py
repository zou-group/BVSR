import argparse
import os

import numpy as np
import pandas as pd


def _default_resource():
    try:
        import importlib.util
        spec = importlib.util.find_spec("liana")
        if spec and spec.submodule_search_locations:
            return os.path.join(list(spec.submodule_search_locations)[0], "resource", "omni_resource.csv")
    except ImportError:
        pass
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bvsr.tasks.prepare_ccc",
                                 description="Build the CCC task data (ccc_input.h5ad, ccc_truth.csv) from "
                                             "OpenProblems v1 resources.")
    ap.add_argument("--src", required=True, help="OpenProblems v1 tnbc_wu2021 log_cp10k dataset.h5ad")
    ap.add_argument("--resource", default=_default_resource(),
                    help="liana consensus resource CSV (default: from the installed liana package)")
    ap.add_argument("--out", default=os.environ.get("CCC_DATA")
                    or os.path.join(os.environ.get("BVSR_DATA", "data"), "ccc"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import anndata as ad
    import scipy.sparse as sp

    if not a.resource or not os.path.exists(a.resource):
        ap.error("ligand-receptor resource not found; pip install liana==1.8.1 or pass --resource")
    os.makedirs(a.out, exist_ok=True)

    print(f"loading {a.src} ...", flush=True)
    adata = ad.read_h5ad(a.src)
    print(f"  {adata.shape}  layers={list(adata.layers)}  obs={list(adata.obs.columns)[:6]}", flush=True)

    truth = pd.DataFrame(adata.uns["ccc_target"]).reset_index(drop=True)
    truth.columns = [str(c) for c in truth.columns]
    assert {"ligand", "target", "response"} <= set(truth.columns), truth.columns
    truth["response"] = truth["response"].astype(int)
    print(f"  truth: {truth.shape}  positives={int(truth.response.sum())} "
          f"({truth.response.mean():.3f})  ligands={truth.ligand.nunique()} targets={truth.target.nunique()}",
          flush=True)

    rng = np.random.default_rng(a.seed)
    ligs = np.array(sorted(truth.ligand.unique()))
    rng.shuffle(ligs)
    dev_ligs = set(ligs[: len(ligs) // 2])
    truth["fold"] = np.where(truth.ligand.isin(dev_ligs), "dev", "test")
    for f in ("dev", "test"):
        s = truth[truth.fold == f]
        print(f"  fold={f}: {len(s)} rows, {int(s.response.sum())} positive, {s.ligand.nunique()} ligands",
              flush=True)

    counts = adata.layers["counts"]
    if not sp.issparse(counts):
        counts = sp.csr_matrix(counts)
    out = ad.AnnData(X=counts.astype(np.float32).tocsr(), obs=adata.obs[["label"]].copy(),
                     var=pd.DataFrame(index=adata.var.index.astype(str)))
    out.layers["counts"] = out.X

    res = pd.read_csv(a.resource)
    res = res.rename(columns={"source_genesymbol": "ligand_genesymbol",
                              "target_genesymbol": "receptor_genesymbol"})
    res = res[["ligand_genesymbol", "receptor_genesymbol"]].drop_duplicates().reset_index(drop=True)
    out.uns["ligand_receptor_resource"] = res
    out.uns["var_names_all"] = np.asarray(adata.uns["var_names_all"], dtype=object) \
        if "var_names_all" in adata.uns else np.asarray(out.var.index, dtype=object)
    out.uns["target_organism"] = int(adata.uns.get("target_organism", 9606))
    out.uns["merge_keys"] = np.asarray(["ligand", "target"], dtype=object)
    out.uns["cell_types"] = np.asarray(sorted(out.obs["label"].astype(str).unique()), dtype=object)

    for k in list(out.uns):
        assert "target" not in k or k in ("target_organism", "cell_types"), k
    assert "ccc_target" not in out.uns and "response" not in out.obs.columns

    ip = os.path.join(a.out, "ccc_input.h5ad")
    tp = os.path.join(a.out, "ccc_truth.csv")
    out.write_h5ad(ip, compression="gzip")
    truth.to_csv(tp, index=False)
    print(f"\nwrote {ip}  ({os.path.getsize(ip) / 1e6:.0f} MB)  {out.shape}", flush=True)
    print(f"wrote {tp}  ({len(truth)} rows)", flush=True)
    print(f"cell types ({len(out.uns['cell_types'])}): {list(out.uns['cell_types'])[:6]} ...", flush=True)
    print(f"LR resource: {res.shape[0]} pairs, {res.ligand_genesymbol.nunique()} unique ligands", flush=True)
    cov = truth.ligand.isin(set(res.ligand_genesymbol)).mean()
    covg = truth.ligand.isin(set(out.var.index)).mean()
    print(f"truth ligands present in LR resource: {cov:.3f} | present as genes in the matrix: {covg:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
