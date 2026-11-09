import warnings


def _patch_onehot():
    try:
        import inspect
        from sklearn.preprocessing import OneHotEncoder
    except Exception:
        return
    params = inspect.signature(OneHotEncoder.__init__).parameters
    new = "sparse_output" in params
    old = "sparse" in params
    if new and old:
        return
    orig = OneHotEncoder.__init__

    def __init__(self, *a, **kw):
        if new and "sparse" in kw:
            kw["sparse_output"] = kw.pop("sparse")
        elif old and "sparse_output" in kw:
            kw["sparse"] = kw.pop("sparse_output")
        return orig(self, *a, **kw)

    OneHotEncoder.__init__ = __init__


def _patch_logreg():
    try:
        import inspect
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return
    if "multi_class" in inspect.signature(LogisticRegression.__init__).parameters:
        return
    orig = LogisticRegression.__init__

    def __init__(self, *a, **kw):
        kw.pop("multi_class", None)
        return orig(self, *a, **kw)

    LogisticRegression.__init__ = __init__


def _patch_iterative_solvers():
    try:
        import inspect
        import scipy.sparse.linalg as sla
    except Exception:
        return
    for name in ("cg", "cgs", "bicg", "bicgstab", "gmres", "lgmres", "minres", "qmr", "gcrotmk"):
        fn = getattr(sla, name, None)
        if fn is None:
            continue
        try:
            p = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            continue
        if "rtol" not in p:
            continue

        def wrap(f):
            def inner(*a, **kw):
                if "tol" in kw:
                    kw["rtol"] = kw.pop("tol")
                return f(*a, **kw)
            return inner

        setattr(sla, name, wrap(fn))


def _patch_pandas_nonzero():
    try:
        import numpy as np
        import pandas as pd
    except Exception:
        return
    for cls in (pd.Series, pd.Index):
        if not hasattr(cls, "nonzero"):
            cls.nonzero = lambda self: np.nonzero(np.asarray(self))


def _patch_pca_sparse():
    try:
        import scipy.sparse as sp
        from sklearn.decomposition import PCA
    except Exception:
        return
    if getattr(PCA, "_bvsr_sparse_ok", False):
        return

    def densify(X):
        if sp.issparse(X) and X.shape[0] * X.shape[1] * 8 < 2 << 30:
            return X.toarray()
        return X

    for meth in ("fit", "fit_transform", "transform"):
        orig = getattr(PCA, meth, None)
        if orig is None:
            continue

        def wrap(f):
            def inner(self, X, *a, **kw):
                return f(self, densify(X), *a, **kw)
            return inner

        setattr(PCA, meth, wrap(orig))
    PCA._bvsr_sparse_ok = True


def csr(X):
    try:
        import scipy.sparse as sp
    except Exception:
        return X
    if sp.issparse(X) and not isinstance(X, (sp.csr_matrix, sp.csc_matrix)):
        return X.tocsr()
    return X


def csr_inplace(adata):
    try:
        if getattr(adata, "X", None) is not None:
            adata.X = csr(adata.X)
    except Exception:
        pass
    for k in list(getattr(adata, "layers", {}) or {}):
        try:
            adata.layers[k] = csr(adata.layers[k])
        except Exception:
            pass
    for k in list(getattr(adata, "obsm", {}) or {}):
        try:
            adata.obsm[k] = csr(adata.obsm[k])
        except Exception:
            pass
    return adata


def install():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fn in (_patch_onehot, _patch_logreg, _patch_iterative_solvers,
                   _patch_pandas_nonzero, _patch_pca_sparse):
            try:
                fn()
            except Exception:
                pass
