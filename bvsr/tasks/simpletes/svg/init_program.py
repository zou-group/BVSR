# EVOLVE-BLOCK-START
def rank_svg(adata):
    # SPARK-X (Zhu, Sun & Zhou, Genome Biology 2021), option='mixture' — a line-by-line numpy port
    # of the official R implementation (xzhoulab/SPARK R/sparkx.R), no-covariate path, which is
    # exactly what the OpenProblems spark_x component runs. Score = -log10(ACAT-combined p-value).
    #
    # Per kernel: center the (transformed) coordinates column-wise; the projection statistic for a
    # gene y over coords Z is  T = n * (y'Z (Z'Z)^-1 Z'y) / (y'y),  whose null distribution is
    # lambda_g * chi^2_d with lambda_g = 1 - n*ybar^2/(y'y) (the kernel eigenvalues Z'Z(Z'Z)^-1 = I_d
    # are exactly ones, so Davies' weighted mixture collapses to a scaled chi-square with d dof).
    # Mixture option tests 11 coordinate maps: raw + 5 Gaussian + 5 cosine bandwidths (quantiles of
    # |centered coord| at 0.2..1.0), combined per gene with the ACAT Cauchy combination.
    import numpy as np
    import scipy.sparse as sp
    from scipy.stats import chi2

    counts = adata.layers['counts']
    counts = counts.tocsr() if sp.issparse(counts) else sp.csr_matrix(counts)
    coords = np.asarray(adata.obsm['spatial'], dtype=np.float64)
    n_obs, n_gene = counts.shape

    # SPARK-X filters all-zero cells/genes; scores for dropped genes fall back to 0.
    cell_tot = np.asarray(counts.sum(axis=1)).ravel()
    keep_c = cell_tot != 0
    C = counts[keep_c].T.tocsr()                      # genes x cells, as in the R code
    coords_k = coords[keep_c]
    gene_tot = np.asarray(C.sum(axis=1)).ravel()
    keep_g = gene_tot != 0
    C = C[keep_g]
    n = C.shape[1]

    y_sq = np.asarray(C.multiply(C).sum(axis=1)).ravel()          # y'y per gene
    y_bar = np.asarray(C.mean(axis=1)).ravel()
    lam_g = 1.0 - n * y_bar ** 2 / y_sq                           # per-gene null scale

    def kernel_pvals(Z):
        Zc = Z - Z.mean(axis=0, keepdims=True)                    # column-center
        loc_inv = np.linalg.inv(Zc.T @ Zc)
        EHL = np.asarray(C @ Zc)                                  # genes x d
        stat = np.einsum('ij,jk,ik->i', EHL, loc_inv, EHL) * n / y_sq
        d = Zc.shape[1]
        return chi2.sf(stat / np.maximum(lam_g, 1e-300), df=d)

    def transloc(x, lker, kind):
        xc = x - x.mean()
        l = np.quantile(np.abs(xc), [0.2, 0.4, 0.6, 0.8, 1.0])[lker]
        if kind == "gaussian":
            return np.exp(-xc ** 2 / (2 * l ** 2))
        return np.cos(2 * np.pi * xc / l)

    def kernel_logsf(Z):
        # natural-log p-values: for chi^2_2, logsf(x) = -x/2 exactly — no underflow, ever
        Zc = Z - Z.mean(axis=0, keepdims=True)
        loc_inv = np.linalg.inv(Zc.T @ Zc)
        EHL = np.asarray(C @ Zc)
        stat = np.einsum('ij,jk,ik->i', EHL, loc_inv, EHL) * n / y_sq
        return -stat / (2.0 * np.maximum(lam_g, 1e-300))

    L = [kernel_logsf(coords_k)]                                  # projection kernel (raw coords)
    for kind in ("gaussian", "cosine"):
        for lker in range(5):
            Z = np.column_stack([transloc(coords_k[:, j], lker, kind)
                                 for j in range(coords_k.shape[1])])
            L.append(kernel_logsf(Z))
    L = np.minimum(np.column_stack(L), 0.0)                       # genes x 11, log p-values

    # ACAT (Cauchy) combination in LOG space. The reference R does: term = w*tan((0.5-p)*pi) for
    # p>=1e-16 and (w/p)/pi below; combined p = Cauchy tail of the sum. A naive float backtransform
    # (0.5 - arctan/pi) cancels to 0 for cct > ~1e16 and TIES all strong genes; R's pcauchy uses the
    # 1/(pi*q) asymptote. We reproduce that asymptote via logsumexp, keeping every gene distinct.
    from scipy.special import logsumexp
    W = 1.0 / L.shape[1]
    P = np.exp(L)
    small = L < np.log(1e-16)
    t_big = np.where(~small, W * np.tan((0.5 - np.minimum(P, 1.0)) * np.pi), 0.0).sum(axis=1)
    # small-p terms: sum_i (w/pi) * exp(-l_i)  -> log-sum-exp
    neg = np.where(small, -L, -np.inf)
    log_t_small = logsumexp(neg, axis=1) + np.log(W / np.pi)      # -inf where no small terms
    score_keep = np.empty(C.shape[0])
    for i in range(C.shape[0]):
        if np.isfinite(log_t_small[i]) and log_t_small[i] > 40:   # cct astronomically dominated
            log_cct = np.logaddexp(log_t_small[i], np.log(max(t_big[i], 1e-300)) if t_big[i] > 0
                                   else -np.inf)
            score_keep[i] = (np.log(np.pi) + log_cct) / np.log(10)   # -log10(1/(pi*cct))
        else:
            cct = t_big[i] + (np.exp(log_t_small[i]) if np.isfinite(log_t_small[i]) else 0.0)
            if cct > 1e15:
                score_keep[i] = (np.log(np.pi) + np.log(cct)) / np.log(10)
            else:
                comb = 0.5 - np.arctan(cct) / np.pi
                score_keep[i] = -np.log10(np.maximum(comb, 1e-300))

    score = np.zeros(n_gene)
    score[np.where(keep_g)[0]] = score_keep
    return score
# EVOLVE-BLOCK-END
