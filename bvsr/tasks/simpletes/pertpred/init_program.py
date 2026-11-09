# EVOLVE-BLOCK-START
# ============================================================
# Perturbation-response PREDICTOR (EDITABLE)
# Seed = mean_across_celltypes (0.311 on the official scale) — the trivial-but-strong control that a
# thousand-team NeurIPS-2023 competition beat by only +0.15 (best existing ensemble 0.465; ceiling
# 1.0; shape-aware transformers LOSE to this average). The unclaimed 0.53 lives in the
# context-by-intervention interaction structure: how a given context reshapes a given
# intervention's response. OpenEvolve should improve HOW an unseen (context, intervention)
# response is assembled from observed cells of the matrix, not merely rename a mean or a ridge.
#
# Contract: def predict_de(de_train, id_map) -> [len(id_map) x n_features] finite float array,
# rows in id_map order.
#   de_train.layers['clipped_sign_log10_pval'] : [614 x ~18k] float response profiles (signed,
#       noise-calibrated differential-expression scores; magnitude = confidence)
#   de_train.obs['cell_type'] : the context of each training pair (6 contexts)
#   de_train.obs['sm_name']   : the intervention of each training pair (~146 compounds)
#   de_train.obs['SMILES']    : a structural string representation of each intervention
#   de_train.var_names        : the ~18k feature names
#   id_map : DataFrame with columns ['id','cell_type','sm_name'] — the 255 query pairs, in output
#       row order (interventions observed in OTHER contexts but not this one)
# Score = 5 official metrics (row RMSE + MAE, lower better; row Pearson + Spearman + cosine, higher
# better), each control-scaled, clipped to [0,1] and averaged — half magnitude, half shape.
# In scope: numpy as np, pandas as pd, scipy.sparse as sp, sklearn, sc (scanpy). Budget ~20
# CPU-minutes / 32 GB; seed anything random.
# ============================================================


def predict_de(de_train, id_map):
    import pandas as pd, numpy as np
    genes = list(de_train.var_names)
    df = pd.DataFrame(np.asarray(de_train.layers['clipped_sign_log10_pval']), columns=genes)
    df['cell_type'] = de_train.obs['cell_type'].values
    m = df.groupby('cell_type', observed=True)[genes].mean()
    return m.loc[id_map['cell_type']].values
# EVOLVE-BLOCK-END
