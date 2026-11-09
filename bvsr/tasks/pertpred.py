from .base import Task


class PertPredTask(Task):
    name = "pertpred"
    marker = "def predict_de"
    title = "PertPred"

    def interface(self):
        return """\
You are designing the algorithm that PREDICTS how a system responds to an intervention it has never
seen in that context. The data: for 614 (context, intervention) pairs — 6 contexts (cell types) x
~146 interventions (small-molecule compounds) with many cells each — you are given the measured
RESPONSE PROFILE: a signed, noise-calibrated differential-expression score per feature (~18k
features; positive = feature up, negative = down, magnitude = confidence). The held-out query is
255 (context, intervention) pairs — interventions observed in OTHER contexts but not this one.
Predict the full response profile for each query pair.

Write EXACTLY one function with this interface:
    def predict_de(de_train, id_map):
        # de_train.layers['clipped_sign_log10_pval'] : [614 x ~18k] float response profiles
        # de_train.obs['cell_type']  : the context of each training pair
        # de_train.obs['sm_name']    : the intervention of each training pair
        # de_train.obs['SMILES']     : a structural string representation of each intervention
        # de_train.var_names         : the ~18k feature names
        # id_map                     : DataFrame with columns ['id','cell_type','sm_name'] — the 255
        #                              query pairs, in output row order
        # Return a [len(id_map) x n_features] float array (finite), rows in id_map order.

THE STRUCTURE YOU HAVE TO WORK WITH:
- The (context x intervention) matrix of profiles is PARTIALLY OBSERVED; queries are held-out cells
  of that matrix. Both context main effects and intervention main effects are strong; the hard part
  is the interaction — how a given context reshapes a given intervention's response.
- Interventions have structure (the SMILES string): similar compounds tend to act similarly, but
  not always — and the strongest responses are the least transferable across contexts.
- Responses are signed and heavy-tailed; most features barely respond to most interventions. The
  evaluation is HALF error-based (row RMSE/MAE) and HALF shape-based (row Pearson/Spearman/cosine),
  so both magnitude calibration and profile shape matter.

In scope (available as globals; helper functions are fine): `numpy as np`, `pandas as pd`,
`scipy.sparse as sp`, `sklearn`, `sc` (scanpy + its anndata). These are the ONLY installed
third-party packages — any other import fails at runtime.
Everything else is FIXED: the data, the held-out truth, the metrics, the anchoring against the
leaderboard's own controls. Candidates that try to read the held-out data are voided.

CONSTRAINTS: function name MUST be `predict_de`, output [255 x n_features] finite floats in id_map
row order. Keep one full run under ~20 CPU-minutes and under 32 GB. Seed anything random."""

    def baseline_pool(self):
        return """\
EXISTING BASELINE POOL. Score = the benchmark's OVERALL score: 5 metrics (row RMSE + MAE, lower
better; row Pearson + Spearman + cosine, higher better), each scaled against the leaderboard's own
controls, clipped to [0,1] and averaged. HIGHER is better. Measured on this eval (official
leaderboard values):

[perfect control]                    1.000  CEILING (the held-out answers themselves).
[nn_retraining_with_pseudolabels]    0.465  BEST existing — the competition-winning neural-network
                                            ensemble with pseudo-label retraining. Note: it is a
                                            multi-model ensemble far exceeding this task's compute
                                            budget; its EDGE over the trivial baseline below is
                                            only +0.15 after a thousand-team competition.
[scape]                              0.446  2nd place (deep ensemble).
[pyboost]                            0.440  3rd place (gradient boosting ensemble).
[lgc_ensemble]                       0.431  4th place.
[jn_ap_op2]                          0.325  5th place.
[mean_across_celltypes control]      0.311  Predict each context's AVERAGE response for every query.
[transformer_ensemble]               0.269  6th place — a transformer ensemble that LOSES to the
                                            average.
[zeros control]                      0.134  Predict all zeros.

The trivial-but-strong baseline's implementation (mean_across_celltypes, 0.311), for you to improve on:
```python
def predict_de(de_train, id_map):
    import pandas as pd, numpy as np
    genes = list(de_train.var_names)
    df = pd.DataFrame(np.asarray(de_train.layers['clipped_sign_log10_pval']), columns=genes)
    df['cell_type'] = de_train.obs['cell_type'].values
    m = df.groupby('cell_type', observed=True)[genes].mean()
    return m.loc[id_map['cell_type']].values
```
Note what the pool tells you: after a thousand teams optimized against this exact score for months,
the best ensemble beats "predict the context average" by only +0.15, shape-aware transformers lose
to it, and the ceiling is 1.0. Whatever closes more of that gap is NOT sitting in any field's
standard toolbox — brute optimization has already been tried. The unclaimed 0.53 lives in the
context-by-intervention interaction structure that no existing method captures."""

    def guidance(self):
        return """
3'. Commit the mechanism to the ACTUAL interaction — the borrowed idea must change HOW the response
   of an unseen (context, intervention) cell is assembled from observed cells of the matrix (which
   structure is shared, which is context-specific, how intervention similarity is used, how
   magnitude is calibrated per context), not merely rename a mean or a ridge regression.
- The score is half shape, half magnitude: a prediction with the right profile shape but wrong
   scale, or right scale but flat shape, both cap well below the baseline.
- Efficiency: under ~20 CPU-minutes / 32 GB; the training matrix is only 614 x ~18k. Deep ensembles
   are outside the budget BY DESIGN — the win condition is a mechanism, not capacity.
- Must plausibly BEAT nn_retraining_with_pseudolabels (0.465), not just mean_across_celltypes
   (0.311)."""

    def baselines(self):
        return {"nn_retraining_with_pseudolabels": 0.465, "scape": 0.446, "pyboost": 0.440,
                "mean_across_celltypes": 0.311}
