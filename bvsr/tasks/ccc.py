from .base import Task


class CCCTask(Task):
    name = "ccc"
    marker = "def infer_ccc"
    title = "CCC"

    def interface(self):
        return """\
You are designing the algorithm that RANKS candidate interactions between senders and receivers in a
population of samples. Each sample belongs to one of ~29 known groups. There is a fixed catalogue of
signalling molecules ("ligands"), each with a known set of cognate partner molecules ("receptors").
For every (ligand, group) pair you must output a score: how strongly is that ligand's signalling
actually ACTIVE in that group. Higher score = more confident the interaction is real.

Write EXACTLY one function with this interface:
    def infer_ccc(adata):
        # adata.layers['counts'] : scipy CSR of int counts, shape [n_samples, n_features]
        #   (rows = samples/cells, cols = features/genes; column names in adata.var.index)
        # adata.obs['label']     : the group (cell type) each sample belongs to
        # adata.uns['ligand_receptor_resource'] : DataFrame with columns
        #     'ligand_genesymbol', 'receptor_genesymbol'  -- the catalogue of known ligand->receptor
        #     pairs. A name containing '_' (e.g. 'ITGA1_ITGB1') is a COMPLEX: every subunit must be
        #     present for the partner to be usable.
        # adata.uns['cell_types'] : the sorted list of group names
        # Set adata.uns['ccc_pred'] to a pandas DataFrame with EXACTLY the columns
        #     ['ligand', 'target', 'score']  -- one row per (ligand, group), 'score' real and finite,
        #     no duplicate (ligand, target) pairs -- then `return adata`.

THE STRUCTURE YOU HAVE TO WORK WITH:
- A ligand's signalling being active in a group is NOT the same as the ligand being abundant: what marks
  a real interaction is a coherent, group-SPECIFIC response, against a background where most features
  vary for unrelated reasons.
- The counts are sparse, non-negative, and vary hugely in scale between samples and between features.
- Only pairs listed in the catalogue are meaningful; a ligand typically has several partners, and they
  differ in how informative they are.
- Groups differ enormously in size (tens to thousands of samples), so per-group estimates differ wildly
  in reliability.

In scope (available as globals; helper functions are fine): `numpy as np`, `pandas as pd`,
`scipy.sparse as sp`, `sklearn`, `sc` (scanpy). You MAY also write your own imports inside the function.

Everything else is FIXED and you must not touch it: which data is loaded, the held-out truth, the
metrics, and the anchoring against the random-baseline floor and the perfect-answer ceiling. You have
NO access to the answer — do not attempt to read it from disk; candidates that reference it are voided.

CONSTRAINTS: the function name MUST be `infer_ccc` and it MUST set `adata.uns['ccc_pred']`. Score every
ligand in the catalogue against every group (~2500 x 29). Keep one full run under ~10 CPU-minutes and
under 16 GB (the matrix is ~42k x 28k; do not densify it)."""

    def baseline_pool(self):
        return """\
EXISTING BASELINE POOL. Score = the benchmark's OVERALL score: the mean of two metrics --- (i) average
precision over the held-out truth pairs, and (ii) an odds ratio comparing true-vs-false positives inside
the top 5% ranked pairs against the rest. Both are RANK-based, so rescaling your scores changes nothing:
the ONLY way up is a better ordering. Anchored so the random-prediction floor is 0.0 and the perfect
answer (the truth itself) is 1.0. HIGHER is better. Measured on this eval:

[true_events]              1.000  CEILING (control; uses the held-out answer — not attainable).
[product_specificity]      0.120  STRONGEST non-control — score a pair by how ENRICHED the ligand's best
                                  cognate partner is in that group (z-scored across groups), weighted by
                                  the ligand's overall abundance. BEAT THIS.
[specificity]              0.073  Partner enrichment in the group (z across groups), unweighted. This is
                                  the one that also holds up on a disjoint held-out set of ligands.
[magnitude]                0.022  Ligand abundance x strongest partner's raw level in the group. Nearly
                                  worthless: raw abundance alone barely beats chance.
[random_events]            0.000  FLOOR — random pairs, random scores.

The strongest baseline's reference implementation (product_specificity), for you to improve on:
```python
def infer_ccc(adata):
    expr  = celltype_mean_log1p_cp10k(adata)          # [n_groups x n_features]
    zexpr = (expr - expr.mean(0)) / expr.std(0)       # enrichment of each feature across groups
    rows = []
    for lig, recs in ligand_to_receptors(adata, expr).items():
        for ct in expr.index:
            # best cognate partner's enrichment in this group (complexes: MIN over subunits)
            s = max(min(zexpr.loc[ct, sub] for sub in r.split('_')) for r in recs)
            rows.append((lig, ct, s * expr[lig].mean()))
    adata.uns['ccc_pred'] = pd.DataFrame(rows, columns=['ligand', 'target', 'score'])
    return adata
```
Note where its score comes from and where it stops: it looks at ONE partner (the max), uses a single
group-level average, and treats every feature's z-score as equally trustworthy. It ignores how many
samples back each estimate, ignores everything downstream of the receptor, and has no notion of a
background/noise level. The gap from 0.12 to the 1.0 ceiling is enormous — this problem is nowhere near
solved, and a genuinely better notion of "this signal is real, not background" can move it a long way."""

    def guidance(self):
        return """
3'. Commit to the mechanism in the ACTUAL scorer — the borrowed idea must change HOW evidence for an
   interaction is accumulated, how the background/noise level is estimated, or how unreliable per-group
   estimates are shrunk, not merely rename a z-score. A generic enrichment score with a fancy docstring
   lands at ~0.07, not above the 0.12 baseline.
- The ranking is all that matters: both metrics are rank-based, so monotone rescaling of your scores is
   a no-op. Do not waste the mechanism on normalization; spend it on the ORDERING.
- Score the FULL grid (every catalogue ligand x every group). Pairs you leave unscored are pushed below
   everything else, and leaving most of the grid tied is measurably worse than ranking it.
- Efficient: one full run under ~10 CPU-minutes, under 16 GB, no densifying the ~42k x 28k matrix.
   numpy/pandas/scipy.sparse/sklearn/scanpy are in scope. The function name MUST be `infer_ccc` and it
   MUST set `adata.uns['ccc_pred']` with columns ['ligand','target','score'].
- Must plausibly BEAT product_specificity (0.120 on this eval), not just clear the floor (0.0). The
   ceiling is 1.0 and the best baseline is at 0.12, so there is a great deal of room."""

    def baselines(self):
        return {"product_specificity": 0.120, "specificity": 0.073, "magnitude": 0.022,
                "random_events": 0.0}
