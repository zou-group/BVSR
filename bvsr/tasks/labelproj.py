from .base import Task


class LabelProjTask(Task):
    name = "labelproj"
    marker = "def project"
    title = "LabelProj"

    def interface(self):
        return """\
You are designing the algorithm that assigns a GROUP LABEL to each unlabeled query sample, given a
large labeled reference. Reference and query were collected in different batches of varying tissue
composition, so the query's feature distribution is SHIFTED relative to the reference — a
classifier that merely memorizes the reference distribution transfers imperfectly. Labels are
imbalanced (some groups are rare), and per-group performance matters as much as overall accuracy:
the evaluation averages accuracy AND macro/micro/weighted F1.

Write EXACTLY one function with this interface:
    def project(train, test):
        # train.layers['counts']     : scipy CSR raw int counts   [~200k samples x ~20k features]
        # train.layers['normalized'] : log-normalized floats, same shape
        # train.obs['label']         : the group label of each reference sample  (THE training signal)
        # train.obs['batch']         : which collection batch each sample came from
        # train.var['hvg'], train.var['hvg_score'] : highly-variable-feature flag / ranking
        # train.obsm['X_pca']        : [~200k x 50] PCA embedding (fit on reference+query jointly)
        # test.*                     : the SAME slots for the ~26k query samples, but obs has NO
        #                              'label' column — predicting it is the task.
        # Return a 1-D array/list of ~26k predicted labels (strings from train's label vocabulary),
        # one per test sample, in test.obs_names order.

THE STRUCTURE YOU HAVE TO WORK WITH:
- The reference is big and imbalanced; the query composition differs from the reference (some groups
  over-/under-represented). Batch-driven distribution shift between reference and query is the core
  difficulty — the ~0.23 headroom that no existing method claims lives there.
- X_pca is shared and cheap; the full count/normalized matrices are available if the mechanism needs
  feature-level information, but everything must fit the budget below.
- Macro-F1 weighs rare groups as much as common ones: a method that ignores rare groups caps its
  score well below one that handles them.

In scope (available as globals; helper functions are fine): `numpy as np`, `pandas as pd`,
`scipy.sparse as sp`, `sklearn`, `sc` (scanpy + its anndata). These are the ONLY installed
third-party packages — any other import fails at runtime. Implement the mechanism from these
primitives.
Everything else is FIXED: which data is loaded, the held-out query labels, the metrics, and the
anchoring against the leaderboard's own floor and ceiling controls. You have NO access to the query
labels — candidates that try to read them are voided.

CONSTRAINTS: the function name MUST be `project` and it MUST return one label per query sample.
Keep one full run under ~20 CPU-minutes and under 40 GB. Deterministic where possible (seed anything
random)."""

    def baseline_pool(self):
        return """\
EXISTING BASELINE POOL. Score = the benchmark's OVERALL score on this dataset: mean of 4 scaled
metrics (accuracy, F1 macro/micro/weighted), each scaled against the leaderboard's own controls
(random/majority floor -> 0, perfect labels -> 1). HIGHER is better. Measured on this eval (official
leaderboard values, same dataset):

[perfect-labels control]   1.000   CEILING (the true labels themselves).
[scanvi_scarches]          0.773   BEST existing — deep semi-supervised VAE with architecture
                                   surgery. Only 0.003 above plain logistic regression.
[logistic_regression]      0.770   Multinomial logistic regression on the shared 50-dim PCA.
[scanvi]                   0.769   Deep semi-supervised VAE.
[knn]                      0.735   k-nearest-neighbour vote in PCA space.
[seurat_transferdata]      0.732   Anchor-based transfer (CCA anchors + weighted vote).
[mlp]                      0.717   Multi-layer perceptron on PCA.
[xgboost]                  0.711   Gradient-boosted trees on PCA.
[naive_bayes]              0.663   Gaussian naive Bayes on PCA.
[majority_vote control]    ~0      FLOOR-ish: predict the commonest label everywhere.

The strongest simple baseline's OFFICIAL implementation (logistic_regression, 0.770), for you to
improve on:
```python
def project(train, test):
    import sklearn.linear_model
    classifier = sklearn.linear_model.LogisticRegression()
    classifier.fit(train.obsm["X_pca"], train.obs["label"].astype(str))
    return classifier.predict(test.obsm["X_pca"])
```
Note what the pool tells you: EVERY existing method — deep generative models, anchor transfer,
boosted trees, foundation models (all published below 0.74 here) — lands within 0.06 of this
8-line classifier, and the ceiling is 1.0. The 0.23 gap is where reference-to-query distribution
shift lives: labels whose appearance changes across batches, rare groups the reference barely
covers, and query regions the reference does not span. A mechanism with a genuinely better notion
of transferring under shift can claim territory no published method has."""

    def guidance(self):
        return """
3'. Commit the mechanism to the ACTUAL transfer — the borrowed idea must change HOW the reference
   distribution is mapped onto the shifted query (how uncertainty/shift is detected, how rare groups
   are protected, how query structure is exploited), not merely rename a regularized classifier.
   A generic classifier variant lands at ~0.77 with the rest of them.
- The evaluation is HALF macro-F1-like: rare labels count. Predicting only common labels caps you.
- Efficiency: under ~20 CPU-minutes / 40 GB. The full matrices are big — sparse ops or the PCA.
   Seed all randomness.
- Must plausibly BEAT scanvi_scarches (0.773 on this eval), not just clear naive_bayes (0.663)."""

    def baselines(self):
        return {"scanvi_scarches": 0.773, "logistic_regression": 0.770, "knn": 0.735,
                "mlp": 0.717, "naive_bayes": 0.663}
