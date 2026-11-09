import os

from .base import Task

_PORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "svg_sparkx_ref.py")


class SVGTask(Task):
    name = "svg"
    marker = "def rank_svg"
    title = "SVG"

    def interface(self):
        return """\
You are designing the algorithm that RANKS features by how spatially organized their variation is.
Each of 2500 samples sits at a known 2-D location on a tissue section; each of 1050 features has a
noisy non-negative count per sample. Some features vary across space in a coherent, structured way
(gradients, patches, periodic patterns of any shape or length scale); others vary only through
sampling noise. Output one score per feature: higher = more confident its variation is spatially
structured. Only the ORDERING of the scores matters.

Write EXACTLY one function with this interface:
    def rank_svg(adata):
        # adata.layers['counts']     : scipy CSR raw int counts, shape [2500 samples x 1050 features]
        # adata.layers['normalized'] : log-normalized floats, same shape
        # adata.obsm['spatial']      : [2500 x 2] float coordinates of each sample
        # Return a length-1050 array of finite scores (one per feature, in adata.var order).

THE STRUCTURE YOU HAVE TO WORK WITH:
- Spatial signal comes in many shapes and scales at once: sharp patches, smooth gradients, periodic
  stripes; a detector tuned to one shape misses others. Robustness across pattern types is what
  separates the leaders from the rest.
- Counts are sparse and overdispersed; expression level and spatial structure are different axes —
  a highly expressed feature need not be spatially organized.
- The evaluation compares your ordering against ground truth WITHIN groups of related features that
  share a pattern but differ in signal strength — so getting the strength-ordering right inside a
  pattern family matters as much as separating signal from noise.

In scope (available as globals; helper functions are fine): `numpy as np`, `pandas as pd`,
`scipy.sparse as sp`, `sklearn`, `sc` (scanpy + its anndata). These are the ONLY installed
third-party packages — any other import fails at runtime.
Everything else is FIXED: the data, the held-out truth, the metric, and the anchoring against the
leaderboard's own random/perfect controls. Candidates that try to read the truth are voided.

CONSTRAINTS: the function name MUST be `rank_svg` and it MUST return one finite score per feature.
Keep one full run under ~10 CPU-minutes and under 8 GB. Deterministic (seed anything random)."""

    def baseline_pool(self):
        with open(_PORT, encoding="utf-8") as f:
            port = f.read()
        return f"""\
EXISTING BASELINE POOL. Score = the benchmark's OVERALL score on this dataset: rank correlation
against ground truth, scaled so a random ordering = 0 and a perfect ordering = 1. HIGHER is better.
Measured on this eval (official leaderboard values, same dataset):

[perfect-ordering control]  1.000   CEILING.
[spark_x]                   0.8531  STRONGEST method on the official leaderboard for this dataset —
                                    and its full implementation is given to you below. BEAT IT.
[gpcounts]                  0.787   Gaussian-process counts model (2nd place).
[spanve]                    0.766   Spatial-neighborhood variation statistic.
[moran_i]                   ~0.72   Classic spatial autocorrelation (Moran's I).
[random-ordering control]   0.000   FLOOR.

The state of the art's FULL implementation (our verbatim numpy port of SPARK-X, measured 0.8461 on
this exact eval — this is the method holding the #1 spot on the leaderboard), for you to improve on:
```python
{port}
```
Note where it stops: every kernel is an ISOTROPIC transform of each coordinate axis independently
(Gaussian bumps and cosines of fixed quantile length scales), the per-gene statistic only sees the
projection onto d=2 transformed coordinates, and the 11 kernels are combined by a generic p-value
average that ignores which patterns co-occur. Anisotropic patterns, interactions between axes,
multi-scale structure and pattern-family information are all invisible to it. The gap from 0.85 to
the 1.0 ceiling is exactly the orderings it gets wrong INSIDE pattern families — a mechanism with a
sharper notion of "how organized is this variation" can claim it."""

    def guidance(self):
        return """
3'. Commit the mechanism to the ACTUAL statistic — the borrowed idea must change WHAT is measured
   about spatial organization (which projections/scales/neighborhood structures are tested, how
   evidence across patterns is pooled, how signal strength is separated from noise level), not
   merely rename the given projection test.
- Only the ordering matters; monotone rescaling is a no-op. Ties destroy score: keep scores distinct.
- The given SPARK-X port is the leaderboard's #1 — matching it teaches nothing; the win condition is
   ordering the hard cases (same pattern, different strength) better than it does.
- Efficiency: under ~10 CPU-minutes / 8 GB, data is only 2500 x 1050. Seed randomness."""

    def baselines(self):
        return {"spark_x": 0.8531, "gpcounts": 0.787, "spanve": 0.766}
