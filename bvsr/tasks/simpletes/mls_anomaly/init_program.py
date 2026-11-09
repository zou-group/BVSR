# EVOLVE-BLOCK-START
# ECOD baseline (Li et al., TKDE 2022) via PyOD -- the published MLS-Bench reference detector.
# Contract: class CustomAnomalyDetector with __init__/fit(X)->self/decision_function(X)->(n,).
class CustomAnomalyDetector:
    """ECOD anomaly detector (PyOD default, matches ADBench)."""

    def __init__(self):
        from pyod.models.ecod import ECOD

        self.model = ECOD()

    def fit(self, X):
        self.model.fit(X)
        return self

    def decision_function(self, X):
        return self.model.decision_function(X)

# EVOLVE-BLOCK-END
