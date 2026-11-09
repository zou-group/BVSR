class CalibrationMethod(BaseEstimator):
    """No calibration — the identity map (the task's own default / null method)."""

    def __init__(self):
        self.is_binary = None

    def fit(self, probs, labels):
        self.is_binary = (probs.ndim == 1)
        return self

    def predict_proba(self, probs):
        if self.is_binary:
            return np.clip(probs, 0, 1)
        probs = np.clip(probs, 1e-15, 1.0)
        return probs / probs.sum(axis=1, keepdims=True)
