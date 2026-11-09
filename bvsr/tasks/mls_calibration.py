import os

from .base import Task
from .mls_calibration_score import BASELINE_SCORES, IDENTITY_METRICS, MEASURED

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _asset(directory, name):
    with open(os.path.join(directory, name), encoding="utf-8") as f:
        return f.read()


class MLSCalibrationTask(Task):
    name = "mls_calibration"
    marker = "class CalibrationMethod"
    title = "MLS-Calibration"

    transplant = ("Implement a REAL transplant of THAT ONE mechanism into the calibration map — "
                  "not a generic method wearing its name. ONE mechanism only; do NOT add other "
                  "tricks.")
    code_format = ("<class CalibrationMethod(BaseEstimator) "
                   "and any module-level helpers; no import lines for numpy/scipy/sklearn top "
                   "level>")

    def interface(self):
        return (
            "OFFICIAL TASK (MLS-Bench `ml-calibration`, verbatim):\n"
            + _asset(_ASSETS, "mls_calibration_task_description.md").strip()
            + "\n\nTHE FIXED PIPELINE YOU PLUG INTO\n"
            "  For each classifier-dataset pair the data is split 60% train / 20% calibrate / 20% "
            "test (seed 42) and the classifier is trained on the train split; you are handed only "
            "its uncalibrated probabilities.\n"
            "  `fit` sees the CALIBRATION split's probabilities and labels; `predict_proba` is "
            "then applied to the TEST split's probabilities. You never see the raw features, the "
            "classifier, or the test labels.\n\n"
            + self._settings_table()
            + "\n\nSCORING (the official score_spec.py, reproduced exactly)\n"
            "  Each of the 12 numbers (3 metrics x 4 settings) is mapped to [0,1] against the "
            "published baselines: the WORST baseline for that column scores 0, the BEST scores "
            "0.5, and the theoretical bound 0.0 scores 1.\n"
            "  A setting's score is the mean of its three metric scores; the TASK score is the "
            "geometric mean of the four settings (each floored at 0.01). Higher is better.\n"
            "  Baselines on this scale — as published, and as re-measured on the exact base "
            "classifiers you are scored against (they differ because the anchors are extremely "
            "tight; the second column is the bar you actually have to clear):\n"
            + "".join(f"    {k:24s} published {v:.4f}   here "
                      f"{MEASURED.get(k + '(measured here)', float('nan')):.4f}\n"
                      for k, v in sorted(BASELINE_SCORES.items(), key=lambda kv: -kv[1]))
            + f"    {'identity (none at all)':24s} published   --     here "
              f"{MEASURED['identity_no_calibration']:.4f}\n"
            + "  A crash, a timeout, a wrong output shape, or an output that is not a valid "
              "probability distribution scores 0.0.\n"
        )

    def _settings_table(self):
        rows = [
            ("rf-mnist", "Random Forest (200 trees) on MNIST, 20000 subsample",
             "10-class, probs (n,10)", 4000, 4000),
            ("mlp-fashion_mnist", "MLP (256,128) on Fashion-MNIST, 20000 subsample",
             "10-class, probs (n,10)", 4000, 4000),
            ("svm-breast_cancer", "RBF SVM (C=10) on Breast Cancer",
             "binary, probs (n,) = P(class 1)", 114, 114),
            ("gbm-madelon", "Gradient Boosting (200 x depth 5) on Madelon",
             "binary, probs (n,) = P(class 1)", 520, 520),
        ]
        out = ["THE FOUR SETTINGS (all four are scored; the task score is their geometric mean)"]
        for lab, clf, shape, ncal, ntest in rows:
            extra = ("  |  uncalibrated: ECE={ECE:.4f} Brier={Brier:.4f} NLL={NLL:.4f}".format(
                ECE=IDENTITY_METRICS[f"ECE_{lab}"], Brier=IDENTITY_METRICS[f"Brier_{lab}"],
                NLL=IDENTITY_METRICS[f"NLL_{lab}"]))
            out.append(f"  {lab:20s} {clf}\n"
                       f"  {'':20s} {shape}; n_cal={ncal}, n_test={ntest}{extra}")
        return "\n".join(out)

    def baseline_pool(self):
        return ("EXISTING BASELINE (temperature scaling, Guo et al. ICML 2017 — the strongest of "
                f"the three published baselines at {BASELINE_SCORES['temperature_scaling']:.4f}):"
                "\n```python\n" + _asset(_ASSETS, "mls_calibration_baseline_temperature.py").strip()
                + "\n```")

    def baselines(self):
        d = dict(BASELINE_SCORES)
        d.update(MEASURED)
        return d

    def guidance(self):
        return (
            "\n3'. Transplant THAT ONE mechanism into the CALIBRATION MAP itself — how the "
            "confidence-to-probability transform is parameterised, fitted, regularised or "
            "extended to many classes — not a rename of an existing baseline.\n"
            "- Write ONLY `class CalibrationMethod(BaseEstimator)` with `fit(self, probs, labels)` "
            "(returns self) and `predict_proba(self, probs)`. Helper methods and module-level "
            "helper functions are fine; no `if __name__` block.\n"
            "- Already bound for you, do NOT re-import: `np` (numpy), `optimize`, `interpolate`, "
            "`special`, `stats`, `linalg` (scipy), `sklearn`, `BaseEstimator`, `math`. Further "
            "`from sklearn... import ...` inside a method is allowed.\n"
            "- The SAME class is used for all four settings. `probs` is (n,) for the two binary "
            "settings (positive-class probability) and (n, C) for the two 10-class settings; "
            "`predict_proba` must return the SAME shape, finite, in [0,1], and for the 2-D case "
            "rows summing to 1 (checked to atol 1e-3). Anything else scores 0.0.\n"
            "- `fit` may only use the calibration probabilities and labels it is given. The "
            "candidate runs in a sandboxed process with no network, no file access and no access "
            "to any dataset loader: reloading a dataset, re-deriving the split, or reading the "
            "test labels is impossible and scores 0.0.\n"
            "- The official ECE uses 15 fixed equal-width bins on the top-label confidence. An "
            "equal-MASS adaptive-bin ECE is recorded next to it as a cross-check, so a method "
            "whose only effect is to snap predictions onto those bins is visible as such — go for "
            "a genuinely better probability estimate.\n"
            "- NUMERICAL TRAP, worth knowing: the Fashion-MNIST MLP returns FLOAT32 probabilities "
            "containing exact 0.0 and exact 1.0, so `np.clip(p, 1e-12, 1 - 1e-12)` does not "
            "actually move 1.0 and a log-odds transform overflows to inf. Clip in float64 with an "
            "epsilon your dtype can represent, or work from log-probabilities.\n"
            "- Budget: all four settings together must finish in well under 3 minutes; seed any "
            "randomness. Degenerate maps (a constant, the class prior, discarding the input "
            "ordering) lose badly on Brier and NLL even when they look good on ECE.")
