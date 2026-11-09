import math

SETTINGS = ["rf-mnist", "mlp-fashion_mnist", "svm-breast_cancer", "gbm-madelon"]
METRICS = ["ECE", "Brier", "NLL"]

LEADERBOARD = {
    "platt_scaling": {
        "ECE_rf-mnist": 0.025314, "Brier_rf-mnist": 0.07754, "NLL_rf-mnist": 0.17028,
        "ECE_mlp-fashion_mnist": 0.009051, "Brier_mlp-fashion_mnist": 0.182132,
        "NLL_mlp-fashion_mnist": 0.369057,
        "ECE_svm-breast_cancer": 0.059499, "Brier_svm-breast_cancer": 0.032844,
        "NLL_svm-breast_cancer": 0.123195,
        "ECE_gbm-madelon": 0.030624, "Brier_gbm-madelon": 0.152134, "NLL_gbm-madelon": 0.469615},
    "temperature_scaling": {
        "ECE_rf-mnist": 0.012242, "Brier_rf-mnist": 0.074941, "NLL_rf-mnist": 0.162236,
        "ECE_mlp-fashion_mnist": 0.009478, "Brier_mlp-fashion_mnist": 0.187667,
        "NLL_mlp-fashion_mnist": 0.382465,
        "ECE_svm-breast_cancer": 0.030495, "Brier_svm-breast_cancer": 0.025011,
        "NLL_svm-breast_cancer": 0.086363,
        "ECE_gbm-madelon": 0.030549, "Brier_gbm-madelon": 0.15199, "NLL_gbm-madelon": 0.469187},
    "isotonic_regression": {
        "ECE_rf-mnist": 0.014427, "Brier_rf-mnist": 0.078758, "NLL_rf-mnist": 0.223328,
        "ECE_mlp-fashion_mnist": 0.012344, "Brier_mlp-fashion_mnist": 0.182667,
        "NLL_mlp-fashion_mnist": 0.400997,
        "ECE_svm-breast_cancer": 0.018848, "Brier_svm-breast_cancer": 0.033275,
        "NLL_svm-breast_cancer": 0.095497,
        "ECE_gbm-madelon": 0.016128, "Brier_gbm-madelon": 0.151974, "NLL_gbm-madelon": 0.581668},
}

GMEAN_EPS = 0.01
GAMMA_MIN, GAMMA_MAX = 0.1, 10.0
REF_SCORE = 0.5
PATH_EDGE = 0.05


def _anchors(col, table=None):
    vals = [b[col] for b in (table or LEADERBOARD).values()]
    return max(vals), min(vals)


def _solve_gamma(floor, bound, ref, ref_score=REF_SCORE):
    if bound == floor:
        return 1.0
    r = (ref - floor) / (bound - floor) if bound > floor else (ref - bound) / (floor - bound)
    r = max(0.0, min(1.0, r))
    if r <= 0.0 or r >= 1.0:
        return 1.0
    rs = max(1e-9, min(1.0 - 1e-9, ref_score))
    return max(GAMMA_MIN, min(GAMMA_MAX, math.log(rs) / math.log(r)))


def _bounded_power(x, floor, bound, gamma):
    if math.isnan(x):
        return 0.0
    if bound == floor:
        return 1.0
    r = (x - floor) / (bound - floor) if bound > floor else (x - bound) / (floor - bound)
    return max(0.0, min(1.0, r)) ** gamma


def _solve_scale(floor, ref, ref_score=REF_SCORE):
    p = max(1e-9, min(1.0 - 1e-9, (ref_score + 1.0) / 2.0))
    z = math.log(p / (1.0 - p))
    return (ref - floor) / z if z else 1.0


def _sigmoid_score(y, floor, scale):
    if math.isnan(y) or y <= floor or scale <= 0:
        return 0.0
    z = (y - floor) / scale
    return 1.0 if z > 30 else 2.0 / (1.0 + math.exp(-z)) - 1.0


def term_score(col, raw, table=None):
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return 0.0
    floor_raw, ref_raw = _anchors(col, table)
    y, y_floor, y_ref, y_bound = -float(raw), -floor_raw, -ref_raw, -0.0
    r_ref = (y_ref - y_floor) / (y_bound - y_floor) if y_bound > y_floor else None
    if r_ref is not None and 0.0 < r_ref < 1.0 and (r_ref < PATH_EDGE or r_ref > 1.0 - PATH_EDGE):
        return _sigmoid_score(y, y_floor, _solve_scale(y_floor, y_ref))
    return _bounded_power(y, y_floor, y_bound, _solve_gamma(y_floor, y_bound, y_ref))


def official_score(metrics, table=None):
    per_setting, per_term = {}, {}
    for s in SETTINGS:
        ts = []
        for m in METRICS:
            col = f"{m}_{s}"
            sc = term_score(col, metrics.get(col), table)
            per_term[col] = sc
            ts.append(sc)
        per_setting[s] = sum(ts) / len(ts)
    vals = list(per_setting.values())
    if all(v <= 0.0 for v in vals):
        return 0.0, per_setting, per_term
    g = math.exp(sum(math.log(max(v, GMEAN_EPS)) for v in vals) / len(vals))
    return g, per_setting, per_term


BASELINE_SCORES = {k: official_score(v)[0] for k, v in LEADERBOARD.items()}

LOCAL_ANCHORS = {
    "platt_scaling": {
        "ECE_rf-mnist": 0.025356, "Brier_rf-mnist": 0.077551, "NLL_rf-mnist": 0.17031,
        "ECE_mlp-fashion_mnist": 0.010251, "Brier_mlp-fashion_mnist": 0.18594,
        "NLL_mlp-fashion_mnist": 0.37706,
        "ECE_svm-breast_cancer": 0.059503, "Brier_svm-breast_cancer": 0.032845,
        "NLL_svm-breast_cancer": 0.123199,
        "ECE_gbm-madelon": 0.031531, "Brier_gbm-madelon": 0.152074, "NLL_gbm-madelon": 0.46934},
    "temperature_scaling": {
        "ECE_rf-mnist": 0.012242, "Brier_rf-mnist": 0.074941, "NLL_rf-mnist": 0.162236,
        "ECE_mlp-fashion_mnist": 0.014493, "Brier_mlp-fashion_mnist": 0.194751,
        "NLL_mlp-fashion_mnist": 0.397007,
        "ECE_svm-breast_cancer": 0.030495, "Brier_svm-breast_cancer": 0.025011,
        "NLL_svm-breast_cancer": 0.086363,
        "ECE_gbm-madelon": 0.0317, "Brier_gbm-madelon": 0.151925, "NLL_gbm-madelon": 0.468895},
    "isotonic_regression": {
        "ECE_rf-mnist": 0.014427, "Brier_rf-mnist": 0.078758, "NLL_rf-mnist": 0.223328,
        "ECE_mlp-fashion_mnist": 0.014071, "Brier_mlp-fashion_mnist": 0.18768,
        "NLL_mlp-fashion_mnist": 0.456863,
        "ECE_svm-breast_cancer": 0.018848, "Brier_svm-breast_cancer": 0.033275,
        "NLL_svm-breast_cancer": 0.095497,
        "ECE_gbm-madelon": 0.021194, "Brier_gbm-madelon": 0.152213, "NLL_gbm-madelon": 0.582392},
}

IDENTITY_METRICS = {
    "ECE_rf-mnist": 0.216274, "Brier_rf-mnist": 0.160211, "NLL_rf-mnist": 0.398369,
    "ECE_mlp-fashion_mnist": 0.061744, "Brier_mlp-fashion_mnist": 0.204025,
    "NLL_mlp-fashion_mnist": 0.482212,
    "ECE_svm-breast_cancer": 0.038062, "Brier_svm-breast_cancer": 0.027545,
    "NLL_svm-breast_cancer": 0.098208,
    "ECE_gbm-madelon": 0.023293, "Brier_gbm-madelon": 0.151274, "NLL_gbm-madelon": 0.467422,
}


def local_score(metrics):
    return official_score(metrics, LOCAL_ANCHORS)


MEASURED = {
    "identity_no_calibration": official_score(IDENTITY_METRICS)[0],
    "platt_scaling(measured here)": official_score(LOCAL_ANCHORS["platt_scaling"])[0],
    "temperature_scaling(measured here)": official_score(LOCAL_ANCHORS["temperature_scaling"])[0],
    "isotonic_regression(measured here)": official_score(LOCAL_ANCHORS["isotonic_regression"])[0],
}
