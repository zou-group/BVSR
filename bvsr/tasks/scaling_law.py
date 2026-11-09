from .base import Task, read_upstream


def _init_program(subtask: str) -> str:
    return read_upstream("SIMPLETES_HOME", "datasets", "scaling_law", subtask, "init_program.py")


class _SLTask(Task):
    subtask = ""
    max_params = 0
    marker = "def fit_scaling_law"

    transplant = ("Implement a REAL transplant of THAT ONE mechanism into the functional form "
                  "and/or the fitting procedure — not a generic method wearing its name. ONE "
                  "mechanism only; do NOT add other tricks.")
    code_format = ("<a SELF-CONTAINED module: your "
                   "numpy/scipy imports followed by scaling_law_func and fit_scaling_law>")

    def guidance(self):
        return ("\n3'. Transplant THAT ONE mechanism into the FUNCTIONAL FORM and/or the FITTING "
                "procedure — the candidate must define scaling_law_func + fit_scaling_law with the "
                "exact signatures, self-contained (imports of numpy/scipy allowed, nothing else).\n"
                f"- HARD CAP: fit_scaling_law may return at most {self.max_params} scalar "
                "parameters (exceeding it scores -1e6).\n"
                "- scaling_law_func must be a pure function of (data_points, params): NO "
                "input-dependent statistics (median/min/max of the batch etc.).\n"
                "- Keep the whole fit under ~5 minutes; prefer multi-start local optimization for "
                "robustness. Predictions with NaN/Inf score -1e6.")


class SLLrBszTask(_SLTask):
    name = "sl_lrbsz"
    title = "SL-LR/BS"
    subtask = "lr_bsz_scaling_law"
    max_params = 26

    def interface(self):
        return ("Discover a scaling law function that models the relationship between learning "
        "rate, batch size, data size, model parameters and training loss.\n"
        "**IMPORTANT: the law must use no more than 26 fitted parameters.**\n\n"
        "DATA (2702 points, per-group fit): features [lr, bsz, data_size, "
        "non_embedding_param_size]; label lm_loss.\n"
        "  lr 2.44e-4..2.21e-2 (log-spaced); bsz 16..2048 (powers of 2); data 2e9..1e11 tokens; "
        "params 6e7..1.07e9; loss 2.1..3.7.\n\n"
        "You MUST write EXACTLY these functions (helpers allowed), self-contained file, "
        "imports limited to numpy/scipy:\n"
        "    def scaling_law_func(data_points, params): ...   # (N,4) array -> predicted loss (N,)\n"
        "    def fit_scaling_law(data_points, loss_values): ... # -> optimized params (<= 26 scalars)\n"
        "scaling_law_func must be pure in (data_points, params) — no input-dependent statistics.\n\n"
        "SCORING: R^2 = 1 - NMSE of the fitted law (per-group fit; any load/fit failure, NaN/Inf "
        "prediction, or parameter-cap breach scores -1e6). The SimpleTES init baseline (separable "
        "power law, BFGS from ones) numerically explodes on these feature scales: measured train "
        "R^2 = -3.9e22 — ANY sane fit beats it. The real bar is capturing the lr-bsz interaction "
        "(each group has an interior lr/bsz optimum with convex loss around it).")

    def baseline_pool(self):
        return ("EXISTING BASELINE (SimpleTES init_program — a separable power law fit by BFGS):\n"
                f"{_init_program(self.subtask)}\n")

    def baselines(self):
        return {"init_program": -7.0e24}


class SLUShapeTask(_SLTask):
    name = "sl_ushape"
    title = "SL-U"
    subtask = "easy_question_scaling_law"
    max_params = 6

    def interface(self):
        return ("Discover a scaling law function that models LLM performance on easy questions as "
        "a function of compute, which exhibits a characteristic U-shaped / double-descent "
        "pattern.\n"
        "**IMPORTANT: the law must use no more than 6 fitted parameters.**\n\n"
        "DATA (389 train points, per-group fit): feature [log_flops] (log10 of FLOPs in 1e21 "
        "units, range ~[-0.9, 2.9]); label brier_score (negative; more negative = better). "
        "Performance first WORSENS with scale, then improves.\n\n"
        "You MUST write EXACTLY these functions (helpers allowed), self-contained file, imports "
        "limited to numpy/scipy:\n"
        "    def scaling_law_func(data_points, params): ...   # (N,1) array -> predicted brier (N,)\n"
        "    def fit_scaling_law(data_points, loss_values): ... # -> optimized params (<= 6 scalars)\n"
        "scaling_law_func must be pure in (data_points, params) — no input-dependent statistics.\n\n"
        "SCORING: R^2 = 1 - NMSE of the fitted law (per-group fit; failures, NaN/Inf, or >6 "
        "params score -1e6). The SimpleTES init baseline (monotone separable power law) measures "
        "train R^2 = -124 — it cannot express a U shape at all. The 6-parameter cap is tight: "
        "the form must encode the non-monotonic U shape itself, not buy it with extra degrees "
        "of freedom.")

    def baseline_pool(self):
        return ("EXISTING BASELINE (SimpleTES init_program — a separable power law fit by BFGS; "
                "note it CANNOT express a U shape):\n"
                f"{_init_program(self.subtask)}\n")

    def baselines(self):
        return {"init_program": -466.8}
