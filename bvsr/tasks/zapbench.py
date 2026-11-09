from .base import Task, read_upstream

_REF = {
    16: {"mean_context": 0.028914, "persistence": 0.031578},
    32: {"mean_context": 0.033450, "persistence": 0.036258},
}


class _ZapTask(Task):
    horizon = 32
    marker = "def forecast"

    transplant = ("Implement a REAL transplant of THAT ONE mechanism into the forecaster -- not a "
                  "generic method wearing its name. ONE mechanism only; do NOT add other tricks.")
    code_format = ("<a SELF-CONTAINED module: your imports "
                   "followed by forecast() and any helpers>")

    def interface(self):
        H = self.horizon
        r = _REF[H]
        return (
            "Forecast cellular-resolution neural activity in a larval zebrafish brain (ZAPBench). "
            "Light-sheet fluorescence microscopy recorded per-neuron activity traces for 71,721 "
            "neurons over 7,879 time steps across 8 visual-stimulus training conditions. Given the "
            f"past 4 time steps of every neuron, forecast the next {H} time steps for every "
            "neuron jointly: 71,721 univariate series predicted together.\n\n"
            "You write ONE function (helpers/classes allowed), a self-contained module WITH its "
            "own imports (numpy, and optionally torch/jax/scipy/sklearn are available):\n\n"
            "    def forecast(X_train, predict_inputs, X_train_condition_lengths=None, **kwargs):\n"
            "        # X_train: (T_train, 71721) float32 -- 8 training conditions concatenated in\n"
            "        #     CONDITIONS_TRAIN order (0,1,2,4,5,6,7,8); train however you like.\n"
            "        # X_train_condition_lengths: (8,) int32 -- per-condition lengths of X_train,\n"
            "        #     same order; NEVER let a training window cross a condition boundary.\n"
            "        # predict_inputs: (n_windows, 4, 71721) float32 -- the 4-step context windows\n"
            "        #     you must forecast forward (stride-1 within each condition's val region).\n"
            "        # kwargs: random_state, time_budget_s, num_train_conditions=8, horizon,\n"
            "        #     context_len.\n"
            f"        return Y_pred      # (n_windows, {H}, 71721) float32, finite\n\n"
            "SCORING: combined_score = -MAE, where MAE = mean|Y_pred - Y_true| over windows, the "
            f"{H} horizon steps and all 71,721 neurons (lower MAE => higher score). A crash, "
            "timeout, wrong shape or any non-finite value scores the failure sentinel -10.0.\n\n"
            "BUDGET: forecast(...) must return within time_budget_s = 600 s (training "
            "included); 1 CUDA GPU (>=16 GB), 8 CPU threads, no network. Trivial references on "
            f"this split (H={H}): mean-context MAE={r['mean_context']:.4f}, "
            f"persistence MAE={r['persistence']:.4f}. Data are pre-normalized traces in roughly "
            "[-0.25, 1.5]."
        )

    def baseline_pool(self):
        init = read_upstream("SIMPLETES_HOME", "datasets", "zapbench", "whole_brain_forecasting",
                             "init_program.py")
        return (
            "EXISTING BASELINE (the official ZAPBench init program: a shared-weight per-neuron MLP "
            "with a learned global brain-state vector, condition-aware windowing, time-boxed "
            "training and a persistence fallback). Write your OWN self-contained forecast(); you "
            "may reuse ideas from it but keep imports inside your module:\n```python\n"
            + init + "\n```"
        )

    def baselines(self):
        r = _REF[self.horizon]
        return {"mean_context": -r["mean_context"], "persistence": -r["persistence"]}

    def guidance(self):
        H = self.horizon
        return (
            "\n3'. Transplant THAT ONE mechanism into the ACTUAL forecaster -- how cross-neuron "
            "structure, condition-specific dynamics, the global brain-state summary, or the "
            "per-neuron temporal model is built and fit -- not a generic method wearing its name.\n"
            "- The output is a SELF-CONTAINED module: your imports (numpy required; "
            "torch/jax/scipy/sklearn optional) followed by forecast() with the exact signature. "
            "Helper functions/classes at module level are fine; no filesystem or network I/O.\n"
            f"- Return EXACTLY (predict_inputs.shape[0], {H}, 71721) float32 and finite; build "
            "training windows strictly inside each condition (use X_train_condition_lengths); "
            "always keep a guaranteed-valid fallback (e.g. persistence) so a failed model still "
            "returns a valid array.\n"
            "- Respect time_budget_s (self-limit training to a fraction of it); seed randomness "
            "with random_state. Keep peak GPU memory under ~16 GB."
        )


class ZapH16Task(_ZapTask):
    name = "zap_h16"
    title = "ZAPBench-16"
    horizon = 16


class ZapH32Task(_ZapTask):
    name = "zap_h32"
    title = "ZAPBench-32"
    horizon = 32
