from __future__ import annotations

from dataclasses import dataclass

ARMS = ("plain", "bvsr")
TASKS = ("svg", "pertpred", "mls_anomaly", "sl_lrbsz", "zap_h16")
GPU_TASKS = ("zap_h16",)

PAPER = dict(
    model="together_ai/moonshotai/Kimi-K3",
    search_model="moonshotai/Kimi-K3",
    chains=4,
    k=40,
    iterations={"plain": 10, "bvsr": 20},
    temperature=0.7,
    max_tokens=32768,
    reasoning_effort="low",
    cut=0.5,
    seed=1,
    npapers=6,
)

_CPU = dict(eval_timeout=3000, eval_concurrency=10, gen_concurrency=4, timeout=300,
            log_interval={"plain": 320, "bvsr": 160}, env={})
_ZAP = dict(eval_timeout=7200, eval_concurrency=8, gen_concurrency=6, timeout=600,
            log_interval={"plain": 40, "bvsr": 20},
            env={"ZAP_TIME_BUDGET": "60", "ZAP_THREADS": "8", "ZAP_MEM_GB": "43.2"})

TASK_SETTINGS = {
    "svg": _CPU,
    "pertpred": _CPU,
    "mls_anomaly": _CPU,
    "sl_lrbsz": _CPU,
    "zap_h16": _ZAP,
}

BRIDGE_TIMEOUT = {"svg": 1200, "pertpred": 1800, "mls_anomaly": 1800, "sl_lrbsz": 1800, "zap_h16": 1200}


@dataclass
class RunConfig:
    arm: str
    task: str
    out: str
    search_model: str = PAPER["search_model"]
    reasoning_effort: str | None = PAPER["reasoning_effort"]
    seed: int = PAPER["seed"]
    npapers: int = PAPER["npapers"]
    cut: float = PAPER["cut"]
    sr_url: str | None = None
    sr_key: str | None = None
    sr_timeout_s: float = 3600.0
    sr_retry_s: float = 15.0

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; choose from {ARMS}")
        if self.task not in TASKS:
            raise ValueError(f"unknown task {self.task!r}; choose from {TASKS}")


def max_generations(iterations: int, k: int, chains: int) -> int:
    return iterations * k * chains


__all__ = ["ARMS", "BRIDGE_TIMEOUT", "GPU_TASKS", "PAPER", "RunConfig", "TASKS", "TASK_SETTINGS",
           "max_generations"]
