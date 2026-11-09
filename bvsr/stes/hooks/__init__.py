from __future__ import annotations

import importlib
import inspect
import threading

ARM_HOOKS = {
    "plain": ["literature"],
    "bvsr": ["bv", "sr"],
}

PATCHED_ATTRS = {
    "literature": frozenset({("simpletes.generator", "Generator", "build_prompt")}),
    "bv": frozenset({("simpletes.generator", "Generator", "build_prompt"),
                     ("simpletes.engine.core", "SimpleTESEngine", "_generate_batch")}),
    "sr": frozenset({("simpletes.engine.runtime", "RuntimeBase", "handle_generation_output"),
                     ("simpletes.policies.base", "TrajectoryPolicyBase", "on_generation_failed")}),
}
assert not (PATCHED_ATTRS["literature"] | PATCHED_ATTRS["bv"]) & PATCHED_ATTRS["sr"]

ENGINE_ATTRS = ("_pending_nodes", "_db_lock", "_failure_records", "_counter_lock", "eval_queue",
                "selector", "_progress_event", "_stop_event", "checkpoint_dir")

_ORIGINALS: list[tuple[type, str, object]] = []


def patch(cls: type, name: str, fn) -> object:
    orig = vars(cls)[name]
    _ORIGINALS.append((cls, name, orig))
    setattr(cls, name, fn)
    return orig


def restore() -> None:
    while _ORIGINALS:
        cls, name, orig = _ORIGINALS.pop()
        setattr(cls, name, orig)


def check_simpletes_resolution() -> bool:
    from simpletes.engine.core import SimpleTESEngine
    from simpletes.engine.runtime import LocalRuntime, RuntimeBase
    from simpletes.generator import GenerationTask, Generator
    from simpletes.policies.balance import BalancePolicy
    from simpletes.policies.base import TrajectoryPolicyBase

    assert "build_prompt" in vars(Generator), "Generator must define build_prompt"
    assert "_generate_batch" in vars(SimpleTESEngine), "SimpleTESEngine must define _generate_batch"
    assert {"prompt", "gen_id", "chain_idx"} <= set(GenerationTask.__dataclass_fields__)
    assert "handle_generation_output" in vars(RuntimeBase), "RuntimeBase must define handle_generation_output"
    assert "handle_generation_output" not in vars(LocalRuntime), \
        "LocalRuntime overrides handle_generation_output; the SR gate on RuntimeBase would be bypassed"
    for name in ("register_batch", "on_child_done", "on_generation_failed"):
        assert name in vars(TrajectoryPolicyBase), f"TrajectoryPolicyBase must define {name}"
        assert name not in vars(BalancePolicy), \
            f"BalancePolicy overrides {name}; the SR gate on TrajectoryPolicyBase would be bypassed"
    assert callable(getattr(SimpleTESEngine, "_handle_pending_finalize", None)), \
        "SimpleTESEngine must provide _handle_pending_finalize"
    init_src = inspect.getsource(SimpleTESEngine.__init__)
    missing = [a for a in ENGINE_ATTRS if f"self.{a}" not in init_src]
    assert not missing, f"SimpleTESEngine.__init__ no longer sets {missing}"
    probe = BalancePolicy(num_chains=1, max_generations=1, k=1)
    assert type(probe._lock) is type(threading.Lock()), "selector._lock must be a threading.Lock"
    assert isinstance(probe._batches, dict), "selector._batches must be a dict"
    return True


def install(arm: str, cfg) -> list:
    if arm not in ARM_HOOKS:
        raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARM_HOOKS)}")
    installed = [importlib.import_module(f"{__name__}.{name}").install(cfg) for name in ARM_HOOKS[arm]]
    check_simpletes_resolution()
    return installed


__all__ = ["ARM_HOOKS", "ENGINE_ATTRS", "PATCHED_ATTRS", "check_simpletes_resolution", "install", "patch",
           "restore"]
