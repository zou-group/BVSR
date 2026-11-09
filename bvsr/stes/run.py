from __future__ import annotations

import argparse
import json
import os
import runpy
import shlex
import sys

from bvsr.cli import HelpFormatter
from bvsr.stes.config import ARMS, GPU_TASKS, PAPER, TASKS, TASK_SETTINGS, RunConfig, max_generations

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(PACKAGE_DIR))
BENCHES_DIR = os.path.join(os.path.dirname(PACKAGE_DIR), "tasks", "simpletes")

UPSTREAM_INIT = {
    "sl_lrbsz": "datasets/scaling_law/lr_bsz_scaling_law/init_program.py",
    "zap_h16": "datasets/zapbench/whole_brain_forecasting/init_program.py",
}
EVOLVE_START, EVOLVE_END = b"# EVOLVE-BLOCK-START\n", b"# EVOLVE-BLOCK-END\n"

PATCH_MARKERS = {
    "simpletes/llm/litellm_client.py": '"k3" in self.model.lower()',
    "simpletes/engine/core.py": "STES_REFUND_PROVIDER_FAILURES",
}


def init_program_bytes(task: str, simpletes_home: str | None) -> bytes:
    if task not in UPSTREAM_INIT:
        with open(os.path.join(BENCHES_DIR, task, "init_program.py"), "rb") as fh:
            return fh.read()
    if not simpletes_home:
        raise SystemExit(f"task {task} takes its initial program from SimpleTES: set SIMPLETES_HOME")
    with open(os.path.join(simpletes_home, UPSTREAM_INIT[task]), "rb") as fh:
        text = fh.read()
    if task == "zap_h16":
        text = EVOLVE_START + text + EVOLVE_END
    return text


def bench_paths(task: str, out: str) -> dict[str, str]:
    d = os.path.join(out, "bench", task)
    return {"init_program": os.path.join(d, "init_program.py"),
            "evaluator": os.path.join(d, "evaluator.py"),
            "instruction": os.path.join(d, f"{task}.txt")}


def materialize_bench(task: str, out: str, simpletes_home: str | None) -> dict[str, str]:
    paths = bench_paths(task, out)
    os.makedirs(os.path.dirname(paths["evaluator"]), exist_ok=True)
    src = os.path.join(BENCHES_DIR, task)
    for key, name in (("evaluator", "evaluator.py"), ("instruction", f"{task}.txt")):
        with open(os.path.join(src, name), "rb") as fi, open(paths[key], "wb") as fo:
            fo.write(fi.read())
    with open(paths["init_program"], "wb") as fo:
        fo.write(init_program_bytes(task, simpletes_home))
    return paths


def simpletes_args(a, paths: dict[str, str]) -> list[str]:
    s = TASK_SETTINGS[a.task]
    args = ["--init-program", paths["init_program"],
            "--evaluator", paths["evaluator"],
            "--instruction", paths["instruction"],
            "--model", a.model,
            "--max-generations", str(max_generations(a.iterations, a.k, a.chains)),
            "--num-chains", str(a.chains), "--k-candidates", str(a.k),
            "--temperature", str(a.temperature), "--max-tokens", str(a.max_tokens),
            "--eval-timeout", str(s["eval_timeout"]), "--init-eval-repeats", "1", "--disable-reflection",
            "--log-interval", str(s["log_interval"][a.arm]),
            "--reasoning-effort", a.reasoning_effort,
            "--save-llm-io",
            "--eval-concurrency", str(s["eval_concurrency"]), "--gen-concurrency", str(s["gen_concurrency"])]
    if a.resume:
        args += ["--resume", os.path.abspath(a.resume)]
    args += ["--timeout", str(s["timeout"]), "--output-path", a.out, "--skip-preflight"]
    return args + list(a.extra)


def run_env(a) -> dict[str, str]:
    env = {"STES_REFUND_PROVIDER_FAILURES": "1",
           "TOGETHER_MIN_INTERVAL": os.environ.get("TOGETHER_MIN_INTERVAL") or "2"}
    key = os.environ.get("TOGETHER_API_KEY")
    if key:
        env["TOGETHER_API_KEY"] = env["TOGETHER_AI_API_KEY"] = key
    env.update(TASK_SETTINGS[a.task]["env"])
    if a.simpletes_home:
        env["SIMPLETES_HOME"] = a.simpletes_home
    pp = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and p != REPO_ROOT]
    env["PYTHONPATH"] = os.pathsep.join([REPO_ROOT] + pp)
    if a.task in GPU_TASKS:
        env["BVSR_GPU_LOCK_DIR"] = os.environ.get("BVSR_GPU_LOCK_DIR") or os.path.join(a.out, ".gpu_slots")
    return env


def check_patch(simpletes_home: str) -> None:
    for rel, marker in PATCH_MARKERS.items():
        with open(os.path.join(simpletes_home, rel), encoding="utf-8") as fh:
            if marker not in fh.read():
                raise SystemExit(f"{simpletes_home} is not patched ({rel} lacks {marker!r}); "
                                 f"bash reproduce.sh --rerun sets up a patched checkout")


def parse_args(argv):
    own, extra = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    ap = argparse.ArgumentParser(prog="python -m bvsr.stes.run", formatter_class=HelpFormatter,
                                 description="Run SimpleTES with the hooks of one arm (paper settings by default).",
                                 epilog="Arguments after '--' are passed to SimpleTES's main.py.")
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--simpletes_home", default=os.environ.get("SIMPLETES_HOME"),
                    help="patched SimpleTES checkout (default $SIMPLETES_HOME)")
    ap.add_argument("--model", default=PAPER["model"], help="SimpleTES generator (litellm id)")
    ap.add_argument("--search_model", default=PAPER["search_model"],
                    help="model driving the literature / BV searches (Together id)")
    ap.add_argument("--chains", type=int, default=PAPER["chains"])
    ap.add_argument("--k", type=int, default=PAPER["k"], help="candidates per step")
    ap.add_argument("--iterations", type=int, default=None,
                    help="steps per chain (default: plain %d, bvsr %d, so both arms verify the same number of "
                         "programs)" % (PAPER["iterations"]["plain"], PAPER["iterations"]["bvsr"]))
    ap.add_argument("--temperature", type=float, default=PAPER["temperature"])
    ap.add_argument("--max_tokens", type=int, default=PAPER["max_tokens"])
    ap.add_argument("--reasoning_effort", default=PAPER["reasoning_effort"])
    ap.add_argument("--cut", type=float, default=PAPER["cut"], help="bvsr: share of each step discarded")
    ap.add_argument("--seed", type=int, default=PAPER["seed"], help="seed of the BV field sequence")
    ap.add_argument("--npapers", type=int, default=PAPER["npapers"], help="papers per step (plain)")
    ap.add_argument("--sr_url", default=os.environ.get("BVSR_SR_URL"),
                    help="SR server URL (default $BVSR_SR_URL); its key is read from $BVSR_SR_KEY")
    ap.add_argument("--sr_retry_s", type=float, default=15.0,
                    help="pause between attempts while the SR server is unreachable")
    ap.add_argument("--sr_timeout_s", type=float, default=3600.0, help="read timeout of one scoring request")
    ap.add_argument("--sr_wait_min", type=float, default=120.0,
                    help="how long to wait for the SR server at start-up")
    ap.add_argument("--resume", default=None, help="SimpleTES checkpoint (db_state_*) to continue from")
    ap.add_argument("--print_only", action="store_true", help="print the command and environment, then exit")
    a = ap.parse_args(own)
    if a.iterations is None:
        a.iterations = PAPER["iterations"][a.arm]
    a.extra = extra
    a.out = os.path.abspath(a.out)
    if a.simpletes_home:
        a.simpletes_home = os.path.abspath(a.simpletes_home)
    return a


def main(argv=None):
    a = parse_args(sys.argv[1:] if argv is None else list(argv))
    home = a.simpletes_home
    if a.print_only:
        paths = bench_paths(a.task, a.out)
        main_py = os.path.join(home or "$SIMPLETES_HOME", "main.py")
        env = run_env(a)
        for k in ("TOGETHER_API_KEY", "TOGETHER_AI_API_KEY"):
            if k in env:
                env[k] = "<set>"
        print("cd " + (home or "$SIMPLETES_HOME"))
        print(" ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env.items())))
        print(shlex.join([sys.executable, main_py] + simpletes_args(a, paths)))
        print(f"# hooks: arm={a.arm}" + (f" sr_url={a.sr_url} cut={a.cut}" if a.arm == "bvsr" else ""))
        return 0

    if not home or not os.path.isfile(os.path.join(home, "main.py")):
        raise SystemExit("SimpleTES not found: set SIMPLETES_HOME to a patched checkout "
                         "(bash reproduce.sh --rerun creates third_party/SimpleTES)")
    check_patch(home)
    env = run_env(a)
    if "TOGETHER_API_KEY" not in env:
        raise SystemExit("TOGETHER_API_KEY is not set")
    cfg = RunConfig(arm=a.arm, task=a.task, out=a.out, search_model=a.search_model,
                    reasoning_effort=a.reasoning_effort, seed=a.seed, npapers=a.npapers, cut=a.cut,
                    sr_url=a.sr_url, sr_key=os.environ.get("BVSR_SR_KEY"), sr_timeout_s=a.sr_timeout_s,
                    sr_retry_s=a.sr_retry_s)
    if a.arm == "bvsr":
        from bvsr.stes.sr_client import SRClient
        h = SRClient(cfg.sr_url, cfg.sr_key, timeout=cfg.sr_timeout_s).wait_ready(a.sr_wait_min * 60)
        print(f"[run] SR server ready: {cfg.sr_url} ckpt={h.get('ckpt_fingerprint')}", flush=True)
    if a.task in GPU_TASKS and not os.environ.get("BVSR_GPUS"):
        print("[run] WARNING: BVSR_GPUS is not set; concurrent evaluations will share GPUs", flush=True)

    os.makedirs(a.out, exist_ok=True)
    paths = materialize_bench(a.task, a.out, home)
    stes_args = simpletes_args(a, paths)
    main_py = os.path.join(home, "main.py")
    with open(os.path.join(a.out, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump({"arm": a.arm, "task": a.task, "simpletes_args": stes_args,
                   "search_model": a.search_model, "seed": a.seed, "npapers": a.npapers, "cut": a.cut,
                   "env": {k: v for k, v in env.items() if "KEY" not in k}}, fh, indent=1)
    os.environ.update(env)

    sys.path.insert(0, home)
    os.chdir(home)
    from bvsr.stes import hooks
    installed = hooks.install(a.arm, cfg)
    print(f"[run] arm={a.arm} task={a.task} out={a.out}\n[run] main.py {' '.join(stes_args)}", flush=True)
    sys.argv = [main_py] + stes_args
    runpy.run_path(main_py, run_name="__main__")
    fatal = [h.fatal for h in installed if getattr(h, "fatal", None)]
    if fatal:
        print(f"[run] stopped before the end: {fatal[0]}; fix the cause and continue with --resume", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
