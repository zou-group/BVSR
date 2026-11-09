import glob
import importlib.util
import json
import os
import shutil
import signal
import sys
import uuid

TIMEOUT = 1500


def main():
    home = os.environ.get("SIMPLETES_ASTRO_HOME")
    if not home:
        sys.exit("SIMPLETES_ASTRO_HOME must point to the SimpleTES checkout with the astrodynamics "
                 "tasks (bash reproduce.sh --rerun creates it)")
    mission = os.environ["ASTROTASK"]
    cand_dir = os.path.abspath(os.environ["CAND_DIR"])
    res_dir = os.path.abspath(os.environ["RES_DIR"])
    shard = int(os.environ.get("SHARD", "0"))
    nshard = int(os.environ.get("NSHARD", "1"))
    mdir = os.path.join(home, "datasets", "astrodynamics", mission)
    os.chdir(mdir)
    os.environ["SIMPLETES_ASTRODYNAMICS_INSTANCE"] = os.path.join(mdir, "instance.json")
    spec = importlib.util.spec_from_file_location("ev", os.path.join(mdir, "evaluator.py"))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    os.makedirs(res_dir, exist_ok=True)

    cands = sorted(glob.glob(os.path.join(cand_dir, "*.py")))
    for i, c in enumerate(cands):
        if i % nshard != shard:
            continue
        stem = os.path.basename(c)[:-3]
        out = os.path.join(res_dir, stem + ".json")
        if os.path.exists(out):
            continue
        tmp = os.path.join(mdir, f"_cand_{os.getpid()}_{uuid.uuid4().hex[:8]}_{stem[:40]}.py")
        rec = {"file": os.path.basename(c), "reward": 0.0, "metrics": {}}
        try:
            shutil.copyfile(c, tmp)

            def _alarm(signum, frame):
                raise TimeoutError("eval timeout")
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(TIMEOUT)
            r = ev.evaluate(tmp)
            signal.alarm(0)
            if isinstance(r, dict):
                rec["metrics"] = {k: v for k, v in r.items()
                                  if isinstance(v, (int, float, str, bool))}
                rec["reward"] = float(r.get("combined_score", 0.0) or 0.0)
        except Exception as e:
            rec["metrics"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        finally:
            signal.alarm(0)
            try:
                os.remove(tmp)
            except OSError:
                pass
        tmp_out = out + ".tmp"
        with open(tmp_out, "w") as fh:
            json.dump(rec, fh)
        os.replace(tmp_out, out)
        print(f"[{shard}/{nshard}] {stem}: {rec['reward']:.4f}", flush=True)


if __name__ == "__main__":
    main()
