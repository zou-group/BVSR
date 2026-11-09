import glob
import json
import os
import subprocess
import sys

TIMEOUT = 1500
FAILURE = -1e6
RUNNER = r"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("sl_evaluator", sys.argv[1])
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
train = ev.evaluate(sys.argv[2])
test = ev.evaluate(sys.argv[2], use_test_data=True) if train.get("r2", -1e6) > -1e6 else {}
print("SLRESULT " + json.dumps({"train": train, "test": test}))
"""


def main():
    simpletes = os.environ.get("SIMPLETES_HOME")
    if not simpletes:
        sys.exit("SIMPLETES_HOME must point to the SimpleTES checkout (bash reproduce.sh --rerun creates it)")
    sltask = os.environ["SLTASK"]
    ev_path = os.path.join(simpletes, "datasets", "scaling_law", sltask, "evaluator.py")
    sh = int(os.environ.get("SHARD", "0"))
    ns = int(os.environ.get("NSHARD", "1"))
    cand, res = os.environ["CAND_DIR"], os.environ["RES_DIR"]
    os.makedirs(res, exist_ok=True)
    mine = sorted(glob.glob(os.path.join(cand, "*.py")))[sh::ns]
    print(f"[{sltask} {sh}/{ns}] {len(mine)} cands", flush=True)
    for i, f in enumerate(mine):
        stem = os.path.basename(f)[:-3]
        out = os.path.join(res, stem + ".json")
        if os.path.exists(out):
            continue
        metrics, reward = {}, None
        try:
            r = subprocess.run([sys.executable, "-c", RUNNER, ev_path, os.path.abspath(f)],
                               capture_output=True, text=True, timeout=TIMEOUT)
            line = next((l for l in r.stdout.splitlines() if l.startswith("SLRESULT ")), "")
            if line:
                d = json.loads(line[len("SLRESULT "):])
                tr, te = d.get("train", {}), d.get("test", {})
                reward = te.get("r2") if te.get("r2") is not None else FAILURE
                metrics = {"r2_train": tr.get("r2"), "nmse_train": tr.get("nmse"),
                           "nmae_train": tr.get("nmae"), "r2_test": te.get("r2"),
                           "nmse_test": te.get("nmse"), "error": tr.get("error")}
            else:
                metrics = {"error": (r.stderr or "no SLRESULT line")[-400:]}
                reward = FAILURE
        except subprocess.TimeoutExpired:
            metrics, reward = {"error": f"subprocess timeout {TIMEOUT}s"}, FAILURE
        except Exception as e:
            metrics, reward = {"error": str(e)[:400]}, FAILURE
        tmp = out + f".tmp{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump({"file": os.path.basename(f), "metrics": metrics, "reward": reward}, fh)
        os.replace(tmp, out)
        print(f"[{sltask} {sh}] {i+1}/{len(mine)} {stem} reward={reward}", flush=True)
    print(f"[{sltask} {sh}] DONE", flush=True)


if __name__ == "__main__":
    main()
