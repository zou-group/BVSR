from __future__ import annotations

import argparse
import glob
import gzip
import json
import os

import numpy as np

from bvsr.cli import HelpFormatter
from bvsr.plotting import (LW_MAIN, N_KEY, N_LABEL, N_TICK, Page, aligned_ticks, key_row, nature_axes,
                           setup_nature, text_width)
from bvsr.stes.config import ARMS, PAPER, TASKS

K, C, ITERATIONS = PAPER["k"], PAPER["chains"], PAPER["iterations"]
XSCALE = {"plain": 1.0, "bvsr": 1 - PAPER["cut"]}
LINE_OF_ARM = {"plain": "simpletes", "bvsr": "bvsr"}
RELEASED_RUN = {"plain": "simpletes", "bvsr": "ours"}
FAIL = -1e5
ZAP_FAIL = -10.0
LEAK_KEY = "leak_match_frac_1e-2"
TITLES = {"svg": "Gene Expression Analysis", "pertpred": "Drug Response Prediction",
          "mls_anomaly": "Anomaly Detection", "sl_lrbsz": "Scaling Law Discovery",
          "zap_h16": "Brain Activity Prediction"}

W, H = 5.5, 1.70
XMAX = 1500
XTICKS = [0, 750, 1500]
SHORT = {"Gene Expression Analysis": "Gene analysis", "Drug Response Prediction": "Drug prediction",
         "Anomaly Detection": "Anomaly detection", "Scaling Law Discovery": "Scaling laws",
         "Brain Activity Prediction": "Brain activity"}
BVSR_COLOUR = "#7DC0A7"
SIMPLE_COLOUR = "#A6A6A6"
LINES = {"simpletes": ("SimpleTES", SIMPLE_COLOUR, LW_MAIN, 3),
         "bvsr": ("SimpleTES + BVSR", BVSR_COLOUR, 1.6, 4)}


def newest_nodes(run_dir: str) -> str | None:
    fs = glob.glob(os.path.join(run_dir, "*", "instance-*", "db_state_*", "nodes.json"))
    return max(fs, key=os.path.getmtime) if fs else None


def load_nodes(run_dir: str) -> list[dict]:
    f = newest_nodes(run_dir)
    if f is None:
        raise SystemExit(f"no SimpleTES checkpoint under {run_dir}")
    with open(f) as fh:
        d = json.load(fh)
    return d.get("nodes", d) if isinstance(d, dict) else d


def zap_failed(sc: float, task: str) -> bool:
    return task.startswith("zap") and sc <= ZAP_FAIL


def valid_score(node: dict, task: str) -> float | None:
    sc = node.get("score")
    if not isinstance(sc, (int, float)) or sc <= FAIL or zap_failed(sc, task):
        return None
    lk = (node.get("metrics") or {}).get(LEAK_KEY)
    if task.startswith("zap") and isinstance(lk, (int, float)) and lk >= 0.5:
        return None
    return float(sc)


def initial_score(nodes: list[dict], task: str) -> float | None:
    for n in nodes:
        if n.get("gen_id") is None and not n.get("parent_ids"):
            sc = n.get("score")
            return float(sc) if isinstance(sc, (int, float)) and sc > FAIL and not zap_failed(sc, task) else None
    return None


def rounds_curve(nodes: list[dict], task: str, arm: str) -> tuple[np.ndarray, np.ndarray]:
    best_round: dict[int, float] = {}
    for n in nodes:
        sc = valid_score(n, task)
        if n.get("gen_id") is None or sc is None:
            continue
        r = n["gen_id"] // C + 1
        best_round[r] = max(best_round.get(r, -np.inf), sc)
    R = min(max(best_round) if best_round else 0, ITERATIONS[arm])
    ys, cur = [], -np.inf
    for r in range(1, R + 1):
        cur = max(cur, best_round.get(r, -np.inf))
        ys.append(cur)
    return np.arange(1, R + 1) * K * C, np.array(ys)


def with_origin(x, y, init, xscale):
    if init is not None:
        x, y = np.concatenate([[0], x]), np.concatenate([[init], y])
    return (np.asarray(x, dtype=float) * xscale).astype(int), np.asarray(y, dtype=float)


def clip_at(x, y, xmax):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = x <= xmax
    xc, yc = x[keep], y[keep]
    if keep.sum() < len(x) and keep.any():
        xc, yc = np.append(xc, xmax), np.append(yc, np.interp(xmax, x, y))
    return xc, yc


def run_curve(task: str, arm: str, run_dir: str) -> tuple[np.ndarray, np.ndarray]:
    nodes = load_nodes(run_dir)
    x, y = with_origin(*rounds_curve(nodes, task, arm), initial_score(nodes, task), XSCALE[arm])
    finite = np.isfinite(y)
    return clip_at(x[finite], y[finite], XMAX)


def load_released(data_dir: str, run_ids: list[str]) -> dict[str, list[dict]]:
    path = os.path.join(data_dir, "nodes.jsonl.gz")
    nodes: dict[str, list[dict]] = {rid: [] for rid in run_ids}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            n = json.loads(line)
            if n["run"] in nodes:
                nodes[n["run"]].append({"gen_id": n["gen_id"], "score": n["score"]})
    missing = [rid for rid, ns in nodes.items() if not ns]
    if missing:
        raise SystemExit(f"runs not in {path}: {missing}")
    return nodes


def released_curve(nodes: list[dict], task: str, arm: str) -> tuple[np.ndarray, np.ndarray]:
    x, y = rounds_curve(nodes, task, arm)
    if len(x) == 0:
        return x, y
    return clip_at(*with_origin(x, y, initial_score(nodes, task), XSCALE[arm]), XMAX)


def paper_runs() -> list[tuple[str, str, str]]:
    return [(task, arm, f"{task}.{RELEASED_RUN[arm]}") for task in TASKS for arm in ARMS]


def compute_released(data_dir: str) -> list[dict]:
    runs = paper_runs()
    nodes = load_released(data_dir, [rid for _, _, rid in runs])
    return compute(runs, lambda task, arm, rid: released_curve(nodes[rid], task, arm))


def parse_run(spec: str) -> tuple[str, str, str]:
    try:
        task, arm, run_dir = spec.split(":", 2)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--run must be TASK:ARM:DIR, got {spec!r}") from None
    if task not in TASKS:
        raise argparse.ArgumentTypeError(f"unknown task {task!r} in {spec!r}; choose from {TASKS}")
    if arm not in ARMS:
        raise argparse.ArgumentTypeError(f"unknown arm {arm!r} in {spec!r}; choose from {ARMS}")
    return task, arm, run_dir


def compute(runs: list[tuple[str, str, str]], curve=run_curve) -> list[dict]:
    panels: dict[str, dict] = {}
    for task, arm, source in runs:
        panel = panels.setdefault(task, {"title": TITLES[task]})
        line = LINE_OF_ARM[arm]
        if line in panel:
            raise SystemExit(f"{task}: more than one run for the {LINES[line][0]!r} line "
                             f"(plain = SimpleTES; bvsr = SimpleTES + BVSR)")
        x, y = curve(task, arm, source)
        if len(x) == 0:
            raise SystemExit(f"{source}: no verified program with a valid score")
        panel[line] = {"x": x.tolist(), "y": y.tolist()}
        print(f"{task:12s} {arm:5s} -> {LINES[line][0]:17s} points={len(x)} best@{x[-1]:.0f}={y[-1]:.4f}",
              flush=True)
    return list(panels.values())


def draw(panels: list[dict]) -> Page:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter, FixedLocator
    pg = Page(W, H)
    n = len(panels)
    ranges = []
    for panel in panels:
        ys = np.r_[tuple(panel[line]["y"] for line in LINES if line in panel)]
        ranges.append((float(np.min(ys)), float(np.max(ys))))
    ticks_per_panel = aligned_ticks(ranges)
    tickpad = (plt.rcParams["ytick.major.size"] + plt.rcParams["ytick.major.pad"]) / 72
    lab_w = [max(text_width(pg.fig, t, fontsize=N_TICK) for t in labels) for _, labels, _ in ticks_per_panel]
    lead, gutter, right = 0.16, 0.13, 0.12
    col = max(lab_w) + tickpad
    first = lead + lab_w[0] + tickpad
    w = (W - first - right - (n - 1) * (col + gutter)) / n
    ax_b, ax_t = 0.30, H - 0.36
    spans = []
    for i, (panel, (ticks, labels, ylim)) in enumerate(zip(panels, ticks_per_panel)):
        x = first + i * (w + gutter + col)
        ax = pg.axes(x, ax_b, w, ax_t - ax_b)
        spans.append((x, x + w))
        nature_axes(ax, grid_y=True)
        for line, (_, colour, lw, z) in LINES.items():
            if line in panel:
                ax.plot(np.asarray(panel[line]["x"], dtype=float), np.asarray(panel[line]["y"], dtype=float),
                        color=colour, lw=lw, zorder=z, solid_capstyle="round", solid_joinstyle="round")
        ax.set_xlim(0, XMAX)
        ax.set_xticks(XTICKS)
        ax.set_xticklabels([str(t) for t in XTICKS])
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FixedFormatter(labels))
        x0, x1 = spans[-1]
        pg.text((x0 + x1) / 2, ax_t + 0.06, SHORT.get(panel["title"], panel["title"]), fontsize=N_LABEL,
                fontweight="bold", color="black", ha="center", va="baseline")
    x_mid = (spans[0][0] + spans[-1][1]) / 2
    pg.text(x_mid, 0.05, "Proposals verified", fontsize=N_LABEL, color="black", ha="center", va="baseline")
    pg.text(0.07, (ax_b + ax_t) / 2, "Best score", fontsize=N_LABEL, color="black", rotation=90,
            ha="center", va="center")
    key_row(pg, H - 0.09, [
        ("line", BVSR_COLOUR, LINES["bvsr"][2]), ("gap", 0.05), ("text", "SimpleTES + BVSR", dict(color="black")),
        ("gap", 0.28), ("line", SIMPLE_COLOUR, LW_MAIN), ("gap", 0.05), ("text", "SimpleTES", dict(color="black")),
    ], x=x_mid, align="center", fontsize=N_KEY)
    return pg


def render(panels: list[dict], out: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    family = setup_nature()
    pg = draw(panels)
    pg.save(out)
    plt.close(pg.fig)
    return family


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bvsr.stes.fig4", formatter_class=HelpFormatter,
                                 description="Best score vs verified proposals of SimpleTES runs (Fig. 4).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", default=None, metavar="DIR",
                     help="released bvsr_simpletes data directory (nodes.jsonl.gz): the paper's runs, "
                          "the SimpleTES and the SimpleTES + BVSR line of each task")
    src.add_argument("--run", action="append", type=parse_run, metavar="TASK:ARM:DIR",
                     help="one line: the run directory of an arm (plain or bvsr) on a task (repeat)")
    ap.add_argument("--out", required=True, help="figure path (.pdf or .png)")
    ap.add_argument("--dump", default=None, help="also write the plotted values (the panels) to this JSON")
    a = ap.parse_args(argv)
    panels = compute_released(a.data) if a.data else compute(a.run)
    if a.dump:
        with open(a.dump, "w") as f:
            json.dump(panels, f)
    family = render(panels, a.out)
    print(f"saved {a.out} ({W} x {H} in, font {family})")


if __name__ == "__main__":
    main()
