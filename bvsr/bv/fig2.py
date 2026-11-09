from __future__ import annotations

import argparse
import json

import numpy as np

from ..cli import HelpFormatter
from ..plotting import BV_DARK, BV_TINT, LW_HERO, N_KEY, N_LABEL, N_LETTER, Page, key_row, nature_axes, setup_nature
from ..tasks import FULL_CONTEXT
from .info_bits import read_bits
from .label_creativity import read_candidates
from .layout import BV_ARM

PANELS = [("c", "nf", "Top Creativity", "top"), ("u", "nf", "Top Utility", "top"),
          ("op", "abs", "Mean Originality", "mean"), ("nv", "abs", "Mean Non-obviousness", "mean")]

W, H = 5.5, 1.55
XTICKS = [0, 5, 10, 20, 100]
XLIM = (-3, 178)
DOT, LINE, BAND = "#7FA3D1", BV_DARK, BV_TINT
HEADERS = {
    "Top Creativity": ("a", "Top creativity"),
    "Top Utility": ("b", "Top utility"),
    "Mean Originality": ("c", "Originality"),
    "Mean Non-obviousness": ("d", "Non-obviousness"),
}


def cell_bits(bits: dict, bench: str, context) -> float | None:
    return bits.get((bench, FULL_CONTEXT if context is None else context))


def cells_and_thresholds(rows: list[dict], bits: dict):
    cells: dict[tuple, list[dict]] = {}
    for r in rows:
        cells.setdefault((r["bench"], r["arm"], r["context"], r["model"]), []).append(r)
    cells = {k: v for k, v in cells.items() if cell_bits(bits, k[0], k[2]) is not None}
    benches = sorted({r["bench"] for r in rows})
    thr = {m: {b: np.percentile([r[m] for r in rows if r["bench"] == b], 99) for b in benches}
           for m in ("c", "u", "op", "nv")}
    return cells, thr


def cell_y(met: str, kind: str, rs: list[dict], bench: str, thr: dict) -> float:
    if kind == "nf":
        return float(np.mean([r[met] > thr[met][bench] for r in rs]) / 0.01)
    return float(np.mean([r[met] for r in rs]))


def frontier(x, y, nbin: int, agg: str, rng):
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    cx, fy, lo, hi = [], [], [], []
    af = (lambda a: np.percentile(a, 95)) if agg == "top" else np.mean
    for bi in np.array_split(np.arange(len(x)), nbin):
        if len(bi) < 3:
            continue
        xx, yy = x[bi], y[bi]
        cx.append(np.median(xx))
        fy.append(af(yy))
        boot = [af(rng.choice(yy, len(yy), True)) for _ in range(1200)]
        lo.append(np.percentile(boot, 5))
        hi.append(np.percentile(boot, 95))
    return map(np.array, (cx, fy, lo, hi))


def jitter(x, rng):
    return x + rng.normal(0, np.where(x < 20, 1.3, 0.06 * x))


def compute(rows: list[dict], bits: dict) -> list[dict]:
    cells, thr = cells_and_thresholds(rows, bits)
    if not cells:
        raise SystemExit("no cell of the candidates has an information value in the bits file")
    baseline = [arm != BV_ARM for _, arm, _, _ in cells]
    rng = np.random.default_rng(0)
    panels = []
    for met, kind, name, agg in PANELS:
        x = np.array([cell_bits(bits, b, ctx) for b, _, ctx, _ in cells], dtype=float)
        y = np.array([cell_y(met, kind, rs, b, thr) for (b, _, _, _), rs in cells.items()], dtype=float)
        cx, fy, lo, hi = frontier(x, y, 5, agg, rng)
        xj = jitter(x, rng)
        panels.append({"ylabel": name, "kind": kind, "xj": xj.tolist(), "y": y.tolist(), "baseline": baseline,
                       "cx": cx.tolist(), "fy": fy.tolist(), "lo": lo.tolist(), "hi": hi.tolist()})
        print(f"{name:22s} dots={len(x)} frontier x=" + ",".join(f"{v:.1f}" for v in cx)
              + " y=" + ",".join(f"{v:.3f}" for v in fy), flush=True)
    print(f"{len(cells)} cells ({sum(baseline)} baseline), {sum(len(v) for v in cells.values())} proposals")
    return panels


def four_ticks(kind: str, ylo: float, yhi: float):
    if kind == "nf":
        s = max([s for s in (1, 2, 3, 4, 5, 10, 20) if 3 * s <= yhi] or [1])
        return [0, s, 2 * s, 3 * s], 0
    best = None
    for s in (0.05, 0.1, 0.2, 0.25, 0.3, 0.5):
        top = np.floor(yhi / s + 1e-9) * s
        ticks = [round(top - k * s, 10) for k in (3, 2, 1, 0)]
        if ticks[0] >= ylo - 1e-12:
            best = (ticks, 2 if s in (0.05, 0.25) else 1)
    if best is None:
        top = np.floor(yhi / 0.05 + 1e-9) * 0.05
        best = ([round(top - k * 0.05, 10) for k in (3, 2, 1, 0)], 2)
    return best


def draw(panels: list[dict]) -> Page:
    from matplotlib.ticker import FixedFormatter, FixedLocator
    pg = Page(W, H)
    pitch = W / len(panels)
    left_pad, right_pad = 0.385, 0.05
    ax_b, ax_t = 0.335, H - 0.27
    made = []
    for i, pan in enumerate(panels):
        px = i * pitch
        ax = pg.axes(px + left_pad, ax_b, pitch - left_pad - right_pad, ax_t - ax_b)
        nature_axes(ax, grid_y=True)
        xj, y = np.asarray(pan["xj"], dtype=float), np.asarray(pan["y"], dtype=float)
        cx, fy, lo, hi = (np.asarray(pan[k], dtype=float) for k in ("cx", "fy", "lo", "hi"))
        base = np.asarray(pan["baseline"], dtype=bool)
        extremes = [y.min(), y.max()]
        ax.fill_between(cx, lo, hi, color=BAND, alpha=0.95, zorder=2, lw=0)
        ax.plot(cx, fy, "-", color=LINE, lw=LW_HERO, zorder=4, solid_capstyle="round", solid_joinstyle="round")
        if len(cx):
            extremes += [lo.min(), hi.max()]
        ax.scatter(xj[base], y[base], s=3.2, color=DOT, alpha=0.70, edgecolor="white", linewidth=0.2, zorder=2.9)
        ax.scatter(xj[~base], y[~base], s=3.2, color=DOT, alpha=0.70, edgecolor="white", linewidth=0.2, zorder=3)
        ax.set_xscale("symlog", linthresh=20, linscale=1.1)
        ax.set_xticks(XTICKS)
        ax.set_xticklabels([str(t) for t in XTICKS])
        ax.minorticks_off()
        ax.set_xlim(*XLIM)
        dmin, dmax = min(extremes), max(extremes)
        plo, phi = (-0.45, dmax * 1.04) if pan["kind"] == "nf" else (dmin - 0.06 * (dmax - dmin),
                                                                        dmax + 0.06 * (dmax - dmin))
        ticks, dec = four_ticks(pan["kind"], plo, phi)
        letter, label = HEADERS[pan["ylabel"]]
        ax.set_ylabel(label, fontsize=N_LABEL, labelpad=2.0)
        pg.text(px + 0.03, ax_t + 0.045, letter, fontsize=N_LETTER, fontweight="bold", color="black",
                va="baseline")
        made.append((ax, pan["kind"], ticks, dec, dmin, dmax))
    pad = 0.04
    r_lo = max(0.06, max((m[2][0] - m[4]) / (m[2][-1] - m[2][0]) for m in made) + pad)
    r_hi = max(0.04, max((m[5] - m[2][-1]) / (m[2][-1] - m[2][0]) for m in made) + pad)
    for ax, kind, ticks, dec, _, _ in made:
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        fmt = [f"{v:.0f}×" for v in ticks] if kind == "nf" else [f"{v:.{dec}f}" for v in ticks]
        ax.yaxis.set_major_formatter(FixedFormatter(fmt))
    for ax, _, ticks, _, _, _ in made:
        span = ticks[-1] - ticks[0]
        ax.set_ylim(ticks[0] - r_lo * span, ticks[-1] + r_hi * span)
    x_mid = (left_pad + (W - right_pad)) / 2
    pg.text(x_mid, 0.05, "Task information given to the proposer (bits)", fontsize=N_LABEL, color="black",
            ha="center", va="baseline")
    key_row(pg, H - 0.085, [
        ("dot", dict(marker="o", ms=2.6, mfc=DOT, mec="white", mew=0.2)), ("gap", 0.05),
        ("text", "One batch of proposals", dict(color="black")), ("gap", 0.2),
        ("line", LINE, LW_HERO), ("gap", 0.05), ("text", "Trend line", dict(color="black")),
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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bvsr.bv.fig2",
                                 description="Fig. 2: creativity, utility, originality and non-obviousness "
                                             "against the task information.",
                                 formatter_class=HelpFormatter)
    ap.add_argument("--candidates", required=True,
                    help="candidates table: candidates.csv of bvsr.bv.label_creativity or the released "
                         "bv_results/candidates.csv.gz")
    ap.add_argument("--bits", required=True,
                    help="info_bits.csv of bvsr.bv.info_bits or the released bv_results/info_bits.csv")
    ap.add_argument("--out", required=True, help="figure path (.pdf or .png)")
    ap.add_argument("--dump", default=None, help="also write the plotted values (the panels) to this JSON")
    a = ap.parse_args(argv)
    panels = compute(read_candidates(a.candidates), read_bits(a.bits))
    if a.dump:
        with open(a.dump, "w") as f:
            json.dump(panels, f)
    family = render(panels, a.out)
    print(f"saved {a.out} ({W} x {H} in, font {family})")


if __name__ == "__main__":
    main()
