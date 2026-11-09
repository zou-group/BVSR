from __future__ import annotations

import argparse
import json
import warnings

import numpy as np

from ..cli import HelpFormatter
from ..plotting import LW_HERO, LW_MAIN, N_KEY, N_LABEL, N_LETTER, SR_DARK, Page, key_row, nature_axes, setup_nature

W, H = 5.5, 1.75
ZERO_SHOT = ["Pointwise", "Setwise", "Listwise", "Tournament"]
SR_LINE = SR_DARK
SFT_COLOUR = "#E3A35F"
ICL_COLOUR = "#6F6F6F"
JUDGE_COLOUR = "#BDBDBD"
RANDOM_COLOUR = "#7A7A7A"
STYLE = {
    "SR": (SR_LINE, LW_HERO, 9), "SFT": (SFT_COLOUR, LW_MAIN, 8), "ICL": (ICL_COLOUR, 0.8, 7),
    **{j: (JUDGE_COLOUR, 0.7, 5) for j in ZERO_SHOT}}
BAND_ALPHA = {"SR": 0.22, "SFT": 0.20}
TITLES = {"Top Creativity": "Creativity", "Top Utility": "Utility"}
YLABEL = "Top proposals kept"
XLABEL = "Candidates discarded"


def curves(res: dict) -> tuple[list[float], list[tuple[str, list[tuple]]]]:
    panels = []
    for p in res["panels"]:
        cs = []
        for lab, m in p["methods"].items():
            if lab not in STYLE:
                warnings.warn(f"{p['title']}: method {lab!r} is not part of Fig. 3 and is not drawn", stacklevel=2)
                continue
            cs.append((lab, m["mean"], m["lo"], m["hi"]))
        panels.append((p["title"], cs))
    return res["grid"], panels


def kept(v):
    return 1.0 - np.asarray(v, dtype=float)


def draw(grid, panels) -> Page:
    from matplotlib.ticker import FixedFormatter, FixedLocator
    pg = Page(W, H)
    grid_x = np.asarray(grid, dtype=float)
    pitch = W / len(panels)
    left_pad, right_pad = 0.50, 0.16
    ax_b, ax_t = 0.29, H - 0.27
    ticks = [0, 0.25, 0.5, 0.75, 1.0]
    labels = ["0%", "25%", "50%", "75%", "100%"]
    for i, (title, cs) in enumerate(panels):
        px = i * pitch
        ax = pg.axes(px + left_pad, ax_b, pitch - left_pad - right_pad, ax_t - ax_b)
        nature_axes(ax, grid_y=True)
        ax.plot([0, 1], kept([0, 1]), color=RANDOM_COLOUR, lw=0.7, ls=(0, (1, 1.6)), zorder=4)
        for lab, mean, lo, hi in cs:
            colour, lw, z = STYLE[lab]
            if lab in BAND_ALPHA:
                ax.fill_between(grid_x, kept(hi), kept(lo), color=colour, alpha=BAND_ALPHA[lab], lw=0,
                                zorder=z - 6)
            ax.plot(grid_x, kept(mean), color=colour, lw=lw, zorder=z, solid_capstyle="round",
                    solid_joinstyle="round")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.03)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_major_formatter(FixedFormatter(labels))
        ax.set_ylabel(YLABEL, fontsize=N_LABEL, labelpad=2.0)
        ax.set_xlabel(XLABEL, fontsize=N_LABEL, labelpad=2.0)
        pg.text(px + 0.03, ax_t + 0.045, "abcdefgh"[i], fontsize=N_LETTER, fontweight="bold", color="black",
                va="baseline")
        x0, x1 = px + left_pad, px + pitch - right_pad
        pg.text((x0 + x1) / 2, ax_t + 0.045, TITLES.get(title, title), fontsize=N_LABEL, fontweight="bold",
                color="black", ha="center", va="baseline")
    x_mid = (left_pad + (W - right_pad)) / 2
    start, end = key_row(pg, H - 0.085, [
        ("line", SR_LINE, LW_HERO), ("gap", 0.05), ("text", "Selective retention", dict(color="black")),
        ("gap", 0.13),
        ("line", SFT_COLOUR, LW_MAIN), ("gap", 0.05), ("text", "SFT regression", dict(color="black")),
        ("gap", 0.13),
        ("line", ICL_COLOUR, 0.9), ("gap", 0.05), ("text", "In-context learning", dict(color="black")),
        ("gap", 0.13),
        ("line", JUDGE_COLOUR, 1.0), ("gap", 0.05), ("text", "Zero-shot judges", dict(color="black")),
        ("gap", 0.13), ("line", RANDOM_COLOUR, 0.8, (0, (1, 1.6))), ("gap", 0.05),
        ("text", "Random", dict(color="black")),
    ], x=x_mid, align="center", fontsize=N_KEY)
    if start < 0.02 or end > W - 0.02:
        warnings.warn(f"the key row does not fit the page ({start:.2f} .. {end:.2f} in)", stacklevel=2)
    return pg


def render(grid, panels, out: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    family = setup_nature()
    pg = draw(grid, panels)
    pg.save(out)
    plt.close(pg.fig)
    return family


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bvsr.sr.fig3", formatter_class=HelpFormatter,
                                 description="Draw Fig. 3 from the output of bvsr.sr.evaluate.")
    ap.add_argument("--curves", required=True, help="JSON written by bvsr.sr.evaluate")
    ap.add_argument("--out", required=True, help="figure path (.pdf or .png)")
    a = ap.parse_args(argv)
    with open(a.curves) as f:
        grid, panels = curves(json.load(f))
    if not panels:
        raise SystemExit(f"{a.curves} has no panel")
    family = render(grid, panels, a.out)
    print(f"saved {a.out} ({W} x {H} in, font {family})")


if __name__ == "__main__":
    main()
