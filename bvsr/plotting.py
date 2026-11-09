from __future__ import annotations

import os

FONT = "Liberation Sans"
FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "fonts")
FONT_FILES = ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf")
FALLBACK_FONT = "DejaVu Sans"
LAYOUT_DPI = 200
SAVE_DPI = 300

N_LETTER, N_LABEL, N_TICK, N_KEY = 9.0, 7.5, 7.0, 7.0
N_AXIS_LW = 0.5
LW_HERO, LW_MAIN = 1.35, 1.0

BV_DARK, BV_TINT = "#1C4C8C", "#DCE8F7"
SR_DARK = "#984C08"


def register_fonts() -> str:
    from matplotlib import font_manager as fm
    paths = [os.path.join(FONT_DIR, f) for f in FONT_FILES]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"bundled font not found: {missing[0]}")
    fm.fontManager.ttflist = [f for f in fm.fontManager.ttflist if f.name != FONT and f.fname not in paths]
    for p in paths:
        fm.fontManager.addfont(p)
    return FONT


def setup_nature() -> str:
    import matplotlib
    family = register_fonts()
    families = [family, FALLBACK_FONT]
    matplotlib.rcdefaults()
    matplotlib.rcParams.update({
        "font.family": families,
        "font.sans-serif": families,
        "font.size": N_LABEL,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "custom", "mathtext.rm": family, "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold", "mathtext.fallback": "stixsans",
        "text.color": "black", "axes.labelcolor": "black", "axes.edgecolor": "black",
        "axes.linewidth": N_AXIS_LW, "axes.labelsize": N_LABEL, "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": "black", "ytick.color": "black", "xtick.labelcolor": "black", "ytick.labelcolor": "black",
        "xtick.labelsize": N_TICK, "ytick.labelsize": N_TICK,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.major.width": N_AXIS_LW,
        "ytick.major.width": N_AXIS_LW, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
        "lines.solid_capstyle": "round", "legend.frameon": False,
        "figure.dpi": LAYOUT_DPI, "savefig.dpi": SAVE_DPI, "figure.facecolor": "white",
    })
    return family


def nature_axes(ax, grid_y: bool = False) -> None:
    for side, sp in ax.spines.items():
        sp.set_visible(side in ("left", "bottom"))
        sp.set_linewidth(N_AXIS_LW)
        sp.set_color("black")
    ax.tick_params(direction="out", length=2.0, width=N_AXIS_LW, pad=1.5, colors="black", labelsize=N_TICK)
    ax.set_facecolor("white")
    if grid_y:
        ax.grid(axis="y", color="#E3E3E3", lw=0.4, zorder=0)
        ax.set_axisbelow(True)


class Page:
    def __init__(self, width: float, height: float):
        import matplotlib.pyplot as plt
        self.w, self.h = width, height
        self.fig = plt.figure(figsize=(width, height))

    def rect(self, x, y, w, h):
        return [x / self.w, y / self.h, w / self.w, h / self.h]

    def axes(self, x, y, w, h, **kw):
        return self.fig.add_axes(self.rect(x, y, w, h), **kw)

    def text(self, x, y, s, **kw):
        kw.setdefault("transform", self.fig.transFigure)
        return self.fig.text(x / self.w, y / self.h, s, **kw)

    def save(self, path: str) -> None:
        save(self.fig, path)


def save(fig, path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".pdf", ".png"):
        raise ValueError(f"figure path must end in .pdf or .png, got {path!r}")
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(path, dpi=SAVE_DPI)


def text_width(fig, s: str, **kw) -> float:
    t = fig.text(0, 0, s, **kw)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def key_row(pg: Page, y: float, items: list, x: float = 0.0, align: str = "left", fontsize: float = N_KEY,
            swatch: float = 0.17) -> tuple[float, float]:
    import matplotlib.lines as mlines
    fig = pg.fig
    W, H = pg.w, pg.h
    mid = y + 0.30 * fontsize / 72
    arts = []
    for it in items:
        kind = it[0]
        if kind == "gap":
            arts.append((None, it[1], "gap"))
        elif kind == "line":
            a = mlines.Line2D([0, swatch / W], [mid / H] * 2, color=it[1], lw=it[2], solid_capstyle="round",
                              ls=it[3] if len(it) > 3 else "-", transform=fig.transFigure)
            fig.add_artist(a)
            arts.append((a, swatch, "line"))
        elif kind == "dot":
            kw = dict(marker="o", ms=3.4, mfc="white", mec="black", mew=0.9)
            kw.update(it[1] if len(it) > 1 else {})
            a = mlines.Line2D([0], [mid / H], ls="none", transform=fig.transFigure, **kw)
            fig.add_artist(a)
            arts.append((a, kw["ms"] / 72 + 0.01, "dot"))
        elif kind == "text":
            kw = dict(fontsize=fontsize, color="black")
            kw.update(it[2] if len(it) > 2 else {})
            a = fig.text(0, y / H, it[1], va="baseline", ha="left", **kw)
            arts.append((a, None, "text"))
        else:
            raise ValueError(f"unknown key item {kind!r}")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    widths = [w if w is not None else a.get_window_extent(r).width / fig.dpi for a, w, _ in arts]
    total = sum(widths)
    cur = x - total if align == "right" else (x - total / 2 if align == "center" else x)
    start = cur
    for (a, _, kind), w in zip(arts, widths):
        if kind == "line":
            a.set_xdata([cur / W, (cur + w) / W])
        elif kind == "dot":
            a.set_xdata([(cur + w / 2) / W])
        elif kind == "text":
            a.set_x(cur / W)
        cur += w
    return start, cur


def aligned_ticks(ranges, n_ticks: int = 3, pad: float = 0.05, max_decimals: int = 3):
    import itertools

    import numpy as np

    def decimals(step):
        d = 0
        while abs(round(step, d) - step) > 1e-9 and d < 8:
            d += 1
        return d

    cands = []
    for lo, hi in ranges:
        rng = hi - lo
        opts = []
        for e in range(-6, 3):
            for m in (1, 2, 2.5, 4, 5):
                step = m * 10.0 ** e
                span = step * (n_ticks - 1)
                if not (0.4 * rng <= span <= 1.6 * rng) or decimals(step) > max_decimals:
                    continue
                for k in range(int(np.floor((lo - rng) / step)) - 1, int(np.ceil((hi + rng) / step)) + 2):
                    ticks = [round((k + j) * step, 10) for j in range(n_ticks)]
                    below = (ticks[0] - lo) / span
                    above = (hi - ticks[-1]) / span
                    if below <= 0.6 and above <= 0.6:
                        opts.append((ticks, step, below, above, m in (2.5, 4)))
        cands.append(opts)
    best = None
    for combo in itertools.product(*cands):
        r_lo = max(pad, max(c[2] for c in combo) + pad)
        r_hi = max(pad, max(c[3] for c in combo) + pad)
        fills = [(hi - lo) / (c[1] * (n_ticks - 1) * (1 + r_lo + r_hi)) for c, (lo, hi) in zip(combo, ranges)]
        score = min(fills) + 0.5 * float(np.mean(fills)) - 0.03 * sum(c[4] for c in combo)
        if best is None or score > best[0]:
            best = (score, combo, r_lo, r_hi)
    if best is None:
        raise ValueError(f"no common y ticks for the panel ranges {ranges} (is a range empty?)")
    _, combo, r_lo, r_hi = best
    out = []
    for ticks, step, *_ in combo:
        span = step * (n_ticks - 1)
        d = decimals(step)
        labels = [f"{t:.{d}f}".replace("-", "−") for t in ticks]
        out.append((ticks, labels, (ticks[0] - r_lo * span, ticks[-1] + r_hi * span)))
    return out


__all__ = ["BV_DARK", "BV_TINT", "FONT", "FONT_DIR", "LAYOUT_DPI", "LW_HERO", "LW_MAIN", "N_KEY", "N_LABEL",
           "N_LETTER", "N_TICK", "Page", "SAVE_DPI", "SR_DARK", "aligned_ticks", "key_row",
           "nature_axes", "register_fonts", "save", "setup_nature", "text_width"]
