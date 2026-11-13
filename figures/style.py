"""Common style of the supplementary figures: Arial (or Liberation Sans), 9.5 pt, 500 dpi.

Figures are written to figures/output/ as PDF and PNG.
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
DATA = os.path.join(os.path.dirname(HERE), "data")
os.makedirs(OUT, exist_ok=True)
IN = 1 / 25.4
DPI = 500
FS = 9.5
_have = {f.name for f in font_manager.fontManager.ttflist}
SANS = next((f for f in ("Arial", "Liberation Sans", "DejaVu Sans") if f in _have), "DejaVu Sans")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [SANS],
        "font.size": FS,
        "axes.labelsize": FS,
        "axes.titlesize": FS,
        "xtick.labelsize": FS - 0.5,
        "ytick.labelsize": FS - 0.5,
        "legend.fontsize": FS - 1,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.5,
        "mathtext.fontset": "custom",
        "mathtext.rm": SANS,
        "mathtext.it": SANS + ":italic",
        "mathtext.bf": SANS + ":bold",
        "mathtext.default": "it",
    }
)
TEAL = "#0F7B84"  # this work: the loop, the previous-frame reference, the network
CORAL = "#D9542B"  # the initial-frame reference and other alternatives
PURPLE = "#5A4FC7"  # two frames without d, phase diversity, continuity
GREY = "#9AA5B1"  # single frame, uncorrected
NAVY = "#11263F"  # oracle, bounds, predictions
GOLD = "#C99A2E"  # a second condition
REF = "0.35"  # the Cramér-Rao reference
LIGHT = "#7FB7BE"  # a lighter teal (second Shack-Hartmann)


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def label(ax, s, dx=-0.2, dy=1.03):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=FS + 2, fontweight="bold", va="bottom", ha="left")


def logticks(ax, axis, ticks, labels=None):
    """Major ticks only on a logarithmic axis."""
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    a.set_minor_locator(matplotlib.ticker.NullLocator())
    if labels is not None:
        a.set_major_formatter(matplotlib.ticker.FixedFormatter(labels))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def pow_labels(ticks):
    return [r"$\mathdefault{10^{%d}}$" % int(round(np.log10(t))) for t in ticks]
