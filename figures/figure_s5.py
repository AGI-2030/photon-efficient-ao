"""Fig. S5: RMS even (a) and odd (b) residual of the 33 modes against iteration for the three input configurations
of Fig. 2 (D/r0 = 6, loop period 0.3 coherence times, 1e4 photons per frame, mean over 64 sequences).
Reads data/network/even_odd_residual.csv.
"""

import os
import csv
import numpy as np
from style import *

rows = [
    r
    for r in csv.reader(open(os.path.join(DATA, "network", "even_odd_residual.csv"), encoding="utf-8"))
    if r and not r[0].startswith("#")
]
D = [dict(zip(rows[0], r)) for r in rows[1:]]
k = np.array([float(r["k"]) for r in D])
ARMS = (
    ("single", "single frame", GREY, "o"),
    ("pair", "two frames", PURPLE, "s"),
    ("full", "two frames with $d$", TEAL, "D"),
)


def draw(ax, part, ylab):
    for key, lbl, col, mk in ARMS:
        y = np.array([float(r["%s_%s_rms" % (key, part)]) for r in D])
        ax.plot(k, y, "-", color=col, label=lbl)
        ax.plot(k[::5], y[::5], mk, color=col, mfc="white", mew=1.0, ms=3.2)
    ax.set_xlabel("iteration $k$")
    ax.set_xlim(0, 30)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_ylim(0, 0.4)
    ax.set_yticks([0, 0.2, 0.4])
    ax.set_title(part + " modes", fontsize=FS - 1, color="0.3", pad=4)
    if ylab:
        ax.set_ylabel("residual (rad)")
    clean(ax)


if __name__ == "__main__":
    fig, axs = plt.subplots(1, 2, figsize=(120 * IN, 55 * IN))
    draw(axs[0], "even", True)
    draw(axs[1], "odd", False)
    axs[1].set_yticklabels([])
    axs[1].legend(
        loc="upper right", handlelength=1.5, borderaxespad=0.1, labelspacing=0.25, fontsize=FS - 1.5
    )
    for ax, lab in zip(axs, "ab"):
        label(ax, lab, -0.22 if ax is axs[0] else -0.1)
    fig.subplots_adjust(wspace=0.15)
    save(fig, "figure_s5")
