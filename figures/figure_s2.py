"""Fig. S2: parameters of the static loop.

(a) Photons for the first sign decision against the defocus added to the first command, worst of 16 aberrations
    of 0.8 rad, with read noise, for a first estimate with the correct sign, the wrong sign or no even part.
(b) Cramer-Rao error of the 33 modes against the half-size of the camera window, 0.8 rad, 1e4 detected photons.
Reads data/design/probe.pkl and data/design/window.pkl (python -m simulations.loop_design probe / window).
"""

import os
import pickle
import numpy as np
from style import *

DC = os.path.join(DATA, "design")
load = lambda name: pickle.load(open(os.path.join(DC, name + ".pkl"), "rb"))
PR = load("probe")
WI = load("window")


def draw_probe(ax):
    AP = PR["AP"]
    po = PR["out"]
    for case, lbl, col, mk in (
        ("right", "correct sign", TEAL, "o"),
        ("wrong", "wrong sign", CORAL, "s"),
        ("hedged", "no even part", PURPLE, "^"),
    ):
        mx = [np.max([po[(0.8, t, 1e4, case, ap)]["nmin_read"] for t in range(16)]) for ap in AP]
        ax.semilogy(AP, mx, mk + "-", color=col, label=lbl)
    ax.axvline(0.2, color=GREY, lw=0.7, ls=":")
    ax.set_xlabel("defocus probe (rad)")
    ax.set_ylabel(r"$N_{\min}$ (photons)")
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    logticks(ax, "y", [1e3, 1e4, 1e5], pow_labels([1e3, 1e4, 1e5]))
    ax.set_ylim(5e2, 3e5)
    ax.legend(
        loc="upper right",
        title="first estimate",
        title_fontsize=FS - 1,
        handlelength=1.5,
        borderaxespad=0.2,
        labelspacing=0.2,
    )
    clean(ax)


def draw_window(ax):
    RW = WI["RW"]
    wo = WI["out"]
    for key, lbl, col, ls in (
        ("one_read_4", "one frame, read noise", CORAL, "-"),
        ("two_read_4", "two frames, read noise", TEAL, "-"),
        ("one_shot_4", "one frame, shot noise", GREY, "--"),
    ):
        y = [np.median([wo[(0.8, t, R)][key] for t in range(8)]) for R in RW]
        ax.semilogy(RW, y, "o", ls=ls, color=col, label=lbl)
    ax.axvline(10, color=GREY, lw=0.7, ls=":")
    ax.set_xlabel("window half-size (λ/D)")
    ax.set_ylabel("Cramér–Rao error (rad)")
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlim(0, 21)
    logticks(ax, "y", [0.1, 1, 10], ["0.1", "1", "10"])
    ax.set_ylim(0.03, 30)
    ax.legend(loc="upper right", handlelength=1.2, borderaxespad=0.2, labelspacing=0.2, fontsize=FS - 2)
    clean(ax)


if __name__ == "__main__":
    fig, axs = plt.subplots(1, 2, figsize=(120 * IN, 56 * IN))
    draw_probe(axs[0])
    draw_window(axs[1])
    for ax, lab in zip(axs, "ab"):
        label(ax, lab, -0.3)
    fig.subplots_adjust(wspace=0.5)
    save(fig, "figure_s2")
    po = PR["out"]
    print(
        "probe 0 vs 0.2 rad, no even part, 0.8 rad: %.0f -> %.0f photons"
        % tuple(
            np.max([po[(0.8, t, 1e4, "hedged", ap)]["nmin_read"] for t in range(16)]) for ap in (0.0, 0.2)
        )
    )
