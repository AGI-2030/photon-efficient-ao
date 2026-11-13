"""Fig. S4: departures from the model.

(a) Boiling against frozen flow: steady-state Strehl ratio of the oracle and of the loops with the previous or the
    initial frame as reference at three loop periods (data/design/boiling.pkl).
(b) Scintillation: discriminability per photon of the loop, of a single frame and of the mismatch between a uniform
    and the actual pupil amplitude, against the log-amplitude variance (data/design/scintillation.pkl).
(c) Pupil asymmetry: discriminability per photon of a single frame and of the loop against the offset of a central
    obscuration and against an illumination gradient (data/theory/pupil_asymmetry.pkl).
"""

import os
import pickle
import numpy as np
from style import *
from matplotlib.lines import Line2D

DC = os.path.join(DATA, "design")
load = lambda name: pickle.load(open(os.path.join(DC, name + ".pkl"), "rb"))
BO = load("boiling")
SC = load("scintillation")
ASYMMETRY = pickle.load(open(os.path.join(DATA, "theory", "pupil_asymmetry.pkl"), "rb"))


def draw_boiling(ax):
    RAT = BO["RAT"]
    for m, lbl, col in (
        ("oracle", "oracle", NAVY),
        ("prev", "previous frame", TEAL),
        ("I0", "initial frame", CORAL),
    ):
        for model, ls, mk in (("frozen", "-", "o"), ("boiling", "--", "s")):
            y = [BO["out"][(r, model, m)]["steady"] for r in RAT]
            ax.plot(RAT, y, ls, marker=mk, color=col, label=lbl if model == "frozen" else None)
    ax.set_xscale("log")
    logticks(ax, "x", [0.1, 0.3, 1], ["0.1", "0.3", "1"])
    ax.set_xlabel(r"$\tau_{\rm loop}/\tau_0$")
    ax.set_ylabel("Strehl ratio")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1.0])
    h, l = ax.get_legend_handles_labels()
    h += [
        Line2D([], [], color="0.3", ls="-", marker="o", ms=3),
        Line2D([], [], color="0.3", ls="--", marker="s", ms=3),
    ]
    l += ["frozen flow", "boiling"]
    ax.legend(h, l, loc="lower left", handlelength=1.6, borderaxespad=0.1, labelspacing=0.2, fontsize=FS - 2)
    clean(ax)


def draw_scintillation(ax):
    SIG = [x for x in SC["SIG"] if x > 0]
    for key, lbl, col, mk in (
        ("F_loop", "loop", TEAL, "o"),
        ("F_single", "single frame", CORAL, "s"),
        ("F_model", "pupil mismatch", GREY, "^"),
    ):
        y = [np.median([SC["out"][(x, t, 0.8)][key] for t in range(8)]) for x in SIG]
        ax.loglog(SIG, y, mk + "-", color=col, label=lbl)
    ax.set_xlabel(r"log-amplitude variance $\sigma_\chi^2$")
    ax.set_ylabel("discriminability per photon")
    logticks(ax, "x", [0.01, 0.1], ["0.01", "0.1"])
    logticks(ax, "y", [0.01, 0.1, 1], ["0.01", "0.1", "1"])
    ax.set_ylim(5e-3, 3)
    ax.legend(loc="lower right", handlelength=1.5, borderaxespad=0.1, labelspacing=0.2, fontsize=FS - 2)
    clean(ax)


def draw_asymmetry(ax):
    for kind, mk, ls in (("obsc", "o", "-"), ("grad", "s", "--")):
        R = ASYMMETRY[kind]
        x = [r["par"] for r in R]
        ax.semilogy(
            x,
            [max(r["F_loop"], 1e-9) for r in R],
            ls,
            marker=mk,
            color=TEAL,
            label="loop" if kind == "obsc" else None,
        )
        ax.semilogy(
            x,
            [max(r["F_single"], 1e-9) for r in R],
            ls,
            marker=mk,
            color=CORAL,
            label="single frame" if kind == "obsc" else None,
        )
    h, l = ax.get_legend_handles_labels()
    h += [
        Line2D([], [], color="0.3", ls="-", marker="o", ms=3),
        Line2D([], [], color="0.3", ls="--", marker="s", ms=3),
    ]
    l += ["obscuration offset", "illumination gradient"]
    ax.set_xlabel("pupil asymmetry")
    ax.set_ylabel("discriminability per photon")
    ax.set_xticks([0, 0.4, 0.8])
    logticks(ax, "y", [1e-8, 1e-4, 1], pow_labels([1e-8, 1e-4, 1]))
    ax.set_ylim(1e-9, 30)
    ax.legend(h, l, loc="lower right", handlelength=1.6, borderaxespad=0.1, labelspacing=0.2, fontsize=FS - 2)
    clean(ax)


if __name__ == "__main__":
    fig, axs = plt.subplots(1, 3, figsize=(180 * IN, 56 * IN))
    draw_boiling(axs[0])
    draw_scintillation(axs[1])
    draw_asymmetry(axs[2])
    for ax, lab in zip(axs, "abc"):
        label(ax, lab, -0.3)
    fig.subplots_adjust(wspace=0.5)
    save(fig, "figure_s4")
