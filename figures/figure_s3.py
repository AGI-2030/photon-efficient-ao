"""Fig. S3: ideal observer and ideal estimator in a time-varying loop (frozen flow, D/r0 = 6).

(a) Steady-state Strehl ratio of the oracle and of the ideal estimator against the loop period, with the fitting
    limit of the 33 modes; (b) sign accuracy of the ideal estimator (data/design/loop_period.csv).
(c) Steady-state Strehl ratio against D/r0 with the fitting residual in the images (data/design/turbulence_strength.pkl).
(d) Steady-state encircled energy of the ideal estimator against photons per frame (data/design/photons.csv).
(e) Steady-state Strehl ratio with the previous frame as reference against the loop gain, with magnitude errors of
    0.02 (solid) and 0.07 rad (dashed) per mode (data/reference_frames/gain.pkl).
(f) Steady-state encircled energy of the network trained at g = 1/2 against the loop gain (data/network/gain_sweep.csv).
"""

import os
import csv
import pickle
import numpy as np
from style import *
from matplotlib.lines import Line2D


def read(name):
    rows = [
        r
        for r in csv.reader(open(os.path.join(DATA, name), encoding="utf-8"))
        if r and not r[0].startswith("#")
    ]
    return [dict(zip(rows[0], r)) for r in rows[1:]]


f = lambda rows, key: np.array([float(r[key]) for r in rows])
RHO = read("design/loop_period.csv")
NPH = read("design/photons.csv")
GN = read("network/gain_sweep.csv")
GD = pickle.load(open(os.path.join(DATA, "reference_frames", "gain.pkl"), "rb"))
DR = pickle.load(open(os.path.join(DATA, "design", "turbulence_strength.pkl"), "rb"))
PERIODS = ((0.1, NAVY, "s"), (0.3, TEAL, "o"), (1.0, GOLD, "^"))
XT = [0.03, 0.1, 0.3, 1]
XL = ["0.03", "0.1", "0.3", "1"]


def draw_a(ax):
    x = f(RHO, "flow")
    sel = x > 0
    x = x[sel]
    for key, lbl, col, mk in (("oracle", "oracle", NAVY, "s"), ("ideal", "ideal estimator", TEAL, "o")):
        m = f(RHO, key + "_strehl")[sel]
        sd = f(RHO, key + "_strehl_sd")[sel]
        ax.plot(x, m, mk + "-", color=col, label=lbl)
        ax.fill_between(x, m - sd, m + sd, color=col, alpha=0.15, lw=0)
    ax.axhline(f(RHO, "strehl_bound")[0], color=GREY, lw=0.8, ls="--", label="fitting bound")
    ax.set_xscale("log")
    logticks(ax, "x", XT, XL)
    ax.set_xlabel(r"$\tau_{\rm loop}/\tau_0$")
    ax.set_ylabel("Strehl ratio")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    ax.legend(loc="lower left", handlelength=1.0, borderaxespad=0.1, labelspacing=0.2, fontsize=FS - 2)
    clean(ax)


def draw_b(ax):
    x = f(RHO, "flow")
    sel = x > 0
    x = x[sel]
    ax.plot(x, f(RHO, "sign_frac_first")[sel], "^-", color=CORAL, label="first decision")
    ax.plot(x, f(RHO, "sign_frac_steady")[sel], "v-", color=TEAL, label="steady state")
    ax.plot(x, f(RHO, "sign_ok")[sel], "s-", color=NAVY, label="per frame")
    ax.axhline(0.5, color=GREY, lw=0.6, ls=":")
    ax.set_xscale("log")
    logticks(ax, "x", XT, XL)
    ax.set_xlabel(r"$\tau_{\rm loop}/\tau_0$")
    ax.set_ylabel("correct sign")
    ax.set_ylim(0.4, 1.02)
    ax.set_yticks([0.5, 0.75, 1.0])
    ax.legend(loc="lower right", handlelength=1.2, borderaxespad=0.2, labelspacing=0.2, fontsize=FS - 1.5)
    clean(ax)


def draw_c(ax):
    out = DR["out"]
    DRS = DR["DR0S"]
    for (Nph, shot), lbl, col, mk in zip(
        DR["CONF"],
        (
            r"$\mathdefault{10^4}$ photons",
            r"$\mathdefault{10^5}$ photons",
            r"$\mathdefault{10^4}$, shot noise",
        ),
        (TEAL, NAVY, PURPLE),
        ("o", "s", "^"),
    ):
        y = [
            np.array([[r["strehl"] for r in rr] for rr in out[(d, Nph, shot)]["res"]])[:, 6:].mean()
            for d in DRS
        ]
        ax.plot(DRS, y, mk + "-", color=col, label=lbl)
    unc = [
        np.array([[r["strehl"] for r in rr] for rr in out[(d, 1e4, False)]["res"]])[:, 0].mean() for d in DRS
    ]
    bound = [np.mean([i["strehl_bound"] for i in out[(d, 1e4, False)]["info"]]) for d in DRS]
    ax.plot(DRS, unc, ":", color=GREY)
    ax.plot(DRS, bound, "--", color=GREY, lw=0.8)
    ax.set_xscale("log")
    logticks(ax, "x", [1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    ax.set_xlabel("$D/r_0$")
    ax.set_ylabel("Strehl ratio")
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.5, 1.0])
    ax.legend(loc="lower left", handlelength=1.2, borderaxespad=0.1, labelspacing=0.2, fontsize=FS - 2)
    clean(ax)


def draw_d(ax):
    N = f(NPH, "photons")
    m = f(NPH, "ideal_ec")
    sd = f(NPH, "ideal_ec_sd")
    ax.semilogx(N, m, "o-", color=TEAL, label="ideal estimator")
    ax.fill_between(N, m - sd, m + sd, color=TEAL, alpha=0.15, lw=0)
    ax.axhline(f(NPH, "oracle_ec")[0], color=NAVY, lw=0.9, ls="--", label="oracle")
    ax.axhline(f(NPH, "ec_bound")[0], color=GREY, lw=0.8, ls="--")
    ax.axhline(f(NPH, "ec_uncorrected")[0], color=GREY, lw=0.8, ls=":")
    logticks(ax, "x", [1e3, 1e4, 1e5, 1e6], pow_labels([1e3, 1e4, 1e5, 1e6]))
    ax.set_xlabel("photons per frame $N$")
    ax.set_ylabel("encircled energy")
    ax.set_ylim(0, 0.8)
    ax.set_yticks([0, 0.4, 0.8])
    ax.legend(loc="lower right", handlelength=1.2, borderaxespad=0.1, labelspacing=0.2, fontsize=FS - 2)
    clean(ax)


def draw_e(ax):
    g = GD["gains"]
    for ratio, col, mk in PERIODS:
        for sa, ls in ((0.02, "-"), (0.07, "--")):
            y = [GD["out"][(ratio, gg, "prev", sa)]["steady"] for gg in g]
            ax.plot(g, y, ls, marker=mk, ms=2.5, color=col)
    ax.axvline(0.5, color=GREY, lw=0.7, ls=":")
    ax.set_xlabel("loop gain $g$")
    ax.set_ylabel("Strehl ratio")
    ax.set_xlim(0.15, 1.05)
    ax.set_xticks([0.2, 0.5, 0.8])
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.5, 1.0])
    hh = [Line2D([], [], color="0.3", ls="-", lw=1.3), Line2D([], [], color="0.3", ls="--", lw=1.3)]
    ax.legend(
        hh,
        ["0.02 rad noise", "0.07 rad noise"],
        loc="lower right",
        handlelength=1.6,
        borderaxespad=0.1,
        labelspacing=0.2,
        fontsize=FS - 2,
    )
    clean(ax)


def draw_f(ax):
    for ratio, col, mk in PERIODS:
        sel = sorted(
            [r for r in GN if abs(float(r["tau_ratio"]) - ratio) < 1e-9], key=lambda r: float(r["gain"])
        )
        g = np.array([float(r["gain"]) for r in sel])
        m = np.array([float(r["ec_mean"]) for r in sel])
        ax.plot(g, m, mk + "-", color=col, label="%g" % ratio)
    ax.axvline(0.5, color=GREY, lw=0.7, ls=":")
    ax.set_xlabel("loop gain $g$")
    ax.set_ylabel("encircled energy")
    ax.set_xlim(0.15, 1.05)
    ax.set_xticks([0.2, 0.5, 0.8])
    ax.set_ylim(0, 0.6)
    ax.set_yticks([0, 0.3, 0.6])
    ax.legend(
        loc="lower left",
        title=r"$\tau_{\rm loop}/\tau_0$",
        title_fontsize=FS - 1.5,
        handlelength=1.5,
        borderaxespad=0.2,
        labelspacing=0.2,
        fontsize=FS - 1.5,
    )
    clean(ax)


if __name__ == "__main__":
    fig, axs = plt.subplots(2, 3, figsize=(180 * IN, 105 * IN))
    axs = axs.ravel()
    for fn, ax in zip((draw_a, draw_b, draw_c, draw_d, draw_e, draw_f), axs):
        fn(ax)
    for ax, lab in zip(axs, "abcdef"):
        label(ax, lab, -0.3)
    fig.subplots_adjust(wspace=0.5, hspace=0.45)
    save(fig, "figure_s3")
