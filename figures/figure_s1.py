"""Fig. S1: numerical checks of the theory.

(a) Separation of the reference images of the true wavefront and of the twin of the residual, divided by the
    product of the even residual and the even probe.
(b) Photons for the sign decision against that product, shot noise (slope -2) and read noise (slope -1).
(c) Monte Carlo sign accuracy of the ideal observer for 25 combinations of residual and probe against N F.
(d) Photons for the sign decision at the residual and probe of the loop against the change per loop period.
(e) Sign accuracy of the decision from the second-order expansion (Eq. (3)), as used in weak-phase focal-plane
    sensing, and of the exact two-frame decision against the size of the aberration, noise-free frames.
Reads data/theory/sign_accuracy.pkl and data/design/scaling.pkl; panels (a), (b) and (e) are computed here and
stored in data/theory/figure_s1.pkl.
"""

import os
import pickle
import time
import numpy as np
from scipy.special import erfc
from style import *
from aosim import Simulator

s = Simulator(N=512, Dpix=128, nmodes=33)
P, I = s.phase, s.intensity
c1 = s.turbulence_coeffs(4, seed=1)
c2 = s.turbulence_coeffs(4, seed=2)
o1 = s.scale_to_rms(s.odd(c1), 1.0)
e1 = s.scale_to_rms(s.even(c1), 1.0)
e2 = s.scale_to_rms(s.even(c2), 1.0)
# odd residual used throughout the theory figures (rad rms)
SIGMA_O = 0.5
Q = lambda x: 0.5 * erfc(x / np.sqrt(2))
CF = os.path.join(DATA, "theory", "figure_s1.pkl")


def grid():
    ag = np.logspace(-2, 0.3, 22)
    cg = np.logspace(-2, 0.3, 22)
    L0 = np.zeros((ag.size, cg.size))
    F0 = np.zeros_like(L0)
    S0 = np.zeros_like(L0)
    for ia, a in enumerate(ag):
        psi = P(SIGMA_O * o1 + a * e1)
        for ic, al in enumerate(cg):
            ce = P(al * e2)
            phi = psi + ce
            Ia, Ib = I(phi), I(phi - 2 * ce)
            L0[ia, ic] = s.l1(Ia - Ib)
            F0[ia, ic] = s.F_shot(Ia, Ib)
            S0[ia, ic] = s.S2(Ia, Ib)
    return dict(ag=ag, cg=cg, L0=L0, F0=F0, S0=S0)


def second_order_decision():
    """Sign decision from the second-order expansion (3) about the odd part of the reference wavefront, as in weak-phase focal-plane
    sensing, against the exact two-frame decision; noise-free frames; the current frame carries the residual psi, the reference frame the uncorrected
    wavefront psi + d with d the first correction (gain 1/2) plus the 0.2 rad defocus probe"""
    jdef = next(i for i, (j, n, m) in enumerate(s.modes_info) if j == 4)
    dvec = np.zeros(33)
    dvec[jdef] = 0.2

    def taylor2(beta, eps, h=0.05):
        f0 = I(beta)
        fp = I(beta + h * eps)
        fm = I(beta - h * eps)
        return f0 + (fp - fm) / (2 * h) + 0.5 * (fp - 2 * f0 + fm) / h**2

    rows = []
    for lev in (0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.5):
        ok2, okx, ab = [], [], []
        for t in range(24):
            psi_c = s.scale_to_rms(s.turbulence_coeffs(4, seed=300 + t), lev)
            d = psi_c + dvec
            Iref = I(P(psi_c + d))
            beta = P(s.odd(psi_c + d))
            ab.append(s.rms(P(psi_c + d)))
            pred = {
                sg: taylor2(beta, P(s.even((s.twin_coeffs(psi_c) if sg < 0 else psi_c) + d)))
                for sg in (+1, -1)
            }
            ok2.append(np.sum((Iref - pred[+1]) ** 2) < np.sum((Iref - pred[-1]) ** 2))
            okx.append(
                np.abs(Iref - I(P(psi_c + d))).sum() < np.abs(Iref - I(P(s.twin_coeffs(psi_c) + d))).sum()
            )
        rows.append(
            dict(
                level=lev,
                aberration=float(np.median(ab)),
                acc_so=float(np.mean(ok2)),
                acc_exact=float(np.mean(okx)),
            )
        )
        print(
            "  residual %.2f rad (aberration %.2f): second-order decision %.2f, exact %.2f"
            % (lev, rows[-1]["aberration"], rows[-1]["acc_so"], rows[-1]["acc_exact"]),
            flush=True,
        )
    return rows


D = pickle.load(open(CF, "rb")) if os.path.exists(CF) else {}
t0 = time.time()
if "grid" not in D:
    D["grid"] = grid()
if "so" not in D:
    D["so"] = second_order_decision()
if time.time() - t0 > 1:
    pickle.dump(D, open(CF, "wb"))
    print("simulations %.0fs" % (time.time() - t0))
MONTE_CARLO = pickle.load(open(os.path.join(DATA, "theory", "sign_accuracy.pkl"), "rb"))
SCALING = pickle.load(open(os.path.join(DATA, "design", "scaling.pkl"), "rb"))


def draw_bilinear(ax, fig):
    G = D["grid"]
    AG, CG = np.meshgrid(G["ag"], G["cg"], indexing="ij")
    r = G["L0"] / (AG * CG)
    r = r / r[0, 0]
    pc = ax.pcolormesh(G["cg"], G["ag"], r, cmap="viridis", shading="auto", vmin=0, vmax=1.2, rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    logticks(ax, "x", [0.01, 0.1, 1], ["0.01", "0.1", "1"])
    logticks(ax, "y", [0.01, 0.1, 1], ["0.01", "0.1", "1"])
    ax.set_xlabel(r"even probe $\|c_e\|$ (rad)")
    ax.set_ylabel(r"even residual $\|\psi_e\|$ (rad)")
    cb = fig.colorbar(pc, ax=ax, fraction=0.05, pad=0.04, ticks=[0, 0.4, 0.8, 1.2])
    cb.ax.set_title("ratio", fontsize=FS - 1, pad=4)
    cb.ax.tick_params(labelsize=FS - 1.5)
    cb.outline.set_linewidth(0.6)


def draw_photons(ax):
    G = D["grid"]
    AG, CG = np.meshgrid(G["ag"], G["cg"], indexing="ij")
    prod = (AG * CG).ravel()
    dmin, sig_r = 3.0, 2.0
    Ns = (dmin**2 / G["F0"]).ravel()
    Nr = (dmin * sig_r / np.sqrt(G["S0"])).ravel()
    ax.loglog(prod, Ns, ".", ms=2.5, color=TEAL, alpha=0.6, label="shot noise", rasterized=True)
    ax.loglog(prod, Nr, ".", ms=2.5, color=CORAL, alpha=0.6, label="read noise", rasterized=True)
    # reference slopes in the empty corner
    xx = np.array([1e-4, 1e-3])
    ax.loglog(xx, 30 * (xx / 1e-3) ** -2, ":", color=NAVY, lw=0.8)
    ax.loglog(xx, 30 * (xx / 1e-3) ** -1, "--", color=NAVY, lw=0.8)
    ax.text(5e-4, 500, "−2", fontsize=FS - 2, ha="left", va="center")
    ax.text(1.2e-4, 22, "−1", fontsize=FS - 2, ha="left", va="center")
    ax.set_xlabel(r"$\|\psi_e\|\,\|c_e\|$ (rad$^2$)")
    ax.set_ylabel(r"$N_{\min}$ (photons)")
    logticks(ax, "x", [1e-4, 1e-2, 1], pow_labels([1e-4, 1e-2, 1]))
    logticks(ax, "y", [1e1, 1e3, 1e5], pow_labels([1e1, 1e3, 1e5]))
    ax.set_xlim(5e-5, 3)
    ax.set_ylim(3, 3e6)
    ax.legend(loc="upper right", handlelength=1.0, markerscale=3, borderaxespad=0.2)
    clean(ax)


def draw_montecarlo(ax):
    cm = plt.cm.viridis(np.linspace(0.1, 0.9, len(MONTE_CARLO)))
    for k, r in enumerate(MONTE_CARLO):
        ax.semilogx(r["Ns"] * r["F"], r["acc"], "o", ms=2.5, color=cm[k], alpha=0.75, mew=0)
    x = np.logspace(-2, 3, 200)
    ax.semilogx(x, 1 - Q(np.sqrt(x) / 2), "-", color=NAVY, lw=1.0, label=r"$1-Q(\sqrt{NF}/2)$")
    ax.set_xlabel(r"$N\,F$")
    ax.set_ylabel("sign accuracy")
    ax.set_ylim(0.4, 1.02)
    ax.set_yticks([0.5, 0.75, 1.0])
    ax.set_xlim(1e-2, 2e3)
    logticks(ax, "x", [1e-2, 1, 1e2], pow_labels([1e-2, 1, 1e2]))
    ax.axhline(0.5, color=GREY, lw=0.6, ls=":")
    ax.legend(loc="lower right", borderaxespad=0.2)
    clean(ax)


def draw_loop(ax):
    ff = SCALING["ff"]
    RAT = SCALING["RAT"]
    Dl = np.array([np.median([x["delta"] for x in ff[r]]) for r in RAT])
    NS = np.array([np.median([x["nmin_shot"] for x in ff[r]]) for r in RAT])
    NR = np.array([np.median([x["nmin_read"] for x in ff[r]]) for r in RAT])
    ax.loglog(Dl, NS, "o-", color=TEAL, label="shot noise")
    ax.loglog(Dl, NR, "s-", color=CORAL, label="read noise")
    ax.loglog(Dl, NS[0] * (Dl / Dl[0]) ** -2, ":", color=NAVY, lw=0.8)
    ax.text(Dl[-1] * 1.15, NS[0] * (Dl[-1] / Dl[0]) ** -2, "slope −2", fontsize=FS - 2, va="center")
    for r, x, y in zip(RAT, Dl, NR):
        ax.text(x, y * 1.6, "%g" % r, fontsize=FS - 2.5, ha="center", color=CORAL)
    ax.text(Dl[0] * 0.8, NR[0] * 4.5, r"$\tau_{\rm loop}/\tau_0$", fontsize=FS - 2.5, ha="left", color=CORAL)
    ax.set_xlabel("wavefront change per loop period (rad)")
    ax.set_ylabel(r"$N_{\min}$ (photons)")
    logticks(ax, "x", [0.01, 0.1, 1], ["0.01", "0.1", "1"])
    logticks(ax, "y", [1e1, 1e3, 1e5], pow_labels([1e1, 1e3, 1e5]))
    ax.set_xlim(0.01, 1.2)
    ax.set_ylim(3, 3e6)
    ax.legend(loc="lower left", handlelength=1.5, borderaxespad=0.2)
    clean(ax)


def draw_expansion(ax):
    R = D["so"]
    x = [r["aberration"] for r in R]
    ax.semilogx(x, [r["acc_exact"] for r in R], "o-", color=TEAL, label="exact decision")
    ax.semilogx(x, [r["acc_so"] for r in R], "^-", color=CORAL, label="second-order expansion (3)")
    ax.axhline(0.5, color=GREY, lw=0.6, ls=":")
    ax.set_xlabel("aberration (rad)")
    ax.set_ylabel("sign accuracy")
    logticks(ax, "x", [0.3, 1, 3], ["0.3", "1", "3"])
    ax.set_xlim(0.17, 3.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1.0])
    ax.legend(loc="center left", handlelength=1.5, borderaxespad=0.2)
    clean(ax)


if __name__ == "__main__":
    fig, axs = plt.subplots(2, 3, figsize=(180 * IN, 112 * IN))
    axs = axs.ravel()
    draw_bilinear(axs[0], fig)
    draw_photons(axs[1])
    draw_montecarlo(axs[2])
    draw_loop(axs[3])
    draw_expansion(axs[4])
    fig.delaxes(axs[5])
    for ax, lab in zip(axs[:5], "abcde"):
        label(ax, lab, -0.3)
    fig.subplots_adjust(wspace=0.55, hspace=0.45)
    save(fig, "figure_s1")
    print("second-order decision:", [(round(r["aberration"], 2), round(r["acc_so"], 2)) for r in D["so"]])
