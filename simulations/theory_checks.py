"""Numerical checks of the theory with the ideal observer and a classical retrieval algorithm.

Stages:
    sign_accuracy                 Monte Carlo sign accuracy for 25 combinations of even residual and probe (Fig. S1(c))
    pupil_asymmetry               sign information of one frame for a noncentrosymmetric pupil (Fig. S4(c), Remark R2)
    alternating_projection        two-image alternating projection with a known correction, noise-free (Note S5)
    alternating_projection_noise  the same with photon and read noise, with maximum-likelihood refinement (Note S5)
    summary                       print the numbers quoted in Note S5

Usage: python -m simulations.theory_checks <stage> [<stage> ...]. Results are written to data/theory/.
"""

import os
import sys
import time
import pickle
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.special import erfc
from aosim import Simulator, data_dir, ideal_observer_accuracy

CACHE = data_dir("theory")
STAGES = set(sys.argv[1:])
KNOWN = {
    "sign_accuracy",
    "pupil_asymmetry",
    "alternating_projection",
    "alternating_projection_noise",
    "summary",
}
if __name__ == "__main__" and (not STAGES or STAGES - KNOWN):
    sys.exit(__doc__)
Q = lambda x: 0.5 * erfc(x / np.sqrt(2))
save = lambda name, obj: pickle.dump(obj, open(os.path.join(CACHE, name + ".pkl"), "wb"))
load = lambda name: pickle.load(open(os.path.join(CACHE, name + ".pkl"), "rb"))

s = Simulator(N=512, Dpix=128, nmodes=33)
P, I = s.phase, s.intensity
c1 = s.turbulence_coeffs(4, seed=1)
c2 = s.turbulence_coeffs(4, seed=2)
o1 = s.scale_to_rms(s.odd(c1), 1.0)
e1 = s.scale_to_rms(s.even(c1), 1.0)
e2 = s.scale_to_rms(s.even(c2), 1.0)
SIGMA_O = 0.5
crop = lambda im: s.crop(im, 20)

if "sign_accuracy" in STAGES:
    t0 = time.time()
    rng = np.random.default_rng(12)
    grid = [0.05, 0.1, 0.2, 0.4, 0.8]
    Ns = np.logspace(1, 6, 11)
    out = []
    for a in grid:
        for al in grid:
            psi = P(SIGMA_O * o1 + a * e1)
            ce = P(al * e2)
            phi = psi + ce
            # frame 0 of truth and of the residual twin
            Ia, Ib = crop(I(phi)), crop(I(phi - 2 * ce))
            F = s.F_shot(Ia, Ib)
            acc = [ideal_observer_accuracy(Ia, Ib, N, 200, rng) for N in Ns]
            out.append(dict(a=a, al=al, F=F, Ns=Ns, acc=np.array(acc)))
    save("sign_accuracy", out)
    print("sign_accuracy done %.0fs" % (time.time() - t0))


def pupil_variant(kind, par):
    """Pupil amplitude on the 512 grid: "obsc", a central obscuration of radius 0.25 R offset by par R along x,
    or "grad", an illumination 1 + par x / R."""
    x = np.arange(s.N) - (s.N - 1) / 2.0
    X, Y = np.meshgrid(x, x)
    R = s.Dpix / 2.0
    A = s.pupil.astype(float)
    if kind == "obsc":
        A = A * (~(np.hypot(X - par * R, Y) <= 0.25 * R)).astype(float)
    elif kind == "grad":
        A = A * np.clip(1.0 + par * X / R, 0, None)
    return A


def intensity_A(A, ph):
    Iv = np.abs(fftshift(fft2(A * np.exp(1j * ph)))) ** 2
    return Iv / Iv.sum()


if "pupil_asymmetry" in STAGES:
    t0 = time.time()
    out = {}
    phi = P(SIGMA_O * o1 + 0.5 * e1)
    corr = 0.5 * phi
    ce = P(0.5 * 0.5 * e1)
    for kind, pars in (("obsc", [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]), ("grad", [0.0, 0.05, 0.1, 0.2, 0.4, 0.8])):
        rows = []
        for par in pars:
            A = pupil_variant(kind, par)
            # single frame, truth vs global twin
            I0, I0t = crop(intensity_A(A, phi)), crop(intensity_A(A, s.twin_map(phi)))
            # frame 0, truth vs residual twin (loop)
            Ir, Irt = crop(intensity_A(A, phi)), crop(intensity_A(A, phi - 2 * ce))
            rows.append(
                dict(
                    par=par,
                    sep_single=s.l1(I0 - I0t),
                    F_single=s.F_shot(I0, I0t),
                    sep_loop=s.l1(Ir - Irt),
                    F_loop=s.F_shot(Ir, Irt),
                )
            )
        out[kind] = rows
    save("pupil_asymmetry", out)
    print("pupil_asymmetry done %.0fs" % (time.time() - t0))
    for kind in out:
        for r in out[kind]:
            print(
                "  %s par=%.2f single-frame sep %.2e (Nmin %.1e) | loop sep %.3f (Nmin %.0f)"
                % (
                    kind,
                    r["par"],
                    r["sep_single"],
                    9 / max(r["F_single"], 1e-30),
                    r["sep_loop"],
                    9 / r["F_loop"],
                )
            )


def gs_two_image(I0, I1, cphase, A, phi_init, iters, truth, every=1, return_phase=False):
    """Alternating projection (Gerchberg-Saxton type) with two measured intensities and a known correction."""
    sq0, sq1 = np.sqrt(I0), np.sqrt(I1)
    ph = phi_init.copy()
    hist = []
    for it in range(iters):
        U = fftshift(fft2(A * np.exp(1j * ph)))
        U = sq0 * np.exp(1j * np.angle(U))
        u = ifft2(ifftshift(U))
        ph = np.angle(u)
        V = fftshift(fft2(A * np.exp(1j * (ph - cphase))))
        V = sq1 * np.exp(1j * np.angle(V))
        v = ifft2(ifftshift(V))
        ph = np.angle(v) + cphase
        if it % every == 0:
            # 33 modes, Noll j = 4 to 36 (no tip or tilt in the basis)
            err = s.project(ph) - truth
            hist.append(s.rms(P(err)))
    return (np.array(hist), ph) if return_phase else np.array(hist)


if "alternating_projection" in STAGES:
    t0 = time.time()
    rng = np.random.default_rng(13)
    out = []
    ITERS = 300
    for t in range(12):
        coef = s.turbulence_coeffs(4, seed=900 + t)
        coef = s.scale_to_rms(coef, 0.8)
        # first correction with random even sign
        phi = P(coef)
        corr = 0.5 * P(coef * (1 - 2 * (rng.random() < 0.5) * s.even_mask))
        I0, I1 = I(phi), I(phi - corr)
        for start in ("zero", "random"):
            init = (
                np.zeros((s.N, s.N)) if start == "zero" else P(s.scale_to_rms(rng.standard_normal(33), 0.5))
            )
            tt = time.time()
            hist = gs_two_image(I0, I1, corr, s.A, init, ITERS, coef, every=5)
            dt = (time.time() - tt) / ITERS
            out.append(
                dict(
                    trial=t,
                    start=start,
                    hist=hist,
                    dt=dt,
                    final=hist[-1],
                    iters_to_01=int(5 * np.argmax(hist < 0.1)) if (hist < 0.1).any() else None,
                )
            )
            print(
                "  trial %d start=%-6s final err %.3f rad, iters to 0.1 rad: %s, %.1f ms per iteration"
                % (t, start, hist[-1], out[-1]["iters_to_01"], 1e3 * dt)
            )
    save("alternating_projection", out)
    print("alternating_projection done %.0fs" % (time.time() - t0))

ROI_LD = 10.0  # half-width of the camera window in lambda/D (as in photon_comparison.py)
READ_E = 7.0  # read noise of the far-field camera, electrons rms per pixel
EST = np.ones(33, bool)  # all 33 modes (Noll j = 4 to 36) are estimated


def roi_mask():
    px_ld = s.N / s.Dpix
    h = int(round(ROI_LD * px_ld))
    c = s.N // 2
    m = np.zeros((s.N, s.N), bool)
    m[c - h : c + h, c - h : c + h] = True
    return m


def noisy_frames(phi, corr, N, sig_r, rng):
    """Two detected frames (electrons), of the wavefront and of the corrected residual, with Poisson and read noise in the camera window."""
    m = roi_mask()
    out = []
    for ph in (phi, phi - corr):
        Ih = I(ph)
        lam = N * Ih
        y = rng.poisson(lam).astype(float) + sig_r * rng.standard_normal(lam.shape)
        y[~m] = 0.0
        out.append(y)
    return out


def jac_fd(coef, corr, m, h=1e-3):
    """Finite-difference Jacobian of the two normalized window images with respect to the 33 modes."""
    cols = []
    for j in np.where(EST)[0]:
        d = np.zeros(33)
        d[j] = h
        pp, pm = P(coef + d), P(coef - d)
        cols.append(np.concatenate([(I(pp) - I(pm))[m], (I(pp - corr) - I(pm - corr))[m]]) / (2 * h))
    return np.array(cols).T  # (2*npix, 33)


def crb_two_frame(coef, corr, N, sig_r):
    m = roi_mask()
    J = jac_fd(coef, corr, m)
    var = np.concatenate([N * I(P(coef))[m], N * I(P(coef) - corr)[m]]) + sig_r**2
    # rad rms over the 33 modes (orthonormal basis)
    F = (N * J).T @ ((N * J) / var[:, None])
    return np.sqrt(np.trace(np.linalg.inv(F)))


def ml_refine(coef0, corr, y0, y1, N, sig_r, iters=6):
    """Levenberg-Marquardt on the Gaussian approximation to the Poisson and read-noise likelihood, started from coef0."""
    m = roi_mask()
    y = np.concatenate([y0[m], y1[m]])
    coef = coef0.copy()

    def model(c):
        return np.concatenate([N * I(P(c))[m], N * I(P(c) - corr)[m]])

    def ll(c):
        mu = model(c)
        var = np.maximum(mu, 0) + sig_r**2
        return -0.5 * np.sum((y - mu) ** 2 / var + np.log(var))

    lam = 1e-3
    l0 = ll(coef)
    for _ in range(iters):
        mu = model(coef)
        var = np.maximum(mu, 0) + sig_r**2
        J = N * jac_fd(coef, corr, m)
        A = J.T @ (J / var[:, None])
        g = J.T @ ((y - mu) / var)
        dg = np.diag(np.diag(A)) + 1e-9 * np.eye(33)
        acc = False
        for _try in range(4):
            step = np.linalg.solve(A + lam * dg, g)
            cand = coef.copy()
            cand[EST] += step
            l1 = ll(cand)
            if l1 > l0:
                coef, l0, lam, acc = cand, l1, max(lam / 3, 1e-6), True
                break
            lam *= 10
        if not acc or np.sqrt(np.mean(step**2)) < 1e-3:
            break
    return coef


if "alternating_projection_noise" in STAGES:
    t0 = time.time()
    rng = np.random.default_rng(131)
    out = []
    ITERS = 300
    NLEV = [1e3, 1e4, 1e5, 1e6]
    m = roi_mask()
    for t in range(10):
        coef = s.turbulence_coeffs(4, seed=900 + t)
        coef = s.scale_to_rms(coef, 0.8)
        phi = P(coef)
        corr = 0.5 * P(coef * (1 - 2 * (rng.random() < 0.5) * s.even_mask))
        for N in NLEV:
            y0, y1 = noisy_frames(phi, corr, N, READ_E, rng)
            # measured intensities, window only, normalized
            M0, M1 = np.clip(y0, 0, None) / N, np.clip(y1, 0, None) / N
            tt = time.time()
            hist, ph = gs_two_image(
                M0, M1, corr, s.A, np.zeros((s.N, s.N)), ITERS, coef, every=5, return_phase=True
            )
            dt = (time.time() - tt) / ITERS
            # 33-mode estimate at the end of GS
            c_gs = s.project(ph)
            e_gs = s.rms(P(c_gs - coef))
            c_ml = ml_refine(c_gs, corr, y0, y1, N, READ_E)
            e_ml = s.rms(P(c_ml - coef))
            crb = crb_two_frame(coef, corr, N, READ_E)
            out.append(dict(trial=t, N=N, hist=hist, dt=dt, e_gs=e_gs, e_ml=e_ml, crb=crb))
            print(
                "  trial %d N=%.0e: GS %.3f rad, GS+ML %.3f rad, CRB %.3f rad (%.0fs)"
                % (t, N, e_gs, e_ml, crb, time.time() - t0),
                flush=True,
            )
    save("alternating_projection_noise", out)
    print("alternating_projection_noise done %.0fs" % (time.time() - t0))


if "summary" in STAGES:
    out = load("alternating_projection")
    for start in ("zero", "random"):
        ok = np.mean([r["iters_to_01"] is not None for r in out if r["start"] == start])
        print(
            "noise-free frames, %s start: %.0f%% of %d aberrations below 0.1 rad"
            % (start, 100 * ok, len({r["trial"] for r in out}))
        )
    print("%.1f ms per iteration" % (1e3 * np.mean([r["dt"] for r in out])))
    noisy = load("alternating_projection_noise")
    for N in sorted({r["N"] for r in noisy}):
        rows = [r for r in noisy if r["N"] == N]
        med = lambda key: float(np.median([r[key] for r in rows]))
        print(
            "N = %.0e: alternating projection %.3f rad, with ML refinement %.3f rad, CRB %.3f rad"
            % (N, med("e_gs"), med("e_ml"), med("crb"))
        )
