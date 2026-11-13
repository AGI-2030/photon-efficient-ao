"""Photon efficiency of four sensing methods in the same loop with ideal estimators (Fig. 5(a), Table 3, Note S5).

Methods:
    image_loop        far-field frames with the reference frame and the known correction (ideal ML estimator)
    phase_diversity   in-focus and defocused frames, half of the photons each (ideal ML estimator)
    sh_beacon         Shack-Hartmann sensor, 37 subapertures, 532 nm beacon, centroids and least-squares reconstructor
    sh_main_beam      the same Shack-Hartmann sensor on the 1080 nm main beam
All methods use the same aberrations, gain 1/2 and N detected photons per correction at their sensor.

Usage:
    python -m simulations.photon_comparison run <method> <read|shot> [--flow 0.3] [--sigma 0.8]
    python -m simulations.photon_comparison summary
--flow sets tau_loop/tau_0 of frozen-flow sequences (default 0, static aberrations with the initial frame as
reference). Results are written to data/photon_comparison/.
"""

import os
import sys
import time
import pickle
import numpy as np
from numpy.fft import fft2, fftshift
from aosim import Simulator, data_dir

CACHE = data_dir("photon_comparison")

LAM_MAIN, LAM_SH = 1080e-9, 532e-9
D_PUPIL, FOCAL_L, PX_FF = 0.105, 5.0, 5.86e-6  # exit pupil, focusing distance, far-field pixel
# pupil at the SH, lenslet pitch and focal length, SH pixel
D_SH, PITCH, F_SH, PX_SH = 1.4e-3, 200e-6, 10e-3, 4.5e-6
READ_FF, READ_SH = 7.0, 4.0  # read noise, electrons rms
ROI_LD = 10.0  # far-field window half-size, lambda/D
PD_DEFOCUS = 1.0  # defocus diversity, rad rms at 1080 nm
PD_SPLIT = 0.5
SH_WIN = 16  # centroiding window, pixels
G, K, TRIALS = 0.5, 10, 8
SIGMA = 0.8  # tilt-removed aberration, rad rms at 1080 nm
# tau_loop/tau_0 of a frozen-flow sequence; 0 = static aberration (initial frame as reference)
FLOW = 0.0
NLEVELS = np.logspace(2, 6, 9)

s = Simulator(N=384, Dpix=44, nmodes=33)
assert abs(s.ppl - LAM_MAIN * FOCAL_L / D_PUPIL / PX_FF) < 0.2, s.ppl
# the basis (Noll j = 4 to 36) contains no tip or tilt
TILT = np.zeros(33, bool)
EST = ~TILT  # all 33 modes are estimated
NM = int(EST.sum())
Zp_est = s.Z[EST]
HALF = int(round(ROI_LD * s.ppl))
C0 = s.N // 2
NORM = None
PEAK0 = s.intensity(np.zeros((s.N, s.N)))[C0, C0]


def field(ph):
    return fftshift(fft2(s.A * np.exp(1j * ph)))


def inorm(ph):
    """Normalized intensity in the camera window (fraction of all photons per pixel)."""
    global NORM
    Gf = field(ph)
    I = np.abs(Gf) ** 2
    if NORM is None:
        NORM = I.sum()
    return I[C0 - HALF : C0 + HALF, C0 - HALF : C0 + HALF] / NORM


def jac(ph):
    """Derivative of the normalized window intensity with respect to the 33 estimated modes, shape (33, npix)."""
    Gf = field(ph)
    u = s.A * np.exp(1j * ph)
    rows = []
    for j in range(Zp_est.shape[0]):
        Kj = fftshift(fft2(u * Zp_est[j]))
        dI = -2.0 * np.imag(np.conj(Gf) * Kj) / NORM
        rows.append(dI[C0 - HALF : C0 + HALF, C0 - HALF : C0 + HALF].ravel())
    return np.array(rows)


def coef_full(est):
    c = np.zeros(33)
    c[EST] = est
    return c


def strehl(coef):
    return s.intensity(s.phase(coef))[C0, C0] / PEAK0


def normal_eqs(est, frames, sig_r):
    JtWJ = np.zeros((NM, NM))
    JtWr = np.zeros(NM)
    for y, Nph, off in frames:
        ph = s.phase(coef_full(est + off))
        m = Nph * inorm(ph).ravel()
        J = Nph * jac(ph)
        w = 1.0 / (m + sig_r**2)
        JtWJ += (J * w) @ J.T
        JtWr += J @ (w * (y - m))
    return JtWJ, JtWr


def lm_fit(est, frames, sig_r, iters=3, tol=2e-3):
    """Levenberg-Marquardt on the Gaussian-approximated Poisson+read likelihood; a step is accepted only if the likelihood improves;
    stops early when the accepted step is below tol (rad rms of the 33-mode change)."""
    lam = 1e-3
    ll = loglik(est, frames, sig_r)
    for _ in range(iters):
        JtWJ, JtWr = normal_eqs(est, frames, sig_r)
        dg = np.diag(np.diag(JtWJ)) + 1e-9 * np.eye(NM)
        accepted = False
        for _try in range(4):
            step = np.linalg.solve(JtWJ + lam * dg, JtWr)
            cand = est + step
            llc = loglik(cand, frames, sig_r)
            if llc > ll:
                est, ll, lam, accepted = cand, llc, max(lam / 3, 1e-6), True
                break
            lam *= 10
        if not accepted or np.sqrt(np.mean(s.phase(coef_full(step))[s.pupil] ** 2)) < tol:
            break
    return est


def loglik(est, frames, sig_r):
    ll = 0.0
    for y, Nph, off in frames:
        m = Nph * inorm(s.phase(coef_full(est + off))).ravel()
        v = m + sig_r**2
        ll += -0.5 * np.sum((y - m) ** 2 / v) - 0.5 * np.sum(np.log(v))
    return ll


def best_start(est_prev, frames, sig_r):
    """Two hypotheses for the residual after applying g*est_prev: the previous even sign was right, (1-g)*est_prev,
    or it was wrong, (1-g)*odd(est_prev) - (1+g)*even(est_prev). The likelihood of the recorded frames decides.
    """
    ev = s.even_mask[EST]
    h_right = (1 - G) * est_prev
    h_wrong = (1 - G) * est_prev * (~ev) - (1 + G) * est_prev * ev
    return h_wrong if loglik(h_wrong, frames, sig_r) > loglik(h_right, frames, sig_r) else h_right


def twin(est):
    c = coef_full(est)
    c = s.twin_coeffs(c)
    return c[EST]


def record(y_mean, sig_r, rng):
    return rng.poisson(np.clip(np.nan_to_num(y_mean), 0, 1e9)) + sig_r * rng.standard_normal(y_mean.shape)


def build_sh():
    n = 7
    pitch = 2.0 / n
    xs = (np.arange(n) - (n - 1) / 2) * pitch
    x = np.arange(s.N) - (s.N - 1) / 2.0
    X, Y = np.meshgrid(x, x)
    Xn, Yn = X / (s.Dpix / 2.0), Y / (s.Dpix / 2.0)
    # gradients of the 31 modes, rad per pupil radius
    gx = np.array([np.gradient(Zj, axis=1) for Zj in Zp_est]) * (s.Dpix / 2.0)
    gy = np.array([np.gradient(Zj, axis=0) for Zj in Zp_est]) * (s.Dpix / 2.0)
    subs = []
    for cy in xs:
        for cx in xs:
            msk = (np.abs(Xn - cx) <= pitch / 2) & (np.abs(Yn - cy) <= pitch / 2)
            frac = (msk & s.pupil).sum() / max(msk.sum(), 1)
            subs.append((cx, cy, msk & s.pupil, frac))
    subs = sorted(subs, key=lambda t: -t[3])[:37]  # the 37 best-illuminated of the 7x7 grid
    M = np.zeros((2 * 37, NM))
    fr = np.zeros(37)
    for i, (cx, cy, msk, frac) in enumerate(subs):
        M[2 * i] = gx[:, msk].mean(1)
        M[2 * i + 1] = gy[:, msk].mean(1)
        fr[i] = frac
    return subs, M, fr / fr.sum()


SUBS, M_SH, FRAC_SH = build_sh()
# px displacement per rad/radius at 1080 nm
PX_PER_RAD = LAM_MAIN / (2 * np.pi) / (D_SH / 2) * F_SH / PX_SH
SIG_SPOT = 0.42 * LAM_SH * F_SH / PITCH / PX_SH  # Gaussian spot sigma, px, at 532 nm
# the same lenslet array observing the 1080 nm main beam
SIG_SPOT_MAIN = 0.42 * LAM_MAIN * F_SH / PITCH / PX_SH
RECON = np.linalg.pinv(M_SH * PX_PER_RAD)  # px slopes -> 33 modes (rad at 1080 nm)
wgrid = np.arange(SH_WIN) - (SH_WIN - 1) / 2.0
WX, WY = np.meshgrid(wgrid, wgrid)


def sh_measure(est_true, Nph, rng, sig_sh=READ_SH, sig_spot=SIG_SPOT):
    slopes_true = (M_SH @ est_true) * PX_PER_RAD
    meas = np.zeros(2 * 37)
    for i in range(37):
        lim = SH_WIN / 2 - 1.0
        # spot leaving the window saturates
        dx = float(np.clip(slopes_true[2 * i], -lim, lim))
        dy = float(np.clip(slopes_true[2 * i + 1], -lim, lim))
        spot = np.exp(-((WX - dx) ** 2 + (WY - dy) ** 2) / (2 * sig_spot**2))
        spot /= max(spot.sum(), 1e-12)
        img = rng.poisson(np.clip(Nph * FRAC_SH[i] * spot, 0, 1e9)) + sig_sh * rng.standard_normal(spot.shape)
        # iteratively weighted center of gravity (weights = expected spot profile about the current estimate),
        # which approaches the maximum-likelihood centroid and limits the read-noise contribution of empty pixels
        tot = img.sum()
        if tot <= 0:
            meas[2 * i] = meas[2 * i + 1] = 0.0
            continue
        ex, ey = (img * WX).sum() / tot, (img * WY).sum() / tot
        for _ in range(3):
            w = np.exp(-((WX - ex) ** 2 + (WY - ey) ** 2) / (2 * sig_spot**2))
            wi = w * img
            tw = wi.sum()
            if tw <= 0:
                break
            ex, ey = (wi * WX).sum() / tw, (wi * WY).sum() / tw
            ex, ey = float(np.clip(ex, -lim, lim)), float(np.clip(ey, -lim, lim))
        meas[2 * i], meas[2 * i + 1] = ex, ey
    return RECON @ meas


def run_loop(method, phi_seq, Nph, rng, shot_only=False):
    """Closed loop over the coefficient vectors phi_seq, one per iteration (all equal for a static aberration)."""
    sig_ff = 0.0 if shot_only else READ_FF
    c = np.zeros(NM)
    rec = []
    est_prev = None
    I0 = None
    y_prev = None
    c_prev = None
    peak_prev = None
    for k in range(K):
        phi_est = phi_seq[k]
        psi = phi_est - c
        rec.append(
            dict(
                k=k,
                strehl=strehl(coef_full(psi)),
                rms=np.sqrt(np.mean(s.phase(coef_full(psi))[s.pupil] ** 2)),
            )
        )
        yk_pre = None
        if FLOW > 0 and method == "image_loop":
            # reversion rule of the running loop: the frame is recorded; if its peak has fallen by more than half since the last frame, the
            # last increment is reverted for the next frame and no estimate is made from this frame, which becomes the reference of the
            # next one with the reverted increment as the known correction difference
            yk_pre = record(Nph * inorm(s.phase(coef_full(psi))).ravel(), sig_ff, rng)
            peak = np.sort(yk_pre)[-9:].mean()
            if peak_prev is not None and peak < 0.5 * peak_prev and c_prev is not None:
                rec[-1]["reverted"] = True
                rec[-1]["sign_ok"] = False
                y_prev = yk_pre
                c_bad = c.copy()
                c = c_prev.copy()
                c_prev = c_bad
                peak_prev = peak
                continue
            peak_prev = peak
        if method == "sh_beacon":
            est = sh_measure(psi, Nph, rng, 0.0 if shot_only else READ_SH)
        # Shack-Hartmann on the 1080 nm main beam (same light as the image-based loop)
        elif method == "sh_main_beam":
            est = sh_measure(psi, Nph, rng, 0.0 if shot_only else READ_SH, SIG_SPOT_MAIN)
        elif method == "image_loop":
            yk = (
                yk_pre
                if yk_pre is not None
                else record(Nph * inorm(s.phase(coef_full(psi))).ravel(), sig_ff, rng)
            )
            if k == 0:
                I0 = yk
                # ideal single-frame magnitude estimator: true residual with noise at the single-frame Cramer-Rao level,
                # even part with a random sign (the sign is undetermined by one frame, Corollary 1)
                JtWJ, _ = normal_eqs(psi, [(yk, Nph, np.zeros(NM))], sig_ff)
                cov = np.linalg.inv(JtWJ + 1e-9 * np.eye(NM))
                cov = 0.5 * (cov + cov.T)
                noise = rng.multivariate_normal(np.zeros(NM), cov, method="eigh")
                nr = np.sqrt(np.mean(s.phase(coef_full(noise))[s.pupil] ** 2))
                if nr > 1.0:
                    noise *= 1.0 / nr
                sgn = 1.0 if rng.random() < 0.5 else -1.0
                est = psi * (~s.even_mask[EST]) + sgn * psi * s.even_mask[EST] + noise
            elif FLOW == 0:
                # current frame; initial frame shows psi + c (static aberration)
                frames = [(yk, Nph, np.zeros(NM)), (I0, Nph, c)]
                init = best_start(est_prev, frames, sig_ff)
                est = lm_fit(init, frames, sig_ff, iters=6 if k == 1 else 3)
            else:
                # running loop: the wavefront has moved between the two frames, so the magnitudes are fitted to the current frame alone
                # (one fit from each sign hypothesis, which lands in one of the two twin basins) and the previous frame, with the
                # last increment as the known correction difference, decides between the two candidates (Note S3)
                # ideal estimator of the running loop: magnitudes and odd part from the current frame at the single-frame Cramér-Rao
                # level (as at k = 0), and the sign of the even part decided between the two candidates by the likelihood of the
                # previous frame with the last increment as the known correction difference (the ideal observer of Note S3)
                ref = [(y_prev, Nph, c - c_prev)]
                JtWJ, _ = normal_eqs(psi, [(yk, Nph, np.zeros(NM))], sig_ff)
                cov = np.linalg.inv(JtWJ + 1e-9 * np.eye(NM))
                cov = 0.5 * (cov + cov.T)
                noise = rng.multivariate_normal(np.zeros(NM), cov, method="eigh")
                nr = np.sqrt(np.mean(s.phase(coef_full(noise))[s.pupil] ** 2))
                if nr > 1.0:
                    noise *= 1.0 / nr
                base = psi + noise
                est = max((base, twin(base)), key=lambda e: loglik(e, ref, sig_ff))
            y_prev = yk
        elif method == "phase_diversity":
            # defocus (Noll j = 4)
            th = np.zeros(NM)
            th[[i for i, (j, n, m) in enumerate(s.modes_info) if (n, m) == (2, 0)][0]] = PD_DEFOCUS
            Nh = Nph * PD_SPLIT
            yf = record(Nh * inorm(s.phase(coef_full(psi))).ravel(), sig_ff, rng)
            yd = record(Nh * inorm(s.phase(coef_full(psi + th))).ravel(), sig_ff, rng)
            frames = [(yf, Nh, np.zeros(NM)), (yd, Nh, th)]
            init = np.zeros(NM) if k == 0 else best_start(est_prev, frames, sig_ff)
            est = lm_fit(init, frames, sig_ff, iters=6 if k == 0 else 3)
        rec[-1]["sign_ok"] = bool(np.dot(est * s.even_mask[EST], psi * s.even_mask[EST]) > 0)
        est_prev = est
        c_prev = c.copy()
        c = c + G * est
    psi = phi_seq[-1] - c
    rec.append(
        dict(k=K, strehl=strehl(coef_full(psi)), rms=np.sqrt(np.mean(s.phase(coef_full(psi))[s.pupil] ** 2)))
    )
    return rec


def cache_name(method, tag, sigma, flow):
    name = "%s_%s_%s" % (method, tag, "static" if flow == 0 else "flow%.2f" % flow)
    return name + ("" if sigma == 0.8 else "_s%.1f" % sigma) + ".pkl"


def steady_strehl(d, k=10):
    """Steady-state Strehl ratio of each run (mean over iterations 8 to 10), shape (photon levels, runs)."""
    return np.array(
        [[np.mean([r[j]["strehl"] for j in range(k - 2, k + 1)]) for r in d["res"][n]] for n in d["N"]]
    )


def draws():
    """TRIALS aberration sequences of K frames. Static (FLOW = 0): one von Kármán draw scaled to SIGMA, repeated.
    Frozen flow (FLOW > 0): a screen at the D/r0 that gives SIGMA on average, translated by 0.314 r0 FLOW per frame
    (Greenwood coherence time), projected on the 33 modes and scaled so that the first frame has SIGMA rad rms.
    """
    out = []
    if FLOW == 0:
        for t in range(TRIALS):
            c = s.turbulence_coeffs(4, seed=700 + t)
            c[TILT] = 0.0
            c = s.scale_to_rms(c, SIGMA)
            out.append([c[EST]] * K)
        return out
    from scipy.ndimage import shift as ndshift

    sf = Simulator(N=512, Dpix=128, nmodes=33)  # fine grid for the screen translation
    DR0 = (SIGMA**2 / 0.134) ** 0.6  # Noll: tilt-removed variance 0.134 (D/r0)^(5/3)
    px = 0.314 * (sf.Dpix / DR0) * FLOW
    for t in range(TRIALS):
        scr = sf.vk_screen(DR0, rng=np.random.default_rng(700 + t))
        seq = [sf.project(ndshift(scr, (0.0, -k * px), order=1, mode="wrap")) for k in range(K)]
        f = SIGMA / sf.rms(sf.phase(seq[0]))
        out.append([(f * c)[EST] for c in seq])
    return out


if __name__ == "__main__" and (len(sys.argv) < 2 or sys.argv[1] not in ("run", "summary")):
    sys.exit(__doc__)


if __name__ == "__main__" and sys.argv[1] == "run":
    args = sys.argv[2:]
    method = args[0]
    tag = args[1] if len(args) > 1 and not args[1].startswith("--") else "read"
    if "--flow" in args:
        FLOW = float(args[args.index("--flow") + 1])
    if "--sigma" in args:
        SIGMA = float(args[args.index("--sigma") + 1])
    shot_only = tag == "shot"
    t0 = time.time()
    res = {}
    PH = draws()
    for iN, Nph in enumerate(NLEVELS):
        # fixed per method and noise model, so every run reproduces
        mseed = {"image_loop": 1, "phase_diversity": 2, "sh_beacon": 3, "sh_main_beam": 4}[method] + (
            50 if shot_only else 0
        )
        res[Nph] = [
            run_loop(method, PH[t], Nph, np.random.default_rng(10000 * iN + 100 * t + mseed), shot_only)
            for t in range(TRIALS)
        ]
        st = np.array([[r[k]["strehl"] for k in range(K + 1)] for r in res[Nph]])
        print(
            "%s N=%.0e: Strehl by k %s | steady %.3f  (%.0fs)"
            % (method, Nph, np.round(st.mean(0), 2), st[:, -3:].mean(), time.time() - t0),
            flush=True,
        )
    pickle.dump(
        dict(method=method, tag=tag, sigma=SIGMA, flow=FLOW, res=res, N=NLEVELS),
        open(os.path.join(CACHE, cache_name(method, tag, SIGMA, FLOW)), "wb"),
    )
    print("done %.0fs" % (time.time() - t0))


if __name__ == "__main__" and sys.argv[1] == "summary":
    import glob

    for f in sorted(glob.glob(os.path.join(CACHE, "*.pkl"))):
        d = pickle.load(open(f, "rb"))
        med = np.median(steady_strehl(d), 1)
        print(os.path.basename(f), " ".join("%.0e: %.2f" % (n, v) for n, v in zip(d["N"], med)))
