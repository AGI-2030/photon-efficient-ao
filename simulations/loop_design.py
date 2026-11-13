"""Design studies of the loop with ideal estimators (Notes S2 and S3, Figs. S1 to S4).

All stages use the parameters of photon_comparison.py: 1080 nm, 8.7 pixels per lambda/D, read noise of
7 electrons, 33 modes (Noll j = 4 to 36) and gain 1/2.

Stages:
    window               Cramer-Rao error of the 33 modes against the camera window (Fig. S2(b))
    probe                photons of the first sign decision against the defocus added to the first command (Fig. S2(a))
    scaling              Monte Carlo checks of the read-noise exponent and of the photon requirement against the
                         change per loop period (Fig. S1(d))
    reversion_rule       frames in which the reversion rule acts and its effect on the Strehl ratio (Note S3)
    boiling              boiling evolution against frozen flow at the same change per frame (Fig. S4(a))
    turbulence_strength  D/r0 scan with the fitting residual in the images (Fig. S3(c))
    loop_period          oracle and ideal estimator against tau_loop/tau_0 at D/r0 = 6 (Fig. S3(a, b));
                         environment variables FLOWS, NPH, TR, K and TAG split the work across processes
    photons              photon scan of the ideal estimator at D/r0 = 6 and tau_loop/tau_0 = 0.3 (Fig. S3(d));
                         environment variables NPHS, TAG, DR0 and FLOW
    scintillation        discriminability with a log-amplitude screen on the pupil (Fig. S4(b))
    tables               loop_period.csv and photons.csv from the results of the two scans above (Fig. S3(a, b, d))

Usage: python -m simulations.loop_design <stage>, one stage per call. Results are written to data/design/.
"""

import os
import sys
import time
import pickle
import numpy as np
from numpy.fft import fft2, fftshift
from scipy.ndimage import shift as ndshift
from aosim import Simulator, data_dir
from simulations import photon_comparison as sp

STAGES = (
    "window",
    "probe",
    "scaling",
    "reversion_rule",
    "boiling",
    "turbulence_strength",
    "loop_period",
    "photons",
    "scintillation",
    "tables",
)
if __name__ == "__main__" and (len(sys.argv) != 2 or sys.argv[1] not in STAGES):
    sys.exit(__doc__)

CACHE = data_dir("design")
s = sp.s
G = sp.G
READ = sp.READ_FF
NM = sp.NM
EV = s.even_mask  # 15 even modes of the 33
C0 = sp.C0
# defocus index in the 33-mode vector
DEF = [i for i, (j, n, m) in enumerate(s.modes_info) if (n, m) == (2, 0)][0]
EC_R = 1.22  # encircled-energy radius, lambda/D (Airy radius)
# diffraction-limited peak of the normalized full-grid intensity (set below)
PDL = None


def save(name, obj):
    pickle.dump(obj, open(os.path.join(CACHE, name + ".pkl"), "wb"))


def load(name):
    return pickle.load(open(os.path.join(CACHE, name + ".pkl"), "rb"))


def I_full(coef):
    """normalized intensity on the full 384 grid for a 33-coefficient vector"""
    If = np.abs(sp.field(s.phase(coef))) ** 2
    return If / If.sum()


def jac_full(coef):
    """d I_full / d a_j, shape (33, N, N)"""
    ph = s.phase(coef)
    Gf = sp.field(ph)
    u = s.A * np.exp(1j * ph)
    tot = (np.abs(Gf) ** 2).sum()
    J = np.empty((NM, s.N, s.N))
    for j in range(NM):
        J[j] = -2.0 * np.imag(np.conj(Gf) * fftshift(fft2(u * s.Z[j]))) / tot
    return J


def window_mask(R_ld):
    h = int(round(R_ld * s.ppl))
    m = np.zeros((s.N, s.N), bool)
    m[C0 - h : C0 + h, C0 - h : C0 + h] = True
    return m


def crb_rms(frames, sig_r, mask):
    """root of the trace of the Cramér-Rao covariance of the 33 coefficients (rad rms of the 33-mode wavefront error) from frames
    [(coef_seen_by_frame, Nph), ...] under Poisson plus Gaussian read noise (Gaussian approximation), pixels in mask only
    """
    JtWJ = np.zeros((NM, NM))
    for coef, Nph in frames:
        m = Nph * I_full(coef)[mask]
        J = Nph * jac_full(coef)[:, mask]
        w = 1.0 / (m + sig_r**2)
        JtWJ += (J * w) @ J.T
    cov = np.linalg.inv(JtWJ + 1e-12 * np.eye(NM))
    return float(np.sqrt(np.trace(cov)))


def dprime2(Ia, Ib, Nph, sig_r, mask):
    """squared discriminability of two frames with means Nph*Ia and Nph*Ib under Poisson plus read noise (Gaussian approximation)"""
    a, b = Nph * Ia[mask], Nph * Ib[mask]
    v = 0.5 * (a + b) + sig_r**2
    return float((((a - b) ** 2) / v).sum())


def nmin(Ia, Ib, sig_r, mask, dmin=3.0):
    """photon number at which d' reaches dmin (bisection; d'^2 is monotone in N)"""
    lo, hi = 1.0, 1e9
    if dprime2(Ia, Ib, hi, sig_r, mask) < dmin**2:
        return np.inf
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if dprime2(Ia, Ib, mid, sig_r, mask) >= dmin**2:
            hi = mid
        else:
            lo = mid
    return hi


def ec_of(coef, R=EC_R):
    return s.encircled(I_full(coef), R)


PDL = float(
    (np.abs(sp.field(np.zeros((s.N, s.N)))) ** 2 / (np.abs(sp.field(np.zeros((s.N, s.N)))) ** 2).sum())[
        C0, C0
    ]
)


def aberrations(n, rms, seed0=700):
    out = []
    for t in range(n):
        c = s.turbulence_coeffs(4, seed=seed0 + t)
        out.append(s.scale_to_rms(c, rms))
    return out


if __name__ == "__main__" and sys.argv[1] == "window":
    t0 = time.time()
    RW = [2, 3, 4, 5, 7, 10, 15, 20]
    NPH = [1e3, 1e4, 1e5]
    RMS = [0.8, 0.3]
    ab = aberrations(8, 0.8)
    out = {}
    for r in RMS:
        for t, phi in enumerate(ab):
            psi = s.scale_to_rms(phi, r)
            d = psi.copy()
            Icur = I_full(psi)
            Iref_true = I_full(psi + d)
            Iref_twin = I_full(s.twin_coeffs(psi) + d)
            for R in RW:
                mask = window_mask(R)
                rec = dict(npix=int(mask.sum()), frac_in=float(Icur[mask].sum()))
                for Nph in NPH:
                    rec["one_read_%d" % int(np.log10(Nph))] = crb_rms([(psi, Nph)], READ, mask)
                    rec["one_shot_%d" % int(np.log10(Nph))] = crb_rms([(psi, Nph)], 0.0, mask)
                    rec["two_read_%d" % int(np.log10(Nph))] = crb_rms(
                        [(psi, Nph), (psi + d, Nph)], READ, mask
                    )
                    rec["nmin_read_%d" % int(np.log10(Nph))] = nmin(Iref_true, Iref_twin, READ, mask)
                rec["nmin_shot"] = nmin(Iref_true, Iref_twin, 0.0, mask)
                out[(r, t, R)] = rec
        print("rms %.1f done (%.0fs)" % (r, time.time() - t0), flush=True)
    save("window", dict(out=out, RW=RW, NPH=NPH, RMS=RMS))
    print("saved window (%.0fs)" % (time.time() - t0))

if __name__ == "__main__" and sys.argv[1] == "probe":
    t0 = time.time()
    AP = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    NPH = [1e3, 1e4]
    RMS = [0.8, 0.4]
    CASES = ["right", "wrong", "hedged"]
    mask = window_mask(10.0)
    ab = aberrations(16, 0.8)
    out = {}
    for r in RMS:
        for t, phi in enumerate(ab):
            phi = s.scale_to_rms(phi, r)
            rng = np.random.default_rng(900 + t)
            for Nph in NPH:
                # first estimate: truth with noise at the single-frame Cramér-Rao level (read noise), even part with the stated sign
                JtWJ = np.zeros((NM, NM))
                m = Nph * I_full(phi)[mask]
                J = Nph * jac_full(phi)[:, mask]
                w = 1.0 / (m + READ**2)
                cov = np.linalg.inv((J * w) @ J.T + 1e-9 * np.eye(NM))
                cov = 0.5 * (cov + cov.T)
                noise = rng.multivariate_normal(np.zeros(NM), cov, method="eigh")
                nr = s.rms(s.phase(noise))
                if nr > 1.0:
                    noise *= 1.0 / nr
                for case in CASES:
                    sg = {"right": 1.0, "wrong": -1.0, "hedged": 0.0}[case]
                    est = phi * (~EV) + sg * phi * EV + noise
                    for ap in AP:
                        c1 = G * est
                        c1[DEF] += ap
                        psi1 = phi - c1
                        # reference images of truth and residual twin
                        Ia = I_full(psi1 + c1)
                        Ib = I_full(s.twin_coeffs(psi1) + c1)
                        out[(r, t, Nph, case, ap)] = dict(
                            nmin_shot=nmin(Ia, Ib, 0.0, mask),
                            nmin_read=nmin(Ia, Ib, READ, mask),
                            strehl1=float(I_full(psi1)[C0, C0] / I_full(np.zeros(NM))[C0, C0]),
                            rms1=s.rms(s.phase(psi1)),
                            de=s.rms(s.phase(c1 * EV)),
                        )
            print("rms %.1f trial %d (%.0fs)" % (r, t, time.time() - t0), flush=True)
    save("probe", dict(out=out, AP=AP, NPH=NPH, RMS=RMS, CASES=CASES))
    print("saved probe (%.0fs)" % (time.time() - t0))

if __name__ == "__main__" and sys.argv[1] == "scaling":
    t0 = time.time()
    mask = window_mask(10.0)
    rng = np.random.default_rng(5)
    # (i) read-noise exponent: N_min against the product |psi_e| |d_e| for fixed shapes, shot noise and read noise, with a Monte Carlo
    # likelihood-ratio test at the predicted N_min (accuracy should be about 0.93 for d' = 3)
    base = aberrations(1, 0.8)[0]
    eshape = base * EV
    eshape /= s.rms(s.phase(eshape))
    oshape = base * (~EV)
    oshape /= s.rms(s.phase(oshape))
    pshape = np.zeros(NM)
    pshape[DEF] = 1.0
    A = [0.05, 0.1, 0.2, 0.4]
    B = [0.05, 0.1, 0.2, 0.4]
    sc = {}
    for a in A:
        for b in B:
            psi = 0.3 * oshape + a * eshape
            d = b * pshape
            Ia = I_full(psi + d)
            Ib = I_full(s.twin_coeffs(psi) + d)
            rec = dict(prod=a * b, nmin_shot=nmin(Ia, Ib, 0.0, mask), nmin_read=nmin(Ia, Ib, READ, mask))
            # Monte Carlo at the predicted read-noise N_min: Gaussian-approximation LRT between the two hypotheses
            N = rec["nmin_read"]
            if np.isfinite(N) and N < 1e8:
                mA, mB = N * Ia[mask], N * Ib[mask]
                v = 0.5 * (mA + mB) + READ**2
                correct = 0
                T = 400
                for k in range(T):
                    fromA = k < T // 2
                    mu = mA if fromA else mB
                    y = rng.poisson(mu) + READ * rng.standard_normal(mu.shape)
                    llr = -0.5 * np.sum((y - mA) ** 2 / v) + 0.5 * np.sum((y - mB) ** 2 / v)
                    correct += int((llr > 0) == fromA)
                rec["mc_acc_read"] = correct / T
            sc[(a, b)] = rec
            print(
                "a %.2f b %.2f: Nmin shot %.3g read %.3g acc %.2f"
                % (a, b, rec["nmin_shot"], rec["nmin_read"], rec.get("mc_acc_read", np.nan)),
                flush=True,
            )
    # (ii) inverse fourth power: steady state of an oracle loop (exact magnitudes, sign known, gain 1/2) on frozen-flow sequences;
    # N_min of the sign decision at the loop's own residual and probe against the per-frame change
    from simulations import reference_frames as rf

    RAT = [0.03, 0.1, 0.3, 1.0]
    ff = {}
    for ratio in RAT:
        vals = []
        for t in range(8):
            coefs, px = rf.sequence(ratio, 500 + t)
            c = np.zeros(33)
            cprev = None
            psi_prev = None
            for k in range(rf.K):
                psi = coefs[k] - c
                if k >= 4 and cprev is not None:
                    d = c - cprev
                    Ia = I_full(psi + d)
                    Ib = I_full(s.twin_coeffs(psi) + d)
                    vals.append(
                        dict(
                            k=k,
                            nmin_shot=nmin(Ia, Ib, 0.0, mask),
                            nmin_read=nmin(Ia, Ib, READ, mask),
                            delta=s.rms(s.phase(coefs[k] - coefs[k - 1])),
                            rms=s.rms(s.phase(psi)),
                            de=s.rms(s.phase(d * EV)),
                        )
                    )
                cprev = c.copy()
                c = c + G * psi  # oracle: exact residual, gain 1/2
        ff[ratio] = vals
        print(
            "ratio %.2f: median delta %.3f, Nmin shot %.3g read %.3g (%.0fs)"
            % (
                ratio,
                np.median([v["delta"] for v in vals]),
                np.median([v["nmin_shot"] for v in vals]),
                np.median([v["nmin_read"] for v in vals]),
                time.time() - t0,
            ),
            flush=True,
        )
    save("scaling", dict(sc=sc, ff=ff, A=A, B=B, RAT=RAT))
    print("saved scaling (%.0fs)" % (time.time() - t0))


def ideal_running_loop(phi_seq, Nph, rng, rule="peak", shot_only=False, K=None, ec_margin=0.02):
    """Ideal running-loop estimator of photon_comparison.run_loop (previous frame as reference, last increment as probe) with a
    reversion rule: 'none'; 'peak' (revert the last increment when the peak of the recorded frame falls below half of the previous
    one); 'ec' (revert when the encircled energy of the recorded frame, within 1.22 lambda/D of the axis, falls below that of the
    previous frame by more than ec_margin). Returns one record per frame with Strehl, EC, rms, sign_ok, reverted.
    """
    sig_ff = 0.0 if shot_only else READ
    K = len(phi_seq) if K is None else K
    mask = window_mask(10.0)
    rr = (np.hypot(s.U, s.V) <= EC_R)[mask]
    r3 = window_mask(3.0)[mask]
    c = np.zeros(NM)
    rec = []
    y_prev = None
    c_prev = None
    peak_prev = None
    ec_prev = None
    for k in range(K):
        psi = phi_seq[k] - c
        Itrue = I_full(psi)
        r = dict(
            k=k,
            strehl=float(Itrue[C0, C0] / PDL),
            ec=s.encircled(Itrue, EC_R),
            rms=s.rms(s.phase(psi)),
            reverted=False,
            sign_ok=None,
        )
        yk = sp.record(Nph * Itrue[mask], sig_ff, rng)
        # EC of the recorded frame, total taken in a ±3 lambda/D box
        peak = np.sort(yk)[-9:].mean()
        ec_meas = yk[rr].sum() / max(yk[r3].sum(), 1e-9)
        bad = False
        if c_prev is not None and k >= 1:
            if rule == "peak" and peak_prev is not None and peak < 0.5 * peak_prev:
                bad = True
            if rule == "ec" and ec_prev is not None and ec_meas < ec_prev - ec_margin:
                bad = True
        peak_prev, ec_prev = peak, ec_meas
        if bad:
            r["reverted"] = True
            rec.append(r)
            y_prev = yk
            c_bad = c.copy()
            c = c_prev.copy()
            c_prev = c_bad
            continue
        if k == 0 or y_prev is None:
            JtWJ, _ = sp.normal_eqs(psi, [(yk, Nph, np.zeros(NM))], sig_ff)
            cov = np.linalg.inv(JtWJ + 1e-9 * np.eye(NM))
            cov = 0.5 * (cov + cov.T)
            noise = rng.multivariate_normal(np.zeros(NM), cov, method="eigh")
            nr = s.rms(s.phase(noise))
            if nr > 1.0:
                noise *= 1.0 / nr
            sgn = 1.0 if rng.random() < 0.5 else -1.0
            est = psi * (~EV) + sgn * psi * EV + noise
        else:
            ref = [(y_prev, Nph, c - c_prev)]
            JtWJ, _ = sp.normal_eqs(psi, [(yk, Nph, np.zeros(NM))], sig_ff)
            cov = np.linalg.inv(JtWJ + 1e-9 * np.eye(NM))
            cov = 0.5 * (cov + cov.T)
            noise = rng.multivariate_normal(np.zeros(NM), cov, method="eigh")
            nr = s.rms(s.phase(noise))
            if nr > 1.0:
                noise *= 1.0 / nr
            base = psi + noise
            est = max((base, sp.twin(base)), key=lambda e: sp.loglik(e, ref, sig_ff))
        r["sign_ok"] = bool(np.dot(est * EV, psi * EV) > 0)
        rec.append(r)
        y_prev = yk
        c_prev = c.copy()
        c = c + G * est
    return rec


def tilt_removed_rms(sf, s35, sk):
    p = (sk - s35.phase(s35.project(sk) * np.r_[1.0, 1.0, np.zeros(33)])) * sf.pupil
    p[sf.pupil] -= p[sf.pupil].mean()
    return sf.rms(p)


def flow_sequences(DR0, flow, K, trials, seed0=700, scale_to=None):
    """frozen-flow sequences of 33-coefficient vectors at D/r0 = DR0 translated by 0.314 r0 flow per frame; also the high-order
    fitting residual (screen minus its projection on tip, tilt and the 33 modes) rms and the EC bound of that residual
    """
    sf = Simulator(N=512, Dpix=128, nmodes=33)
    s35 = Simulator(N=512, Dpix=128, nmodes=35, jstart=2)
    px = 0.314 * (sf.Dpix / DR0) * flow
    out = []
    for t in range(trials):
        scr = sf.vk_screen(DR0, rng=np.random.default_rng(seed0 + t))
        seq = []
        fit = []
        for k in range(K):
            sk = ndshift(scr, (0.0, -k * px), order=1, mode="wrap")
            c = sf.project(sk)
            seq.append(c)
            if k == 0:
                hi = (sk - s35.phase(s35.project(sk))) * sf.pupil
                hi[sf.pupil] -= hi[sf.pupil].mean()
                fit.append(
                    dict(
                        fit_rms=sf.rms(hi),
                        ec_bound=sf.encircled(sf.intensity(hi), EC_R),
                        strehl_bound=float(
                            sf.intensity(hi)[sf.N // 2, sf.N // 2]
                            / sf.intensity(np.zeros((sf.N, sf.N)))[sf.N // 2, sf.N // 2]
                        ),
                        rms33=sf.rms(sf.phase(c)),
                        rms_tiltrem=tilt_removed_rms(sf, s35, sk),
                    )
                )
        if scale_to is not None:
            f = scale_to / sf.rms(sf.phase(seq[0]))
            seq = [f * c for c in seq]
        out.append((seq, fit[0]))
    return out


if __name__ == "__main__" and sys.argv[1] == "reversion_rule":
    t0 = time.time()
    K = 30
    TR = 12
    RAT = [0.1, 0.3, 1.0]
    NPH = [1e4, 1e5]
    RULES = ["none", "peak", "ec"]
    out = {}
    for ratio in RAT:
        seqs = flow_sequences(6.0, ratio, K, TR, scale_to=0.8)
        for Nph in NPH:
            for rule in RULES:
                R = [
                    ideal_running_loop(
                        seqs[t][0], Nph, np.random.default_rng(3000 + 100 * t + int(ratio * 10)), rule=rule
                    )
                    for t in range(TR)
                ]
                out[(ratio, Nph, rule)] = R
                st = np.array([[r["strehl"] for r in rr] for rr in R])
                rev = np.array([[r["reverted"] for r in rr] for rr in R])
                wrong = np.array([[(r["sign_ok"] is False) for r in rr] for rr in R])
                print(
                    "ratio %.1f N %.0e %-5s: steady Strehl %.3f, reverted %.3f, wrong sign %.3f (%.0fs)"
                    % (
                        ratio,
                        Nph,
                        rule,
                        st[:, 10:].mean(),
                        rev[:, 2:].mean(),
                        wrong[:, 2:].mean(),
                        time.time() - t0,
                    ),
                    flush=True,
                )
        save("reversion_rule", dict(out=out, RAT=RAT, NPH=NPH, RULES=RULES, K=K, TR=TR))
    print("saved reversion_rule (%.0fs)" % (time.time() - t0))

if __name__ == "__main__" and sys.argv[1] == "boiling":
    from simulations import reference_frames as rf

    t0 = time.time()
    RAT = [0.1, 0.3, 1.0]
    TR = 20
    METH = ["oracle", "prev", "I0"]
    # covariance of the 33 coefficients at D/r0 = 6 from 300 screens
    C = np.array(
        [rf.s.project(rf.s.vk_screen(rf.DR0, rng=np.random.default_rng(20000 + i))) for i in range(300)]
    )
    Sigma = np.cov(C.T)
    L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(33))
    trS = np.trace(Sigma)
    out = {}
    for ratio in RAT:
        seqs = [rf.sequence(ratio, 500 + t)[0] for t in range(TR)]
        delta2 = np.mean([np.mean([np.sum((sq[k] - sq[k - 1]) ** 2) for k in range(1, rf.K)]) for sq in seqs])
        rho = 1.0 - delta2 / (2 * trS)
        boils = []
        for t in range(TR):
            rng = np.random.default_rng(600 + t)
            c = L @ rng.standard_normal(33)
            sq = [c]
            for k in range(1, rf.K):
                c = rho * c + np.sqrt(1 - rho**2) * (L @ rng.standard_normal(33))
                sq.append(c)
            boils.append(sq)
        d2b = np.mean([np.mean([np.sum((sq[k] - sq[k - 1]) ** 2) for k in range(1, rf.K)]) for sq in boils])
        for model, S in (("frozen", seqs), ("boiling", boils)):
            for m in METH:
                R = [rf.run(S[t], m, np.random.default_rng(8000 + 100 * t)) for t in range(TR)]
                st = np.array([[r[k]["strehl"] for k in range(rf.K)] for r in R])
                pc = np.array([[r[k]["correct"] for k in range(rf.K)] for r in R])
                out[(ratio, model, m)] = dict(
                    steady=float(st[:, -3:].mean()),
                    steady_sd=float(st[:, -3:].mean(1).std()),
                    pc=float(pc[:, 1:].mean()),
                    st_k=st.mean(0).tolist(),
                )
                print(
                    "ratio %.1f %-8s %-7s steady %.3f P(correct) %.2f  [delta2 frozen %.3f boiling %.3f rho %.3f] (%.0fs)"
                    % (
                        ratio,
                        model,
                        m,
                        out[(ratio, model, m)]["steady"],
                        out[(ratio, model, m)]["pc"],
                        delta2,
                        d2b,
                        rho,
                        time.time() - t0,
                    ),
                    flush=True,
                )
        save("boiling", dict(out=out, RAT=RAT, METH=METH))
    print("saved boiling (%.0fs)" % (time.time() - t0))


class Model:
    """far-field model on a finer pupil grid (88 pixels across the pupil, 8.73 pixels per lambda/D as in the laboratory) so that the
    fitting residual beyond the 33 modes is resolved; images are formed from a full phase map (33-mode part plus high-order residual)
    while the estimator's forward model uses the 33 modes only"""

    def __init__(self, N=768, Dpix=88, roi_ld=10.0):
        self.s = Simulator(N=N, Dpix=Dpix, nmodes=33)
        self.s35 = Simulator(N=N, Dpix=Dpix, nmodes=35, jstart=2)
        self.N = N
        self.C0 = N // 2
        self.h = int(round(roi_ld * self.s.ppl))
        self.sl = slice(self.C0 - self.h, self.C0 + self.h)
        self.PDL = float(self.I(np.zeros((N, N)))[self.C0, self.C0])
        self.Rroi = np.hypot(self.s.U, self.s.V)[self.sl, self.sl].ravel()

    def I(self, ph):
        If = np.abs(fftshift(fft2(self.s.A * np.exp(1j * ph)))) ** 2
        return If / If.sum()

    def roi(self, img):
        return img[self.sl, self.sl].ravel()

    def jac_roi(self, ph):
        Gf = fftshift(fft2(self.s.A * np.exp(1j * ph)))
        u = self.s.A * np.exp(1j * ph)
        tot = (np.abs(Gf) ** 2).sum()
        return np.array(
            [self.roi(-2.0 * np.imag(np.conj(Gf) * fftshift(fft2(u * self.s.Z[j]))) / tot) for j in range(NM)]
        )

    def normal_eqs(self, est, frames, sig_r):
        JtWJ = np.zeros((NM, NM))
        for y, Nph, off in frames:
            ph = self.s.phase(est + off)
            m = Nph * self.roi(self.I(ph))
            J = Nph * self.jac_roi(ph)
            w = 1.0 / (m + sig_r**2)
            JtWJ += (J * w) @ J.T
        return JtWJ

    def loglik(self, est, frames, sig_r):
        ll = 0.0
        for y, Nph, off in frames:
            m = Nph * self.roi(self.I(self.s.phase(est + off)))
            v = m + sig_r**2
            ll += -0.5 * np.sum((y - m) ** 2 / v) - 0.5 * np.sum(np.log(v))
        return ll

    def sequences(self, DR0, flow, K, trials, seed0=700):
        """frozen-flow frames: 33-mode coefficients (tilt removed by the steering mirror) and the high-order residual map of each frame"""
        px = 0.314 * (self.s.Dpix / DR0) * flow
        out = []
        for t in range(trials):
            scr = self.s.vk_screen(DR0, rng=np.random.default_rng(seed0 + t))
            coefs, his, info = [], [], None
            for k in range(K):
                sk = ndshift(scr, (0.0, -k * px), order=1, mode="wrap")
                c35 = self.s35.project(sk)
                hi = (sk - self.s35.phase(c35)) * self.s.pupil
                hi[self.s.pupil] -= hi[self.s.pupil].mean()
                coefs.append(c35[2:].copy())
                his.append(hi)
                if k == 0:
                    notilt = (sk - self.s35.phase(c35 * np.r_[1.0, 1.0, np.zeros(33)])) * self.s.pupil
                    notilt[self.s.pupil] -= notilt[self.s.pupil].mean()
                    Ihi = self.I(hi)
                    info = dict(
                        fit_rms=self.s.rms(hi),
                        rms33=self.s.rms(self.s.phase(c35[2:])),
                        rms_tiltrem=self.s.rms(notilt),
                        ec_bound=self.s.encircled(Ihi, EC_R),
                        strehl_bound=float(Ihi[self.C0, self.C0] / self.PDL),
                    )
            out.append((coefs, his, info))
        return out


def loop_hi(M, coefs, his, Nph, rng, rule="peak", shot_only=False, ec_margin=0.02):
    """ideal running-loop estimator (as ideal_running_loop) with the high-order residual present in every image but absent from the model"""
    sig_ff = 0.0 if shot_only else READ
    K = len(coefs)
    rr = M.Rroi <= EC_R
    r3 = np.abs(M.s.U[M.sl, M.sl].ravel()) <= 3.0
    r3 &= np.abs(M.s.V[M.sl, M.sl].ravel()) <= 3.0
    c = np.zeros(NM)
    rec = []
    y_prev = None
    c_prev = None
    peak_prev = None
    ec_prev = None
    for k in range(K):
        psi = coefs[k] - c
        ph_true = M.s.phase(psi) + his[k]
        Itrue = M.I(ph_true)
        r = dict(
            k=k,
            strehl=float(Itrue[M.C0, M.C0] / M.PDL),
            ec=M.s.encircled(Itrue, EC_R),
            rms=M.s.rms(ph_true),
            rms33=M.s.rms(M.s.phase(psi)),
            reverted=False,
            sign_ok=None,
        )
        yk = sp.record(Nph * M.roi(Itrue), sig_ff, rng)
        peak = np.sort(yk)[-9:].mean()
        ec_meas = yk[rr].sum() / max(yk[r3].sum(), 1e-9)
        bad = False
        if c_prev is not None and k >= 1:
            if rule == "peak" and peak_prev is not None and peak < 0.5 * peak_prev:
                bad = True
            if rule == "ec" and ec_prev is not None and ec_meas < ec_prev - ec_margin:
                bad = True
        peak_prev, ec_prev = peak, ec_meas
        if bad:
            r["reverted"] = True
            rec.append(r)
            y_prev = yk
            c_bad = c.copy()
            c = c_prev.copy()
            c_prev = c_bad
            continue
        JtWJ = M.normal_eqs(psi, [(yk, Nph, np.zeros(NM))], sig_ff)
        cov = np.linalg.inv(JtWJ + 1e-9 * np.eye(NM))
        cov = 0.5 * (cov + cov.T)
        noise = rng.multivariate_normal(np.zeros(NM), cov, method="eigh")
        nr = M.s.rms(M.s.phase(noise))
        if nr > 1.0:
            noise *= 1.0 / nr
        if k == 0 or y_prev is None:
            sgn = 1.0 if rng.random() < 0.5 else -1.0
            est = psi * (~EV) + sgn * psi * EV + noise
        else:
            base = psi + noise
            ref = [(y_prev, Nph, c - c_prev)]
            est = max((base, M.s.twin_coeffs(base)), key=lambda e: M.loglik(e, ref, sig_ff))
        r["sign_ok"] = bool(np.dot(est * EV, psi * EV) > 0)
        # per-mode metric of the main text (Methods, Metrics)
        big = EV & (np.abs(psi) > 0.05)
        r["sign_frac"] = float(np.mean(np.sign(est[big]) == np.sign(psi[big]))) if big.any() else None
        r["sign_frac_w"] = (
            float(np.sum((psi[big] ** 2) * (np.sign(est[big]) == np.sign(psi[big]))) / np.sum(psi[big] ** 2))
            if big.any()
            else None
        )
        r["err33"] = float(M.s.rms(M.s.phase(est - psi)))
        rec.append(r)
        y_prev = yk
        c_prev = c.copy()
        c = c + G * est
    return rec


if __name__ == "__main__" and sys.argv[1] == "turbulence_strength":
    t0 = time.time()
    K = 10
    TR = 8
    FLOW = 0.3
    DR0S = [1, 2, 3, 4, 6, 8, 12, 16]
    CONF = [(1e4, False), (1e5, False), (1e4, True)]
    M = Model()
    out = {}
    print("model ready: ppl %.2f, ROI %d px (%.0fs)" % (M.s.ppl, 2 * M.h, time.time() - t0), flush=True)
    for DR0 in DR0S:
        seqs = M.sequences(float(DR0), FLOW, K, TR)
        for Nph, shot in CONF:
            R = [
                loop_hi(
                    M,
                    seqs[t][0],
                    seqs[t][1],
                    Nph,
                    np.random.default_rng(4000 + 100 * t + DR0),
                    rule="peak",
                    shot_only=shot,
                )
                for t in range(TR)
            ]
            out[(DR0, Nph, shot)] = dict(
                res=R,
                info=[sq[2] for sq in seqs],
                delta=[
                    np.mean([M.s.rms(M.s.phase(sq[0][k] - sq[0][k - 1])) for k in range(1, K)]) for sq in seqs
                ],
            )
            st = np.array([[r["strehl"] for r in rr] for rr in R])
            ec = np.array([[r["ec"] for r in rr] for rr in R])
            ok = np.array([[(r["sign_ok"] is True) for r in rr] for rr in R])
            inf = out[(DR0, Nph, shot)]["info"]
            print(
                "D/r0 %2d N %.0e shot %d: Strehl unc %.2f steady %.3f (bound %.2f), EC unc %.2f steady %.3f (bound %.2f), fit rms %.2f, rms33 %.2f, sign ok %.2f (%.0fs)"
                % (
                    DR0,
                    Nph,
                    shot,
                    st[:, 0].mean(),
                    st[:, 6:].mean(),
                    np.mean([f["strehl_bound"] for f in inf]),
                    ec[:, 0].mean(),
                    ec[:, 6:].mean(),
                    np.mean([f["ec_bound"] for f in inf]),
                    np.mean([f["fit_rms"] for f in inf]),
                    np.mean([f["rms33"] for f in inf]),
                    ok[:, 1:].mean(),
                    time.time() - t0,
                ),
                flush=True,
            )
        save("turbulence_strength", dict(out=out, DR0S=DR0S, CONF=CONF, K=K, TR=TR, FLOW=FLOW))
    print("saved turbulence_strength (%.0fs)" % (time.time() - t0))


def oracle_hi(M, coefs, his, g=None):
    """Oracle given the 33 coefficients of every frame; limited by servo lag (gain g, one-frame delay) and by the fitting residual."""
    g = G if g is None else g
    c = np.zeros(NM)
    rec = []
    for k in range(len(coefs)):
        psi = coefs[k] - c
        ph_true = M.s.phase(psi) + his[k]
        Itrue = M.I(ph_true)
        rec.append(
            dict(
                k=k,
                strehl=float(Itrue[M.C0, M.C0] / M.PDL),
                ec=M.s.encircled(Itrue, EC_R),
                rms=M.s.rms(ph_true),
                rms33=M.s.rms(M.s.phase(psi)),
            )
        )
        c = c + g * psi
    return rec


if __name__ == "__main__" and sys.argv[1] == "loop_period":
    t0 = time.time()
    DR0 = float(os.environ.get("DR0", "6"))
    FLOWS = [float(x) for x in os.environ.get("FLOWS", "0,0.03,0.1,0.2,0.3,0.5,0.7,1.0").split(",")]
    NPH = float(os.environ.get("NPH", "1e4"))
    TR = int(os.environ.get("TR", "16"))
    K = int(os.environ.get("K", "16"))
    TAG = os.environ.get("TAG", "a")
    M = Model()
    out = {}
    print(
        "model ready (%.0fs); flows %s, N %.0e, %d sequences, %d frames"
        % (time.time() - t0, FLOWS, NPH, TR, K),
        flush=True,
    )
    for flow in FLOWS:
        seqs = M.sequences(DR0, flow, K, TR)
        orc = [oracle_hi(M, sq[0], sq[1]) for sq in seqs]
        est = [
            loop_hi(
                M,
                seqs[t][0],
                seqs[t][1],
                NPH,
                np.random.default_rng(5000 + 100 * t + int(1000 * flow)),
                rule="none",
            )
            for t in range(TR)
        ]
        out[flow] = dict(
            oracle=orc,
            ideal=est,
            info=[sq[2] for sq in seqs],
            delta=[
                np.mean([M.s.rms(M.s.phase(sq[0][k] - sq[0][k - 1])) for k in range(1, K)]) for sq in seqs
            ],
        )
        so = np.array([[r["strehl"] for r in rr] for rr in orc])
        se = np.array([[r["strehl"] for r in rr] for rr in est])
        eo = np.array([[r["ec"] for r in rr] for rr in orc])
        ee = np.array([[r["ec"] for r in rr] for rr in est])
        ok = np.array([[(r["sign_ok"] is True) for r in rr] for rr in est])
        h = K // 2
        sf = np.array([[np.nan if r["sign_frac"] is None else r["sign_frac"] for r in rr] for rr in est])
        print(
            "flow %.2f: oracle Strehl %.3f EC %.3f | ideal Strehl %.3f EC %.3f sign ok %.2f per-mode %.2f (k=1 %.2f) | change/frame %.3f rad (%.0fs)"
            % (
                flow,
                so[:, h:].mean(),
                eo[:, h:].mean(),
                se[:, h:].mean(),
                ee[:, h:].mean(),
                ok[:, 1:].mean(),
                np.nanmean(sf[:, h:]),
                np.nanmean(sf[:, 1]),
                np.mean(out[flow]["delta"]),
                time.time() - t0,
            ),
            flush=True,
        )
        save("loop_period_" + TAG, dict(out=out, DR0=DR0, NPH=NPH, TR=TR, K=K))
    print("saved loop_period_%s (%.0fs)" % (TAG, time.time() - t0))

if __name__ == "__main__" and sys.argv[1] == "photons":
    t0 = time.time()
    DR0 = float(os.environ.get("DR0", "6"))
    FLOW = float(os.environ.get("FLOW", "0.3"))
    NPHS = [float(x) for x in os.environ.get("NPHS", "1e3,2e3,5e3,1e4,3e4,1e5,1e6").split(",")]
    TR = int(os.environ.get("TR", "16"))
    K = int(os.environ.get("K", "16"))
    TAG = os.environ.get("TAG", "a")
    M = Model()
    out = {}
    seqs = M.sequences(DR0, FLOW, K, TR)
    orc = [oracle_hi(M, sq[0], sq[1]) for sq in seqs]
    print(
        "model ready (%.0fs); photons %s, flow %.2f, %d sequences, %d frames"
        % (time.time() - t0, NPHS, FLOW, TR, K),
        flush=True,
    )
    for Nph in NPHS:
        est = [
            loop_hi(
                M,
                seqs[t][0],
                seqs[t][1],
                Nph,
                np.random.default_rng(6000 + 100 * t + int(np.log10(Nph) * 10)),
                rule="none",
            )
            for t in range(TR)
        ]
        out[Nph] = dict(oracle=orc, ideal=est, info=[sq[2] for sq in seqs])
        se = np.array([[r["strehl"] for r in rr] for rr in est])
        ee = np.array([[r["ec"] for r in rr] for rr in est])
        sf = np.array([[np.nan if r["sign_frac"] is None else r["sign_frac"] for r in rr] for rr in est])
        h = K // 2
        print(
            "N %.0e: ideal Strehl %.3f EC %.3f | per-mode sign k=1 %.2f steady %.2f | err33 %.2f rad (%.0fs)"
            % (
                Nph,
                se[:, h:].mean(),
                ee[:, h:].mean(),
                np.nanmean(sf[:, 1]),
                np.nanmean(sf[:, h:]),
                np.mean([[r["err33"] for r in rr[h:]] for rr in est]),
                time.time() - t0,
            ),
            flush=True,
        )
        save("photons_" + TAG, dict(out=out, DR0=DR0, FLOW=FLOW, TR=TR, K=K))
    print("saved photons_%s (%.0fs)" % (TAG, time.time() - t0))

if __name__ == "__main__" and sys.argv[1] == "scintillation":
    t0 = time.time()
    SIG = [0.0, 0.01, 0.03, 0.1, 0.2, 0.4]
    out = {}
    mask = window_mask(10.0)
    ab = aberrations(8, 0.8)

    def amp_screen(sig2, rng):
        """log-amplitude screen with a Kolmogorov-like spectrum, variance sig2 over the pupil"""
        chi = s.vk_screen(4.0, rng=rng)
        chi = chi * s.pupil
        chi[s.pupil] -= chi[s.pupil].mean()
        chi *= np.sqrt(sig2) / s.rms(chi)
        return np.exp(chi) * s.pupil

    def I_amp(A, ph):
        If = np.abs(fftshift(fft2(A * np.exp(1j * ph)))) ** 2
        return If / If.sum()

    for sig2 in SIG:
        for t, phi in enumerate(ab):
            rng = np.random.default_rng(1500 + t)
            A = amp_screen(sig2, rng) if sig2 > 0 else s.A
            for r in (0.8, 0.4):
                # static loop after the first correction, correct sign
                phi_r = s.scale_to_rms(phi, r)
                psi = 0.5 * phi_r
                d = 0.5 * phi_r
                ph_t = s.phase(phi_r)
                ph_T = s.phase(s.twin_coeffs(phi_r))
                # one frame: truth against global twin
                F_single = s.F_shot(I_amp(A, ph_t) * mask, I_amp(A, ph_T) * mask)
                # reference frame: truth against residual twin
                F_loop = s.F_shot(
                    I_amp(A, s.phase(psi + d)) * mask, I_amp(A, s.phase(s.twin_coeffs(psi) + d)) * mask
                )
                # modelling error: image predicted with a uniform pupil against the actual image with the amplitude screen, as a fraction of the loop separation
                mis = s.F_shot(I_amp(A, s.phase(phi_r)) * mask, I_amp(s.A, s.phase(phi_r)) * mask)
                out[(sig2, t, r)] = dict(
                    F_single=F_single,
                    F_loop=F_loop,
                    F_model=mis,
                    nmin_single=9.0 / max(F_single, 1e-30),
                    nmin_loop=9.0 / F_loop,
                )
        print(
            "sigma_chi^2 %.2f: single %.2e loop %.2e model mismatch %.2e (%.0fs)"
            % (
                sig2,
                np.median([out[(sig2, t, 0.8)]["F_single"] for t in range(8)]),
                np.median([out[(sig2, t, 0.8)]["F_loop"] for t in range(8)]),
                np.median([out[(sig2, t, 0.8)]["F_model"] for t in range(8)]),
                time.time() - t0,
            ),
            flush=True,
        )
    save("scintillation", dict(out=out, SIG=SIG))
    print("saved scintillation (%.0fs)" % (time.time() - t0))


if __name__ == "__main__" and sys.argv[1] == "tables":
    # mean over the sequences and over the last K/2 frames; sd over the sequences of their steady-state means
    import csv
    import glob

    def runs(prefix):
        out, k = {}, None
        for path in sorted(glob.glob(os.path.join(CACHE, prefix + "_*.pkl"))):
            d = pickle.load(open(path, "rb"))
            out.update(d["out"])
            k = d["K"]
        return out, k // 2

    def values(seqs, key):
        return np.array([[np.nan if r[key] is None else float(r[key]) for r in rr] for rr in seqs])

    def summary(d, h):
        so, eo = values(d["oracle"], "strehl"), values(d["oracle"], "ec")
        si, ei = values(d["ideal"], "strehl"), values(d["ideal"], "ec")
        sign_ok = np.array([[r["sign_ok"] is True for r in rr] for rr in d["ideal"]], float)
        frac, frac_w = values(d["ideal"], "sign_frac"), values(d["ideal"], "sign_frac_w")
        return {
            "oracle_strehl": so[:, h:].mean(),
            "oracle_strehl_sd": so[:, h:].mean(1).std(),
            "oracle_ec": eo[:, h:].mean(),
            "oracle_ec_sd": eo[:, h:].mean(1).std(),
            "ideal_strehl": si[:, h:].mean(),
            "ideal_strehl_sd": si[:, h:].mean(1).std(),
            "ideal_ec": ei[:, h:].mean(),
            "ideal_ec_sd": ei[:, h:].mean(1).std(),
            "sign_ok": sign_ok[:, 1:].mean(),
            "sign_frac_steady": np.nanmean(frac[:, h:]),
            "sign_frac_first": np.nanmean(frac[:, 1]),
            "sign_frac_w_steady": np.nanmean(frac_w[:, h:]),
            "err33_steady": np.nanmean(values(d["ideal"], "err33")[:, h:]),
            "change_per_frame": np.mean(d["delta"]) if "delta" in d else np.nan,
            "ec_bound": np.mean([i["ec_bound"] for i in d["info"]]),
            "strehl_bound": np.mean([i["strehl_bound"] for i in d["info"]]),
            "ec_uncorrected": eo[:, 0].mean(),
        }

    tables = {
        "loop_period": [
            "oracle_strehl",
            "oracle_strehl_sd",
            "oracle_ec",
            "oracle_ec_sd",
            "ideal_strehl",
            "ideal_strehl_sd",
            "ideal_ec",
            "ideal_ec_sd",
            "sign_ok",
            "sign_frac_steady",
            "sign_frac_first",
            "sign_frac_w_steady",
            "err33_steady",
            "change_per_frame",
            "ec_bound",
            "strehl_bound",
            "ec_uncorrected",
        ],
        "photons": [
            "ideal_ec",
            "ideal_ec_sd",
            "ideal_strehl",
            "sign_frac_first",
            "sign_frac_steady",
            "sign_frac_w_steady",
            "err33_steady",
            "oracle_ec",
            "ec_uncorrected",
            "ec_bound",
        ],
    }
    for prefix, first in (("loop_period", "flow"), ("photons", "photons")):
        out, h = runs(prefix)
        with open(os.path.join(CACHE, prefix + ".csv"), "w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow([first] + tables[prefix])
            for key in sorted(out):
                s = summary(out[key], h)
                writer.writerow([key] + [round(float(s[c]), 4) for c in tables[prefix]])
        print("wrote %s.csv (%d rows)" % (prefix, len(out)))
