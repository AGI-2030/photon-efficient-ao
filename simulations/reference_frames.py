"""Sign decision with a wavefront that moves between frames (Fig. 1(e) and Fig. S3(e)).

A frozen-flow screen at D/r0 = 6 is translated by 0.314 (r0 in pixels) tau_loop/tau_0 per frame. The
even sign is chosen by the two-frame test with the initial frame ("I0") or the previous frame ("prev")
as reference, by continuity with the previous estimate, or at random; the oracle is given the true sign.

Usage:
    python -m simulations.reference_frames run <ratio>   one loop period tau_loop/tau_0 (Fig. 1(e))
    python -m simulations.reference_frames gain          loop gain with magnitude noise (Fig. S3(e))
Results are written to data/reference_frames/.
"""

import os
import sys
import time
import pickle
import numpy as np
from scipy.ndimage import shift as ndshift
from aosim import Simulator, data_dir

CACHE = data_dir("reference_frames")

s = Simulator(N=512, Dpix=128, nmodes=33)
P, I = s.phase, s.intensity
crop = lambda im: s.crop(im, 20)
PEAK0 = I(np.zeros((s.N, s.N)))[s.N // 2, s.N // 2]

DR0, G, NPH, K, TRIALS = 6, 0.5, 1e4, 10, 20
RATIOS = [0.03, 0.1, 0.3, 1.0]  # tau_loop / tau_0
METHODS = ["oracle", "I0", "prev", "continuity", "random"]


def sequence(ratio, seed):
    """Frozen-flow screen translated by v*tau_loop per frame; tau_0 = 0.314 r0/v (Greenwood)."""
    rng = np.random.default_rng(seed)
    scr = s.vk_screen(DR0, rng=rng)
    px = 0.314 * (s.Dpix / DR0) * ratio  # pixels per frame
    coefs = []
    for k in range(K):
        sk = ndshift(scr, (0.0, -k * px), order=1, mode="wrap")
        coefs.append(s.project(sk))
    return coefs, px


def run(coefs, method, rng, g=G, sig_mult=0.0, sig_add=0.0):
    """Closed loop with gain g; sig_mult and sig_add are multiplicative and additive (rad) errors of the magnitudes."""
    c = np.zeros(33)
    frames, cs, rec = [], [], []
    prev_e = None
    for k in range(K):
        psi = coefs[k] - c
        Ik = rng.poisson(crop(I(P(psi))) * NPH)
        frames.append(Ik)
        cs.append(c.copy())
        Tpsi = s.twin_coeffs(psi)
        if method == "oracle":
            choose = True
        elif k == 0 or method == "random":
            choose = rng.random() < 0.5
        elif method in ("I0", "prev"):
            j = 0 if method == "I0" else k - 1
            # frozen-model prediction of frame j, hypothesis psi
            muA = crop(I(P(c + psi - cs[j]))) * NPH
            muB = crop(I(P(c + Tpsi - cs[j]))) * NPH  # same, hypothesis T psi
            fl = 1e-15
            choose = (frames[j] * np.log((muA + fl) / (muB + fl))).sum() - (muA - muB).sum() > 0
        elif method == "continuity":
            choose = float(np.dot(s.even(psi), prev_e)) >= 0
        psi_hat = psi if choose else Tpsi
        prev_e = s.even(psi_hat)
        Ir = I(P(psi))
        rec.append(
            dict(
                k=k,
                correct=bool(choose),
                rms_e=s.rms(P(s.even(psi))),
                rms=s.rms(P(psi)),
                strehl=Ir[s.N // 2, s.N // 2] / PEAK0,
            )
        )
        if sig_mult > 0 or sig_add > 0:
            psi_hat = psi_hat * (1 + sig_mult * rng.standard_normal(33)) + sig_add * rng.standard_normal(33)
        c = c + g * psi_hat
    return rec


# additive magnitude noise per mode (rad), as from a photon-limited estimator
GAINS, GRATIOS, SM, SAS = [0.2, 0.3, 0.5, 0.7, 0.85, 1.0], [0.1, 0.3, 1.0], 0.0, [0.02, 0.07]
if __name__ == "__main__" and (len(sys.argv) < 2 or sys.argv[1] not in ("run", "gain")):
    sys.exit(__doc__)


if __name__ == "__main__" and sys.argv[1] == "gain":
    t0 = time.time()
    out = {}
    for ratio in GRATIOS:
        seqs = [sequence(ratio, 500 + t)[0] for t in range(TRIALS)]
        for sa in SAS:
            for g in GAINS:
                for m in ("oracle", "prev"):
                    R = [
                        run(
                            seqs[t],
                            m,
                            np.random.default_rng(7000 + 100 * t + int(10 * g)),
                            g=g,
                            sig_mult=SM,
                            sig_add=sa,
                        )
                        for t in range(TRIALS)
                    ]
                    st = np.array([[r[k]["strehl"] for k in range(K)] for r in R])
                    pc = np.array([[r[k]["correct"] for k in range(K)] for r in R])
                    rms = np.array([[r[k]["rms"] for k in range(K)] for r in R])
                    out[(ratio, g, m, sa)] = dict(
                        steady=float(st[:, -3:].mean()),
                        steady_sd=float(st[:, -3:].mean(1).std()),
                        pc=float(pc[:, 1:].mean()),
                        rms_steady=float(rms[:, -3:].mean()),
                    )
                    print(
                        "ratio %.1f noise %.2f g %.2f %-6s steady Strehl %.3f (sd %.3f), rms %.3f, P(correct) %.2f  (%.0fs)"
                        % (
                            ratio,
                            sa,
                            g,
                            m,
                            out[(ratio, g, m, sa)]["steady"],
                            out[(ratio, g, m, sa)]["steady_sd"],
                            out[(ratio, g, m, sa)]["rms_steady"],
                            out[(ratio, g, m, sa)]["pc"],
                            time.time() - t0,
                        ),
                        flush=True,
                    )
    drifts = {
        ratio: float(
            np.mean(
                [
                    np.mean([s.rms(P(c[k] - c[k - 1])) for k in range(1, K)])
                    for c in [sequence(ratio, 500 + t)[0] for t in range(5)]
                ]
            )
        )
        for ratio in GRATIOS
    }
    pickle.dump(
        dict(gains=GAINS, ratios=GRATIOS, SM=SM, SAS=SAS, drifts=drifts, out=out),
        open(os.path.join(CACHE, "gain.pkl"), "wb"),
    )
    print("gain done %.0fs" % (time.time() - t0))

if __name__ == "__main__" and sys.argv[1] == "run":
    ratio = float(sys.argv[2])
    t0 = time.time()
    res = {m: [] for m in METHODS}
    drift = []
    for t in range(TRIALS):
        coefs, px = sequence(ratio, 500 + t)
        drift.append(np.mean([s.rms(P(coefs[k] - coefs[k - 1])) for k in range(1, K)]))
        for m in METHODS:
            rng = np.random.default_rng(1000 * t + METHODS.index(m))
            res[m].append(run(coefs, m, rng))
    out = dict(
        ratio=ratio,
        px=px,
        drift=float(np.mean(drift)),
        res=res,
        rms0=float(np.mean([s.rms(P(sequence(ratio, 500 + t)[0][0])) for t in range(3)])),
    )
    pickle.dump(out, open(os.path.join(CACHE, "ratio_%.2f.pkl" % ratio), "wb"))
    print(
        "ratio %.2f: shift %.1f px/frame, drift %.3f rad/frame, aberration %.2f rad, %.0fs"
        % (ratio, px, out["drift"], out["rms0"], time.time() - t0)
    )
    for m in METHODS:
        R = res[m]
        pc = np.mean([[r[k]["correct"] for k in range(K)] for r in R], 0)
        st = np.mean([[r[k]["strehl"] for k in range(K)] for r in R], 0)
        print("  %-10s P(correct) %s | Strehl %s" % (m, np.round(pc[1:], 2), np.round(st, 2)))
