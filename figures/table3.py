"""Table 3: steady-state Strehl ratio of the four sensing methods with ideal estimators.

Median over eight static aberrations or eight frozen-flow sequences of 0.8 rad, mean over iterations 8 to 10.
Reads data/photon_comparison/*.pkl (python -m simulations.photon_comparison run <method> <read|shot> [--flow r]).
"""

import glob
import os
import pickle

import numpy as np

from style import DATA

METHODS = (
    ("image_loop", "image-based loop"),
    ("sh_beacon", "SH sensor, 532 nm beacon"),
    ("sh_main_beam", "SH sensor, 1080 nm beam"),
    ("phase_diversity", "phase diversity"),
)
NOISE = {"shot": "shot noise only", "read": "with read noise"}


def steady(d, k=10):
    st = np.array(
        [[np.mean([r[j]["strehl"] for j in range(k - 2, k + 1)]) for r in d["res"][n]] for n in d["N"]]
    )
    return d["N"], np.median(st, 1)


if __name__ == "__main__":
    data = {}
    for fn in glob.glob(os.path.join(DATA, "photon_comparison", "*.pkl")):
        d = pickle.load(open(fn, "rb"))
        data[(d["method"], d["tag"], round(float(d.get("flow", 0.0)), 2))] = d
    for flow, title in (
        (0.0, "static aberrations"),
        (0.3, "loop period 0.3 coherence times"),
        (1.0, "loop period 1 coherence time"),
    ):
        for tag in ("shot", "read"):
            print("%s, %s" % (title, NOISE[tag]))
            for method, label in METHODS:
                if (method, tag, flow) in data:
                    N, med = steady(data[(method, tag, flow)])
                    print("  %-26s" % label, "  ".join("%.0e: %.2f" % (n, v) for n, v in zip(N, med)))
