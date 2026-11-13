import argparse
import csv
import datetime as dt
import math
import os
from multiprocessing import Pool

import numpy as np

from .common import add_data_argument, clock, epoch, interp, load_atm, r865, window

NBIN = 8
PHASES = ("ref", "late", "hold", "after")


def r865_frames(fr):
    return float(np.mean([r865(f, 4) for f in fr]))


def phases(path):
    def ld(n):
        p = f"{path}/{n}"
        return np.load(p) if os.path.exists(p) else None

    ref = ld("ref_frames_u16.npy")
    fr = ld("frames_u16.npy")
    n_loop = len(fr) - 2
    late = fr[2 + n_loop // 2 :]
    return dict(ref=ref, late=late, hold=ld("hold_frames_u16.npy"), after=ld("after_frames_u16.npy")), n_loop


def job(path):
    ph, _ = phases(path)
    out = {}
    for k in PHASES:
        v = ph[k]
        if v is None or len(v) == 0:
            out[f"R_{k}"] = math.nan
            out[f"n_{k}_frames"] = 0
            continue
        out[f"R_{k}"] = r865(v.mean(0, dtype=np.float64), NBIN)
        out[f"n_{k}_frames"] = len(v)
    for k in PHASES:
        out[f"Rinst_{k}"] = r865_frames(ph[k]) if ph[k] is not None and len(ph[k]) else math.nan
    return out


def read_runs(root):
    with open(os.path.join(root, "runs.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    add_data_argument(ap)
    ap.add_argument("--out", default="r865_all_runs.csv")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    root = os.path.join(a.data, "network_loop")
    runs = read_runs(root)
    recs = []
    paths = []
    for day in sorted({r["day"] for r in runs}):
        atm = load_atm(a.data, day)
        rows = sorted((r for r in runs if r["day"] == day), key=lambda r: r["t_start"])
        for m in rows:
            mdir = f"{root}/{m['folder']}"
            E = epoch(dt.datetime.fromisoformat(m["t_end"]))
            rate = float(m["actual_rate_hz"])
            n = int(m["n_loop"])
            assert n == len(np.load(f"{mdir}/frames_u16.npy", mmap_mode="r")) - 2
            has_hold = int(m["n_hold"]) > 0
            t_after0 = E - 5
            t_hold0 = E - 10 if has_hold else E - 5
            t_loop0 = t_hold0 - n / rate
            t_ref0 = t_loop0 - 5
            t_late_mid = (t_loop0 + n / rate / 2) + n / rate / 4
            t_ref_mid = t_ref0 + 2.5
            rr = window(atm["r0"], t_ref0, E)
            Tm = np.mean([np.mean(window(atm[f"T{i}"], t_ref0, E)) for i in (1, 2, 3)])
            Wm = np.mean([np.mean(window(atm[f"W{i}"], t_ref0, E)) for i in (1, 2, 3)])
            rec = dict(m)
            rec.update(
                t_ref0=clock(t_ref0),
                t_epoch_ref0=t_ref0,
                t_epoch_end=E,
                r0_run_cm=float(rr.mean()) if len(rr) else interp(atm["r0"], (t_ref0 + E) / 2),
                r0_min_cm=float(rr.min()) if len(rr) else math.nan,
                r0_max_cm=float(rr.max()) if len(rr) else math.nan,
                r0_n=len(rr),
                r0_open_cm=interp(atm["r0"], t_ref_mid),
                r0_closed_cm=interp(atm["r0"], t_late_mid),
                cn2=float(np.mean(window(atm["r0"], t_ref0, E, 1))) if len(rr) else math.nan,
                T_mean_C=float(Tm),
                wind_mean_mps=float(Wm),
                heater_W=float(np.mean(window(atm["heater"], t_ref0, E))),
                fan_W=float(np.mean(window(atm["fan"], t_ref0, E))),
            )
            recs.append(rec)
            paths.append(mdir)
    print(len(recs), "measurements; computing R86.5 ...", flush=True)
    with Pool(a.jobs) as p:
        res = p.map(job, paths, chunksize=2)
    for r, o in zip(recs, res):
        assert o["n_ref_frames"] == int(r["n_ref"]) and o["n_hold_frames"] == int(r["n_hold"])
        assert o["n_after_frames"] == int(r["n_after"])
        r["n_late"] = o["n_late_frames"]
        r.update({k: v for k, v in o.items() if k.startswith(("R_", "Rinst_"))})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as g:
        w = csv.DictWriter(g, fieldnames=list(recs[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(recs)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
