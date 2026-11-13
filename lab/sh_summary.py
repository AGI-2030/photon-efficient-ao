import argparse
import csv
import os

import numpy as np

from . import sh_frames as ffm
from .common import add_data_argument, clock, load_atm, window


def by_run(data, RR):
    atm = load_atm(data, "20261104", ("r0", "heater", "T1", "T2", "T3"))
    r0s = atm["r0"]
    rows = []
    for tag in ffm.SETS:
        for run in ffm.RUNS:
            ref = [r for r in RR if r["set"] == tag and r["run"] == run and r["phase"] == "ref"]
            aft = [r for r in RR if r["set"] == tag and r["run"] == run and r["phase"] == "after"]
            tref = np.array([float(r["t_epoch"]) for r in ref])
            taf = np.array([float(r["t_epoch"]) for r in aft])
            Rref = np.mean([float(r["r865"]) for r in ref])
            Raf = np.mean([float(r["r865"]) for r in aft])
            r0_ref = window(r0s, tref[0], tref[-1])
            r0_ref = (
                float(np.mean(r0_ref)) if len(r0_ref) else float(np.interp(tref.mean(), r0s[0], r0s[1][:, 0]))
            )
            r0_af = float(np.mean(window(r0s, taf[0], taf[-1])))
            t0, t1 = tref[0], taf[-1]
            T = float(np.mean([window(atm[f"T{i}"], t0, t1).mean() for i in (1, 2, 3)]))
            rows.append(
                dict(
                    set=tag,
                    run=run,
                    ref_start=clock(tref[0]),
                    ref_end=clock(tref[-1]),
                    after_start=clock(taf[0]),
                    after_end=clock(taf[-1]),
                    r0_ref=r0_ref,
                    r0_after=r0_af,
                    T_C=T,
                    heater_W=float(window(atm["heater"], t0, t1).mean()),
                    R_ref=Rref,
                    R_after=Raf,
                    reduction_pct=100 * (1 - Raf / Rref),
                )
            )
    for r in rows:
        print(
            r["set"],
            r["run"],
            "r0 ref %.2f after %.2f | R ref %.2f after %.2f | reduction %.1f%%"
            % (r["r0_ref"], r["r0_after"], r["R_ref"], r["R_after"], r["reduction_pct"]),
        )
    return rows


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as g:
        w = csv.DictWriter(g, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    add_data_argument(ap)
    ap.add_argument("--out", default="sh_comparison.csv")
    ap.add_argument("--frames")
    a = ap.parse_args()
    outdir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(outdir, exist_ok=True)
    frames_csv = a.frames or os.path.join(outdir, "sh_frames.csv")
    if not a.frames:
        write(frames_csv, ffm.compute(a.data))
    with open(frames_csv) as f:
        R = list(csv.DictReader(f))
    write(a.out, by_run(a.data, R))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
