import argparse
import csv
import datetime as dt
import math
import os

import numpy as np

from .common import add_data_argument, epoch, r865

SETS = ("90", "150")
RUNS = ("run1", "run2", "run3")
PHASES = ("ref", "after")


def frame_metrics(w, bg):
    w = w.astype(np.float32)
    tot = float(w.sum())
    if tot <= 0:
        return dict(sum=0, peak=0, r865=math.nan, bg=bg)
    return dict(sum=tot, peak=float(w.max()), r865=r865(w.astype(np.float64), 4), bg=bg)


def compute(data):
    rows = []
    for tag in SETS:
        for run in RUNS:
            for ph in PHASES:
                d = os.path.join(data, "sh_comparison", tag, run)
                fr = np.load(os.path.join(d, f"{ph}_frames_u8.npy"))
                with open(os.path.join(d, f"{ph}_frames.csv")) as f:
                    meta = list(csv.DictReader(f))
                assert len(meta) == len(fr)
                for m, w in zip(meta, fr):
                    t = epoch(dt.datetime.fromisoformat(m["time"]))
                    rows.append(
                        dict(
                            set=tag,
                            run=run,
                            phase=ph,
                            t_epoch=t,
                            counter=int(m["counter"]),
                            **frame_metrics(w, float(m["background"])),
                        )
                    )
                print(tag, run, ph, len(fr), "frames", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    add_data_argument(ap)
    ap.add_argument("--out", default="sh_frames.csv")
    a = ap.parse_args()
    rows = compute(a.data)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
