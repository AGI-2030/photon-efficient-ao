import argparse
import csv
import os

import numpy as np

SETTINGS = [
    ("no_turb", "room temperature"),
    ("turb_1", "90 °C"),
    ("turb_2", "90 °C, fan 5%"),
    ("turb_3", "90 °C, fan 50%"),
    ("turb_4", "150 °C"),
    ("turb_6", "210 °C"),
    ("turb_5", "270 °C"),
]
COLUMNS = [
    "setting",
    "repeat",
    "r0_cm",
    "r0_min_cm",
    "r0_max_cm",
    "flow_mps",
    "before_px",
    "after_px",
    "reduction_pct",
    "zero_command_change_pct",
]


def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def network_rows(runs):
    out = []
    for cond, label in SETTINGS:
        for rep in sorted({int(r["repeat"]) for r in runs if r["condition"] == cond}):
            rr = [r for r in runs if r["condition"] == cond and int(r["repeat"]) == rep]
            assert len(rr) == 10, (cond, rep, len(rr))
            g = [r for r in rr if float(r["gain"]) == 0.25]
            z = [r for r in rr if float(r["gain"]) == 0.0]
            assert len(g) == 3 and len(z) == 1
            before = np.mean([float(r["Rinst_ref"]) for r in g])
            after = np.mean([float(r["Rinst_late"]) for r in g])
            z = z[0]
            out.append(
                dict(
                    setting=label,
                    repeat=rep,
                    r0_cm=np.mean([float(r["r0_run_cm"]) for r in rr]),
                    r0_min_cm=min(float(r["r0_min_cm"]) for r in rr),
                    r0_max_cm=max(float(r["r0_max_cm"]) for r in rr),
                    flow_mps=np.mean([float(r["wind_mean_mps"]) for r in rr]),
                    before_px=before,
                    after_px=after,
                    reduction_pct=100 * (1 - after / before),
                    zero_command_change_pct=100 * (1 - float(z["Rinst_late"]) / float(z["Rinst_ref"])),
                )
            )
    return out


def sh_rows(sh):
    out = []
    for tag in ("90", "150"):
        for r in sorted((r for r in sh if r["set"] == tag), key=lambda r: r["run"]):
            x = [float(r["r0_ref"]), float(r["r0_after"])]
            out.append(
                dict(
                    setting=f"{tag} °C, SH loop",
                    repeat=int(r["run"][3:]),
                    r0_cm=np.mean(x),
                    r0_min_cm=min(x),
                    r0_max_cm=max(x),
                    flow_mps=0.0,
                    before_px=float(r["R_ref"]),
                    after_px=float(r["R_after"]),
                    reduction_pct=float(r["reduction_pct"]),
                    zero_command_change_pct=None,
                )
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="data/lab/r865_all_runs.csv")
    ap.add_argument("--sh", default="data/lab/sh_comparison.csv")
    ap.add_argument("--out", default="data/lab/table_s3.csv")
    a = ap.parse_args()
    rows = network_rows(read(a.runs)) + sh_rows(read(a.sh))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as g:
        w = csv.writer(g, lineterminator="\n")
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow(
                [r["setting"], r["repeat"]] + ["" if r[k] is None else f"{r[k]:.3f}" for k in COLUMNS[2:]]
            )
    for r in rows:
        z = r["zero_command_change_pct"]
        r0 = f"r0 {r['r0_cm']:.1f} ({r['r0_min_cm']:.1f}-{r['r0_max_cm']:.1f})"
        r86 = f"{r['before_px']:.1f} -> {r['after_px']:.1f}  {r['reduction_pct']:+.1f}%"
        zero = "-" if z is None else f"{z:+.1f}%"
        print(f"{r['setting']:<18} {r['repeat']}  {r0}  flow {r['flow_mps']:.2f}  {r86}  zero {zero}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
