"""Encircled energy around the frames in which the network reverses the even part of the residual (Fig. 3(d)).

    python scripts/evaluate.py weights/film_resnet_full.pt loop_period --out runs/loop_period/film_resnet_full
    python scripts/sign_events.py runs/loop_period/film_resnet_full --out data/network

The event definition is given in aonet/sign_events.py. For every loop period of the loop_period experiment the
script writes the event-aligned encircled energy (sign_events.csv) and the event counts with the change of the
encircled energy after the start of an event (sign_events_summary.csv).
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from aonet.analysis import even_mode_mask
from aonet.config import load_config
from aonet.evaluation import EXPERIMENTS
from aonet.sign_events import analyze, pupil_gram

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_SEED = 202_610_211  # added to the seed and the trajectory group of the experiment
CONDITIONS = "FiLM-ResNet with the full input, D/r0 = 6, N = 1e4, g = 0.5, 64 sequences of 30 frames"


def write(path: Path, comment: str, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        handle.write(f'"# {comment}"\n')
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs", help="output folder of scripts/evaluate.py for the loop_period experiment")
    parser.add_argument("--out", default="data/network")
    args = parser.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    experiment = EXPERIMENTS["loop_period"]
    gram = pupil_gram(load_config(ROOT / "configs" / "film_resnet_full.toml").optics)
    even = even_mode_mask()
    aligned, summary = [], []
    for index, cond in enumerate(experiment.conditions):
        trace = dict(np.load(runs / f"cell_{index:03d}_network.npz"))
        result = analyze(trace, gram, even, BOOTSTRAP_SEED + experiment.seed + cond.group)
        ee = result["ee"]
        for lag, mean, se, low, high in zip(ee["lag"], ee["mean"], ee["se"], ee["low"], ee["high"]):
            aligned.append(
                [cond.tau_ratio, int(lag), mean, se, low, high, result["complete_events"], ee["sequences"]]
            )
        at = dict(zip(ee["lag"].tolist(), ee["mean"]))
        summary.append(
            [
                cond.tau_ratio,
                result["events"],
                result["complete_events"],
                ee["sequences"],
                result["reversed_fraction"],
                at[1] - at[-2],
                at[1] - at[-1],
                result["drop_le_0.05"],
                result["change_le_0.05"],
            ]
        )
    write(
        out / "sign_events.csv",
        f"{CONDITIONS}; mean encircled energy at frames -2 to +5 around the start of an event (q <= -0.2), the events "
        "of each sequence averaged first; standard error and 95% bootstrap interval over the sequences (Fig. 3(d))",
        ["tau_ratio", "lag", "ee_mean", "ee_se", "ee_low", "ee_high", "events", "sequences"],
        aligned,
    )
    write(
        out / "sign_events_summary.csv",
        f"{CONDITIONS}; events from frame 1 on, complete events have frames -2 to +5 within frames 1 to 29; changes "
        "of the event-aligned mean encircled energy from lag -2 and lag -1 to lag +1; fractions of the complete "
        "events whose encircled energy at lag +1 is not more than 0.05 below, or within 0.05 of, that at lag -1",
        [
            "tau_ratio",
            "events",
            "complete_events",
            "sequences",
            "reversed_frames",
            "ee_change_from_lag_m2",
            "ee_change_from_lag_m1",
            "drop_le_0.05",
            "change_le_0.05",
        ],
        summary,
    )


if __name__ == "__main__":
    main()
