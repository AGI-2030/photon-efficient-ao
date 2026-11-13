"""Run a trained network in the closed loop under the conditions of a figure of the paper.

    python scripts/evaluate.py weights/film_resnet_full.pt grid --out runs/grid/film_resnet_full
    python scripts/evaluate.py weights/film_resnet_full.pt loop_period --methods network oracle cramer_rao \
        --out runs/loop_period/film_resnet_full
    python scripts/evaluate.py weights/film_resnet_static.pt static --out runs/static

Experiments (aonet/evaluation.py):
    grid         36 cells of D/r0, loop period and photons, Fig. 2(d, e) and Table S1
    main         cell 26 of the grid (D/r0 = 6, 0.3 coherence times, 1e4 photons), Fig. 2(b, c) and Fig. S5
    strength     D/r0 at 0.3 coherence times and 1e5 photons, Fig. 3(a)
    photons      photons per frame at D/r0 = 6, Fig. 3(b)
    loop_period  loop period at D/r0 = 6 and 1e4 photons, Fig. 3(c)
    gain         loop gain at D/r0 = 6 and 1e4 photons, Fig. S3(f)
    static       the eight static aberrations of Table 3 at nine photon levels, Fig. 5(a)

Methods: network, oracle (true residual as estimate) and cramer_rao (aonet/cramer_rao.py). The traces (one .npz
per condition and method, sequences by frames) and summary.csv go to --out. The steady state of summary.csv runs
from --first-steady-frame to the last frame.
"""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from aonet.analysis import summary_row
from aonet.evaluation import EXPERIMENTS, evaluate, evaluate_static
from aonet.models import load_model


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("weights", help="weights/<name>.pt")
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS) + ["static"])
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--precision", choices=("bf16", "fp32"), default="bf16", help="network precision on a GPU"
    )
    parser.add_argument(
        "--methods", nargs="+", default=["network"], choices=("network", "oracle", "cramer_rao")
    )
    parser.add_argument("--first-steady-frame", type=int, default=15)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", 0)
    model, cfg = load_model(args.weights)
    out = Path(args.out)

    if args.experiment == "static":
        path = evaluate_static(model, cfg, out / "static.npz", device, precision=args.precision)
        data = np.load(path)
        strehl = (
            data["network_strehl"][:, :, -1, :, 8:].astype(float).mean(axis=(2, 3))
        )  # 7 e-, frames 8 to 10
        for n, s in zip(data["photons"], strehl.T):
            print(
                f"{n:9.0f} photons  Strehl ratio {s.mean():.3f} +- {s.std(ddof=1):.3f} over the aberrations"
            )
        return

    experiment = EXPERIMENTS[args.experiment]
    evaluate(model, cfg, experiment, out, device, methods=tuple(args.methods), precision=args.precision)
    rows = []
    for index, condition in enumerate(experiment.conditions):
        for method in args.methods:
            trace = dict(np.load(out / f"cell_{index:03d}_{method}.npz"))
            rows.append(
                {
                    "cell": index,
                    **asdict(condition),
                    "method": method,
                    **summary_row(trace, args.first_steady_frame),
                }
            )
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
