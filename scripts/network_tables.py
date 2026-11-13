"""Tables of data/network from the outputs of scripts/evaluate.py.

    python scripts/network_tables.py --runs runs --weights weights --out data/network

Expected outputs of scripts/evaluate.py (a missing input skips its table):
    runs/main/<network>/        the main cell for the 12 networks of Fig. 2 (or runs/grid/<network>/, cell 26)
    runs/gain/film_resnet_full/ the gain experiment
    runs/static/static.npz      the static experiment
"""

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import torch

from aonet.analysis import even_mode_mask, mean_sd, parity_rms, sign_ratio, steady, steady_sign
from aonet.evaluation import EXPERIMENTS
from aonet.models import load_model

BACKBONES = {"film_resnet": "FiLM-ResNet", "vit": "ViT", "plain_cnn": "Plain CNN", "fno": "FNO"}
INPUTS = {"single_frame": "single", "two_frames": "pair", "full": "full"}
MAIN = (
    "D/r0 = 6, tau_loop/tau_0 = 0.3, g = 0.5, N = 1e4 photons over the focal plane, 64 sequences of 30 frames"
)
ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, comment: str, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        handle.write(f'"# {comment}"\n')
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {path}")


def skipped(tables: str, missing: str) -> None:
    print(f"skipped {tables}: missing {missing}")


def main_trace(runs: Path, network: str) -> dict | None:
    for folder, cell in (("main", 0), ("grid", 26)):
        path = runs / folder / network / f"cell_{cell:03d}_network.npz"
        if path.exists():
            return dict(np.load(path))
    return None


def input_ablation(runs: Path, out: Path) -> None:
    traces = {label: main_trace(runs, f"film_resnet_{name}") for name, label in INPUTS.items()}
    if any(t is None for t in traces.values()):
        skipped(
            "input_ablation_by_iteration.csv, even_odd_residual.csv", "main or grid runs of film_resnet_*"
        )
        return
    even = even_mode_mask()
    rows, parity_rows = [], []
    frames = traces["full"]["ee"].shape[1]
    for k in range(frames):
        ec = [v for t in traces.values() for v in mean_sd(t["ee"][:, k])]
        sign = [
            v for t in traces.values() for v in mean_sd(sign_ratio(t["even_correct"], t["even_count"])[:, k])
        ]
        rows.append([k] + ec + sign)
        rms = [float(r[:, k].mean()) for t in traces.values() for r in parity_rms(t["label"], even)]
        parity_rows.append([k] + rms)
    labels = list(INPUTS.values())
    write(
        out / "input_ablation_by_iteration.csv",
        f"{MAIN}; sign accuracy over even modes with |true coefficient| > 0.05 rad, averaged over frames and then "
        "over sequences (Fig. 2(b, c))",
        ["k"]
        + [f"{a}_{b}" for a in labels for b in ("ec", "ec_sd")]
        + [f"{a}_{b}" for a in labels for b in ("sign", "sign_sd")],
        rows,
    )
    write(
        out / "even_odd_residual.csv",
        f"{MAIN}; RMS of the even and odd parts of the 33-mode residual (rad), mean over sequences (Fig. S5)",
        ["k"] + [f"{a}_{b}_rms" for a in labels for b in ("even", "odd")],
        parity_rows,
    )


def backbones(runs: Path, weights: Path, out: Path) -> None:
    rows = []
    for backbone, label in BACKBONES.items():
        for name, input_label in INPUTS.items():
            trace = main_trace(runs, f"{backbone}_{name}")
            path = weights / f"{backbone}_{name}.pt"
            if trace is None or not path.exists():
                skipped("backbones.csv", f"main or grid run and weights of {backbone}_{name}")
                return
            model, _ = load_model(path)
            state = torch.load(path, map_location="cpu", weights_only=True)
            ec = mean_sd(steady(trace["ee"], 15))
            sign = mean_sd(steady_sign(trace["even_correct"], trace["even_count"], 15))
            params = sum(p.numel() for p in model.parameters())
            rows.append([label, input_label, *ec, *sign, params, state["iteration"]])
    write(
        out / "backbones.csv",
        f"{MAIN}, steady state over k = 15 to 29 (Fig. 2(d, e), Table S1)",
        ["backbone", "input", "ec_mean", "ec_sd", "sign_mean", "sign_sd", "params", "best_iter"],
        rows,
    )


def gain_sweep(runs: Path, out: Path) -> None:
    folder = runs / "gain" / "film_resnet_full"
    conditions = EXPERIMENTS["gain"].conditions
    if not all((folder / f"cell_{i:03d}_network.npz").exists() for i in range(len(conditions))):
        skipped("gain_sweep.csv", str(folder))
        return
    rows = []
    for i, c in enumerate(conditions):
        values = steady(np.load(folder / f"cell_{i:03d}_network.npz")["ee"], 20)
        rows.append([c.tau_ratio, c.gain, *mean_sd(values), len(values)])
    write(
        out / "gain_sweep.csv",
        "FiLM-ResNet with the full input, trained at g = 0.5 and run at other loop gains; D/r0 = 6, N = 1e4, "
        "64 sequences, steady state over k = 20 to 29 (Fig. S3(f))",
        ["tau_ratio", "gain", "ec_mean", "ec_sd", "n"],
        rows,
    )


def static_network(runs: Path, out: Path) -> None:
    path = runs / "static" / "static.npz"
    if not path.exists():
        skipped("static_network.csv", str(path))
        return
    from simulations.photon_comparison import steady_strehl

    data = np.load(path)
    read = list(data["read_noise"]).index(7.0)
    strehl = data["network_strehl"][:, :, read, :, 8:].astype(np.float64)
    network = strehl.mean(axis=(2, 3)).T  # (photons, aberrations)
    with open(ROOT / "data" / "photon_comparison" / "image_loop_read_static.pkl", "rb") as handle:
        reference = steady_strehl(pickle.load(handle))  # ideal estimator, (photons, aberrations)
    rows = []
    for n, net, ref in zip(data["photons"], network, reference):
        rows.append([round(float(n)), *mean_sd(net), *mean_sd(ref), *mean_sd(ref - net)])
    write(
        out / "static_network.csv",
        "network trained on static aberrations, on the eight aberrations of Table 3 (0.8 rad RMS), initial frame "
        "as reference, g = 0.5, read noise 7 e-, Strehl ratio over k = 8 to 10, 16 noise realizations per "
        "aberration; ref = ideal estimator of photon_comparison.py on the same aberrations; diff = ref minus "
        "network (Fig. 5(a))",
        ["photons", "strehl_mean", "strehl_sd", "ref_strehl_mean", "ref_strehl_sd", "diff_mean", "diff_sd"],
        rows,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--weights", default="weights")
    parser.add_argument("--out", default="data/network")
    args = parser.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    input_ablation(runs, out)
    backbones(runs, Path(args.weights), out)
    gain_sweep(runs, out)
    static_network(runs, out)


if __name__ == "__main__":
    main()
