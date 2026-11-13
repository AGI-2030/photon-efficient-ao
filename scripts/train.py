"""Train one network from a configuration file.

    python scripts/train.py configs/film_resnet_full.toml --out runs/train/film_resnet_full
    python scripts/train.py configs/film_resnet_full.toml --out runs/smoke --device cpu --smoke

The [run] table of the file sets the input configuration and the batch settings used for the released weights.
A full run takes 8 to 23 hours on one A100 GPU (data/network/training_log.csv).
"""

import argparse

import torch

from aonet.config import load_config, load_run_settings
from aonet.training import smoke_config, train


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", help="configs/<name>.toml")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument(
        "--device", help="cuda, cuda:1 or cpu (default: training.device of the configuration)"
    )
    parser.add_argument("--micro-batch", type=int, help="split each optimizer batch (saves memory)")
    parser.add_argument(
        "--rollout-batch", type=int, help="sequences simulated at once (changes the random stream)"
    )
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--resume", help="latest.pt of an interrupted run in the same output folder")
    parser.add_argument("--smoke", action="store_true", help="two short iterations on few sequences")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run = load_run_settings(args.config)
    device = torch.device(args.device or cfg.training.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", 0)
    micro_batch = args.micro_batch or run["micro_batch"]
    rollout_batch = args.rollout_batch or run["rollout_batch"]
    max_iterations = args.max_iterations or run["max_iterations"]
    if args.smoke:
        cfg, micro_batch, rollout_batch, max_iterations = smoke_config(cfg), 4, None, None
    train(
        cfg,
        run["inputs"],
        args.out,
        device,
        micro_batch=micro_batch,
        rollout_batch=rollout_batch,
        max_iterations=max_iterations,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
