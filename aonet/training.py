"""DAgger training with the network inside the simulated loop.

Iteration i:
  1. Run episodes_per_iteration sequences. Each sequence is driven, with probability beta_i, by the true residual
     plus Gaussian noise and otherwise by the current network. Every visited state is stored with its true residual
     (the label) in the replay buffer.
  2. Take updates_per_iteration optimizer steps on batches drawn uniformly from the buffer.
During the first warmup_iterations the true residual drives every sequence with a wider noise, and the largest
D/r0 and loop period of the training draws grow linearly over curriculum_iterations.

Every random draw comes from a generator seeded by (seed, 0, iteration, ...), so a run with the same settings on
the same hardware produces the same data. The rollout batch and the device change the stream. The checkpoint with
the highest mean steady-state Strehl ratio on the validation grid is kept as best.pt.
"""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from dataclasses import fields, replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .config import ClosedLoopConfig
from .environment import ClosedLoopEnvironment, EpisodeBatch, sign_accuracy
from .models import build_model
from .optics import noll_indices
from .replay import ReplayBuffer


def mix_seed(*parts: int) -> int:
    seed = 0
    for p in parts:
        seed = (seed * 1_000_003 + int(p)) % (2**62)
    return seed


def seeded_generator(device: torch.device, *parts: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(mix_seed(*parts))
    return generator


def autocast(device: torch.device, precision: str):
    if precision == "bf16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def dagger_schedule(cfg: ClosedLoopConfig, iteration: int) -> dict[str, float]:
    d = cfg.dagger
    warm = iteration < d.warmup_iterations
    t = max(0, iteration - d.warmup_iterations)
    decay = max(0.0, 1.0 - t / max(1, d.beta_decay_iterations))
    beta = 1.0 if warm else d.beta_end + (d.beta_start - d.beta_end) * decay
    frac = min(1.0, iteration / max(1, d.curriculum_iterations))
    tr = cfg.turbulence
    dmax = d.curriculum_start_d_over_r0_max + frac * (
        tr.d_over_r0_range[1] - d.curriculum_start_d_over_r0_max
    )
    if cfg.reference.phase_mode == "static_modal_33":
        tmax = 0.0
        dmax = cfg.reference.static_shape_d_over_r0
    else:
        log_start = math.log(d.curriculum_start_tau_ratio_max)
        tmax = math.exp(log_start + frac * (math.log(tr.tau_ratio_range[1]) - log_start))
    return {
        "beta": beta,
        "warmup": float(warm),
        "d_over_r0_max": dmax,
        "tau_ratio_max": tmax,
        "oracle_noise_rad": d.warmup_oracle_noise_rad if warm else d.oracle_noise_rad,
    }


def relative_loss(prediction: torch.Tensor, label: torch.Tensor, floor_rad: float) -> torch.Tensor:
    """Squared error divided by the energy of the residual, with a floor of floor_rad per mode.

    Large transients and the small steady-state residual then carry similar weight.
    """
    nmodes = label.shape[-1]
    denom = label.pow(2).sum(-1) + nmodes * floor_rad**2
    return ((prediction.float() - label).pow(2).sum(-1) / denom).mean()


class LRSchedule:
    """Linear warmup, then cosine decay to final_fraction of the base learning rate."""

    def __init__(self, optimizer, warmup: int, total: int, final_fraction: float):
        self.opt, self.warmup, self.total, self.final = optimizer, warmup, total, final_fraction
        self.base = [g["lr"] for g in optimizer.param_groups]

    def set(self, step: int) -> float:
        if step < self.warmup:
            f = (step + 1) / self.warmup
        else:
            t = min(1.0, (step - self.warmup) / max(1, self.total - self.warmup))
            f = self.final + (1 - self.final) * 0.5 * (1 + math.cos(math.pi * t))
        for g, b in zip(self.opt.param_groups, self.base):
            g["lr"] = b * f
        return self.base[0] * f


def _azimuthal_order(cfg: ClosedLoopConfig, device: torch.device) -> torch.Tensor:
    noll = noll_indices(cfg.optics.noll_start - 1 + cfg.optics.nmodes, 1)
    return torch.tensor([m for (_, _, m) in noll[cfg.optics.noll_start - 1 :]], device=device)


def _cell_metrics(name: str, st, steady: slice) -> dict:
    return {
        f"{name}_steady_strehl": float(st.strehl[:, steady].mean()),
        f"{name}_steady_residual_rms": float(st.residual_rms[:, steady].mean()),
        f"{name}_steady_dm_residual_norm": float(st.dm_residual_norm[:, steady].mean()),
        f"{name}_steady_ee": float(st.ee[:, steady].mean()),
        f"{name}_estimate_error_norm": float(st.estimate_error_norm[:, steady].mean()),
        f"{name}_strehl_by_frame": st.strehl.mean(0).tolist(),
        f"{name}_ee_by_frame": st.ee.mean(0).tolist(),
    }


@torch.no_grad()
def validate(
    cfg: ClosedLoopConfig, env: ClosedLoopEnvironment, policy, include_oracle: bool
) -> dict[str, dict]:
    """Validation grid used to select the checkpoint; each cell has its own fixed seed.

    Frozen flow: every (D/r0, tau_loop/tau_0, photons) of cfg.evaluation, seeded by (evaluation.seed, cell).
    Static aberrations: every (RMS, photons, read noise) of cfg.reference, seeded by (evaluation.seed, 701, RMS
    index), so the same aberrations appear at every photon and read-noise level.
    """
    ev, ref, device = cfg.evaluation, cfg.reference, env.device
    n = ev.episodes_per_cell
    steady = slice(ev.frames_per_episode - ev.steady_frames, None)
    azimuthal_order = _azimuthal_order(cfg, device)

    def full(value):
        return torch.full((n,), float(value), device=device)

    runs = {"network": (policy, torch.zeros(n, dtype=torch.bool, device=device))}
    if include_oracle:
        runs["oracle"] = (None, torch.ones(n, dtype=torch.bool, device=device))
    static = ref.phase_mode == "static_modal_33"
    if static:
        cells = [
            (rms_index, rms, photons, read_noise)
            for rms_index, rms in enumerate(ref.static_validation_rms)
            for photons in ref.static_validation_photons
            for read_noise in ref.static_validation_read_noise_e
        ]
    else:
        cells = [(a, b, c) for a in ev.d_over_r0 for b in ev.tau_ratio for c in ev.photons]
    results: dict[str, dict] = {}
    for index, cell in enumerate(cells):
        if static:
            rms_index, rms, photons, read_noise = cell
            out = {"static_rms_rad": float(rms), "read_noise_e": float(read_noise), "photons": float(photons)}
            key = f"rms={rms:g}|nph={photons:.0e}|read={read_noise:g}"
        else:
            dr0, tau, photons = cell
            out = {}
            key = f"dr0={dr0:g}|tau={tau:g}|nph={photons:.0e}"
        for name, (pol, mask) in runs.items():
            if static:
                generator = seeded_generator(device, ev.seed, 701, rms_index)
                _, st = env.rollout(
                    full(ref.static_shape_d_over_r0),
                    full(0.0),
                    full(photons),
                    ev.frames_per_episode,
                    generator,
                    pol,
                    mask,
                    full(0.0),
                    collect=False,
                    static_rms=full(rms),
                    read_noise_e=full(read_noise),
                )
            else:
                generator = seeded_generator(device, ev.seed, index)
                _, st = env.rollout(
                    full(dr0),
                    full(tau),
                    full(photons),
                    ev.frames_per_episode,
                    generator,
                    pol,
                    mask,
                    torch.zeros(n, device=device),
                    collect=False,
                )
            out.update(_cell_metrics(name, st, steady))
            if name == "network":
                out["open_loop_strehl"] = float(st.strehl_open.mean())
                signs = sign_accuracy(st.action, st.label, azimuthal_order)
                out["network_sign_accuracy"] = float(signs[:, steady].mean())
                out["network_sign_accuracy_by_frame"] = signs.mean(0).tolist()
        results[key] = out
    return results


def summarize(results: dict[str, dict]) -> dict[str, float]:
    keys = [k for k in next(iter(results.values())) if not k.endswith("by_frame")]
    return {k: float(np.mean([r[k] for r in results.values() if k in r])) for k in keys}


def smoke_config(cfg):
    """A few small iterations, to check the installation."""
    cfg = replace(
        cfg,
        dagger=replace(
            cfg.dagger,
            iterations=2,
            episodes_per_iteration=4,
            rollout_batch=4,
            updates_per_iteration=2,
            batch_size=8,
            buffer_capacity=1000,
            warmup_iterations=1,
        ),
        evaluation=replace(
            cfg.evaluation,
            interval=2,
            checkpoint_interval=2,
            episodes_per_cell=2,
            d_over_r0=cfg.evaluation.d_over_r0[-1:],
            tau_ratio=cfg.evaluation.tau_ratio[-1:],
            photons=cfg.evaluation.photons[:1],
        ),
        reference=replace(
            cfg.reference,
            static_validation_rms=cfg.reference.static_validation_rms[:1],
            static_validation_photons=cfg.reference.static_validation_photons[-1:],
            static_validation_read_noise_e=cfg.reference.static_validation_read_noise_e[-1:],
        ),
    )
    cfg.validate()
    return cfg


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(
    cfg: ClosedLoopConfig,
    inputs: str,
    out_dir: str | Path,
    device: torch.device,
    *,
    micro_batch: int | None = None,
    rollout_batch: int | None = None,
    max_iterations: int | None = None,
    resume: str | Path | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Train one network and write best.pt, latest.pt, metrics.jsonl and the validation results to out_dir.

    micro_batch splits each optimizer batch to save memory (the gradient is unchanged); rollout_batch sets how many
    sequences are simulated at once and changes the random stream. resume continues from a latest.pt: the replay
    buffer is not saved, so it is refilled with sequences driven by the network, and the continuation is not
    identical to an uninterrupted run.
    """
    cfg.validate()
    dg, tr = cfg.dagger, cfg.training
    micro_batch = micro_batch or dg.batch_size
    rollout_batch = rollout_batch or dg.rollout_batch
    out_dir = Path(out_dir)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    precision = tr.precision if device.type == "cuda" else "fp32"

    env = ClosedLoopEnvironment(cfg, device)
    torch.manual_seed(cfg.seed)
    model = build_model(cfg.model, cfg.optics.nmodes, cfg.optics.network_image_size, inputs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr.learning_rate, weight_decay=tr.weight_decay)
    lr_schedule = LRSchedule(
        optimizer, tr.warmup_updates, dg.iterations * dg.updates_per_iteration, tr.final_lr_fraction
    )
    buffer = ReplayBuffer(dg.buffer_capacity, cfg.optics.network_image_size, cfg.optics.nmodes)
    start, update_step, best_score = 0, 0, -math.inf
    if resume is not None:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["config"] != cfg.to_dict() or state["inputs"] != inputs:
            raise ValueError("the checkpoint was written with another configuration")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, update_step, best_score = state["iteration"], state["update_step"], state["best_score"]
        torch.set_rng_state(state["rng_cpu"])
        if device.type == "cuda" and state.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
    out_dir.mkdir(parents=True, exist_ok=resume is not None)
    _write_json(
        out_dir / "run.json",
        {
            "config": cfg.to_dict(),
            "inputs": inputs,
            "micro_batch": micro_batch,
            "rollout_batch": rollout_batch,
            "device": str(device),
            "precision": precision,
            "resumed_from": None if resume is None else str(resume),
            "torch": torch.__version__,
        },
    )

    def policy(previous, current, delta, log_exposure):
        with autocast(device, precision):
            return model(previous, current, delta, log_exposure).float()

    def rollouts(iteration: int, force_policy: bool = False) -> dict:
        schedule = dagger_schedule(cfg, iteration)
        model.eval()
        done, strehl, chunk = 0, [], 0
        while done < dg.episodes_per_iteration:
            size = min(rollout_batch, dg.episodes_per_iteration - done)
            generator = seeded_generator(device, cfg.seed, 0, iteration, chunk, int(force_policy))
            dr0, tau, photons = env.sample_conditions(
                size, generator, schedule["d_over_r0_max"], schedule["tau_ratio_max"]
            )
            beta = 0.0 if force_policy else schedule["beta"]
            expert = torch.rand(size, generator=generator, device=device) < beta
            noise = torch.rand(size, generator=generator, device=device) * schedule["oracle_noise_rad"]
            driver = None if bool(expert.all()) else policy
            batch, stats = env.rollout(
                dr0, tau, photons, cfg.loop.frames_per_episode, generator, driver, expert, noise
            )
            buffer.add(
                EpisodeBatch(**{f.name: getattr(batch, f.name).detach().cpu() for f in fields(EpisodeBatch)})
            )
            on_policy = ~expert
            if bool(on_policy.any()):
                strehl.append(float(stats.strehl[on_policy, -5:].mean()))
            done += size
            chunk += 1
        model.train()
        return {**schedule, "rollout_policy_strehl": float(np.mean(strehl)) if strehl else float("nan")}

    def latest(iteration: int) -> dict:
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "update_step": update_step,
            "best_score": best_score,
            "config": cfg.to_dict(),
            "inputs": inputs,
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }

    if resume is not None:
        refill = start
        while buffer.size < min(buffer.capacity, dg.batch_size * 50):
            rollouts(refill, force_policy=dagger_schedule(cfg, refill)["warmup"] == 0.0)
            refill += 1

    last = dg.iterations if max_iterations is None else min(dg.iterations, start + max_iterations)
    started = time.monotonic()
    for iteration in range(start, last):
        info = rollouts(iteration)
        generator = seeded_generator(device, cfg.seed, 0, iteration, 999_999)
        losses = []
        for _ in range(dg.updates_per_iteration):
            lr = lr_schedule.set(update_step)
            # the indices are drawn on the training device, the samples are gathered on the host
            index = torch.randint(0, buffer.size, (dg.batch_size,), generator=generator, device=device).cpu()
            sampled = buffer.gather(index)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), device=device)
            for offset in range(0, dg.batch_size, micro_batch):
                end = min(offset + micro_batch, dg.batch_size)
                previous, current, delta, log_exposure, label = (v[offset:end].to(device) for v in sampled)
                with autocast(device, precision):
                    prediction = model(previous, current, delta, log_exposure)
                loss = relative_loss(prediction, label, tr.loss_floor_rad)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at iteration {iteration}")
                weighted = loss * ((end - offset) / dg.batch_size)
                weighted.backward()
                batch_loss += weighted.detach()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr.gradient_clip_norm)
            optimizer.step()
            losses.append(batch_loss)
            update_step += 1
        record = {
            "iteration": iteration + 1,
            "update_step": update_step,
            "lr": lr,
            "loss": float(torch.stack(losses).mean()),
            "buffer": buffer.size,
            "elapsed_s": time.monotonic() - started,
            **info,
        }
        if (iteration + 1) % cfg.evaluation.interval == 0 or iteration + 1 == last:
            model.eval()
            results = validate(cfg, env, policy, include_oracle=iteration + 1 == cfg.evaluation.interval)
            model.train()
            summary = summarize(results)
            record["validation"] = summary
            score = summary["network_steady_strehl"]
            _write_json(out_dir / "validation" / f"iter_{iteration + 1:06d}.json", results)
            if score > best_score:
                best_score = score
                best = {
                    "model": model.state_dict(),
                    "config": cfg.to_dict(),
                    "inputs": inputs,
                    "iteration": iteration + 1,
                    "validation": summary,
                }
                _save(out_dir / "best.pt", best)
        with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        log(
            f"[{iteration + 1:5d}] loss {record['loss']:.4f}  lr {lr:.2e}  beta {info['beta']:.2f}"
            + (
                f"  validation Strehl {record['validation']['network_steady_strehl']:.3f}"
                if "validation" in record
                else ""
            )
            + f"  {record['elapsed_s']:.0f} s"
        )
        if (iteration + 1) % cfg.evaluation.checkpoint_interval == 0 or iteration + 1 == last:
            _save(out_dir / "latest.pt", latest(iteration + 1))
    return out_dir
