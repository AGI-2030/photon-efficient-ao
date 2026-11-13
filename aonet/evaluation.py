"""Closed-loop runs of trained networks under fixed conditions (the network results of the paper).

A condition fixes D/r0, tau_loop/tau_0, the photons per frame, the loop gain and a trajectory group. The phase
screens of batch b are drawn from a generator seeded by (seed, group, b, 11), the camera noise of frame k by
(seed, group, b, 29) + k * 1_000_003, and the noise of the Cramer-Rao reference by (seed, group, b, 419) +
k * 1_000_003. Every network, input configuration and reference therefore sees the same turbulence; the Poisson
draws depend on the image and so differ between methods.

Methods: "network" (with the reference frame it was trained with), "oracle" (the true residual as estimate) and
"cramer_rao" (aonet.cramer_rao).

The photon number is the expected number of detected photons over the whole focal plane before the camera window.
The encircled energy is computed from the noise-free image of the residual.
"""

from __future__ import annotations

import itertools
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from .config import ClosedLoopConfig
from .cramer_rao import CramerRaoReference
from .environment import ClosedLoopEnvironment
from .optics import noll_indices
from .static_scene import FixedShape
from .training import mix_seed, seeded_generator
from .turbulence import Trajectory

FRAME_SEED_STRIDE = 1_000_003


@dataclass(frozen=True)
class Condition:
    d_over_r0: float
    tau_ratio: float
    photons: float
    gain: float = 0.5
    group: int = 0


@dataclass(frozen=True)
class Experiment:
    seed: int
    conditions: tuple[Condition, ...]
    sequences: int = 64
    batch: int = 8
    frames: int = 30


def _grid() -> tuple[Condition, ...]:
    cells = itertools.product((0.6, 1.0, 2.0, 4.0, 6.0, 8.0), (0.1, 0.3, 1.0), (1e4, 1e6))
    return tuple(Condition(a, b, c, 0.5, i) for i, (a, b, c) in enumerate(cells))


def _photon_scan() -> tuple[Condition, ...]:
    levels = [round(10 ** (2 + k / 4), 8) for k in range(17)]
    scan = tuple(Condition(6.0, tau, n, 0.5, i) for i, tau in enumerate((0.1, 0.3, 1.0)) for n in levels)
    # beyond the training range, at 0.3 coherence times only
    return scan + (Condition(6.0, 0.3, 10**6.5, 0.5, 1), Condition(6.0, 0.3, 1e7, 0.5, 1))


EXPERIMENTS = {
    # Fig. 2(d, e), Table S1: 36 cells, every network
    "grid": Experiment(2_100_000, _grid()),
    # Fig. 2(b, c), Fig. S5: cell 26 of the grid
    "main": Experiment(2_100_000, (Condition(6.0, 0.3, 1e4, 0.5, 26),)),
    # Fig. 3(a): D/r0 at 0.3 coherence times and 1e5 photons
    "strength": Experiment(2_400_000, tuple(Condition(a, 0.3, 1e5) for a in (0.6, 1, 2, 3, 4, 5, 6, 7, 8))),
    # Fig. 3(b): photons per frame at D/r0 = 6
    "photons": Experiment(2_800_000, _photon_scan()),
    # Fig. 3(c): loop period at D/r0 = 6 and 1e4 photons
    "loop_period": Experiment(
        2_600_000,
        tuple(Condition(6.0, t, 1e4, 0.5, i) for i, t in enumerate((0, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 1))),
    ),
    # Fig. S3(f): loop gain at D/r0 = 6 and 1e4 photons
    "gain": Experiment(
        2_500_000,
        tuple(
            Condition(6.0, tau, 1e4, g, i)
            for i, tau in enumerate((0.1, 0.3, 1.0))
            for g in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        ),
    ),
}


@torch.no_grad()
def run_loop(
    env: ClosedLoopEnvironment,
    model: torch.nn.Module | None,
    trajectory: Trajectory,
    photons: torch.Tensor,
    frames: int,
    *,
    reference: str,
    noise_seed: int,
    precision: str = "bf16",
    cramer_rao: CramerRaoReference | None = None,
    cramer_rao_seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Expose, estimate and update for every frame; return per-frame records of every sequence.

    reference is "previous" or "initial" (the reference frame given to the network), "oracle" (the true residual
    replaces the estimate) or "cramer_rao". Scalars have shape (B, K), coefficients and commands (B, K, 33). command
    is the command under which the frame was exposed, delta_command the change since the previous frame, and
    network_delta the command difference given to the network (since the first frame for the initial reference).
    """
    loop = env.cfg.loop
    device = env.basis.dm.device
    batch = photons.numel()
    command = torch.zeros(batch, env.nmodes, device=device)
    previous_command = command.clone()
    initial_command = command.clone()
    pending: deque[torch.Tensor] = deque()
    previous_image = initial_image = previous_counts = None
    half = env.camera.half
    coordinates = torch.arange(2 * half, device=device, dtype=torch.float32) - half
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    core = torch.hypot(y, x) <= 1.22 * env.cfg.optics.pixels_per_lambda_over_d
    indices = noll_indices(env.cfg.optics.noll_start - 1 + env.nmodes)
    order = torch.tensor([m for _, _, m in indices[env.cfg.optics.noll_start - 1 :]], device=device)
    even_modes = order.abs() % 2 == 0
    trace: dict[str, list[torch.Tensor]] = {}

    def record(name: str, value: torch.Tensor) -> None:
        trace.setdefault(name, []).append(value.clone())

    for frame in range(frames):
        phase = trajectory.frame(frame)
        coefficients = env.basis.project(phase)
        phase = env.basis.remove_low_order(phase, coefficients)
        dm_coefficients = coefficients[:, env.basis.dm_slice]
        label = dm_coefficients - command
        residual = phase - env.basis.phase(command)
        fitting_residual = phase - env.basis.phase(dm_coefficients)

        clean = env.camera.intensity(residual, diversity=False)
        clean_open = env.camera.intensity(phase, diversity=False)
        clean_fitting = env.camera.intensity(fitting_residual, diversity=False)
        exposed = clean
        if env.cfg.optics.diversity_defocus_rad != 0:
            exposed = env.camera.intensity(residual, diversity=True)
        generator = torch.Generator(device=device)
        generator.manual_seed((int(noise_seed) + frame * FRAME_SEED_STRIDE) % (2**63))
        counts = env.camera.expose(exposed, photons, generator)
        image = env.camera.network_input(counts)
        counts_sum = counts.sum(dim=(-2, -1))
        log_exposure = counts_sum.clamp_min(1.0).log10() - 4.0
        if initial_image is None:
            initial_image = image
        reference_image = initial_image if reference == "initial" else previous_image
        reference_image = image if reference_image is None else reference_image
        delta = command - previous_command
        network_delta = command - initial_command if reference == "initial" else delta

        if reference == "oracle":
            estimate = label.clone()
        elif reference == "cramer_rao":
            noise = torch.Generator(device=device)
            noise.manual_seed((int(cramer_rao_seed) + frame * FRAME_SEED_STRIDE) % (2**63))
            estimate = cramer_rao.estimate(label, photons, previous_counts, delta, noise)
        else:
            use_bf16 = precision == "bf16" and device.type == "cuda"
            with torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext():
                estimate = model(reference_image, image, network_delta, log_exposure).float()

        for suffix, intensity in (("", clean), ("_open", clean_open), ("_fitting", clean_fitting)):
            record(f"ee{suffix}", intensity[..., core].sum(-1))
            record(f"strehl{suffix}", intensity[:, half, half] / env.camera.peak0)
        record("residual_rms", env.basis.rms(residual))
        record("dm_residual_norm", label.norm(dim=-1))
        record("estimate_error_norm", (estimate - label).norm(dim=-1))
        record("estimate", estimate)
        record("label", label)
        record("command", command)
        record("delta_command", delta)
        record("network_delta", network_delta)
        eligible = label.abs() > 0.05
        correct = torch.sign(estimate) == torch.sign(label)
        for name, modes in (("odd", ~even_modes), ("even", even_modes)):
            record(f"{name}_correct", (correct & eligible & modes).sum(-1))
            record(f"{name}_count", (eligible & modes).sum(-1))
        record("counts_sum", counts_sum)
        record("counts_expected", photons * exposed.sum(dim=(-2, -1)))

        update = loop.gain * estimate
        if frame == 0 and 0 <= env.defocus_index < env.nmodes:
            update[:, env.defocus_index] += loop.probe_defocus_rad
        pending.append(update)
        previous_command = command
        if len(pending) >= loop.command_delay_frames:
            command = (command + pending.popleft()).clamp(-loop.command_limit_rad, loop.command_limit_rad)
        previous_image = image
        previous_counts = counts

    return {name: torch.stack(values, dim=1) for name, values in trace.items()}


def evaluate(
    model: torch.nn.Module,
    cfg: ClosedLoopConfig,
    experiment: Experiment,
    out_dir: str | Path,
    device: torch.device,
    *,
    methods: tuple[str, ...] = ("network",),
    precision: str = "bf16",
    tf32: bool = True,
    log=print,
) -> Path:
    """Run every condition of an experiment and write one trace file per condition and method.

    The traces are saved as out_dir/cell_XXX_<method>.npz, with one row per sequence and one column per frame.
    tf32 allows TF32 matrix products on the GPU (convolutions use TF32 in any case).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = True
    else:
        precision = "fp32"
    model = model.to(device).eval()
    references = {"network": cfg.reference.mode, "oracle": "oracle", "cramer_rao": "cramer_rao"}
    methods = {name: references[name] for name in methods}
    environments: dict[float, tuple[ClosedLoopEnvironment, CramerRaoReference | None]] = {}
    for index, cond in enumerate(experiment.conditions):
        if cond.gain not in environments:
            env = ClosedLoopEnvironment(replace(cfg, loop=replace(cfg.loop, gain=cond.gain)), device)
            environments[cond.gain] = (
                env,
                CramerRaoReference(env.camera) if "cramer_rao" in methods else None,
            )
        env, cramer_rao = environments[cond.gain]
        parts: dict[str, dict[str, list[np.ndarray]]] = {name: {} for name in methods}
        for chunk, begin in enumerate(range(0, experiment.sequences, experiment.batch)):
            n = min(experiment.batch, experiment.sequences - begin)
            dr0 = torch.full((n,), float(cond.d_over_r0), device=device)
            tau = torch.full((n,), float(cond.tau_ratio), device=device)
            photons = torch.full((n,), float(cond.photons), device=device)
            screens = seeded_generator(device, experiment.seed, cond.group, chunk, 11)
            trajectory = env.screens.episode(dr0, tau, experiment.frames, screens)
            noise_seed = mix_seed(experiment.seed, cond.group, chunk, 29)
            for name, reference in methods.items():
                trace = run_loop(
                    env,
                    model,
                    trajectory,
                    photons,
                    experiment.frames,
                    reference=reference,
                    noise_seed=noise_seed,
                    precision=precision,
                    cramer_rao=cramer_rao,
                    cramer_rao_seed=mix_seed(experiment.seed, cond.group, chunk, 419),
                )
                for key, value in trace.items():
                    parts[name].setdefault(key, []).append(value.cpu().numpy())
        for name, payload in parts.items():
            arrays = {key: np.concatenate(values, axis=0) for key, values in payload.items()}
            np.savez_compressed(out_dir / f"cell_{index:03d}_{name}.npz", **arrays)
        first = next(iter(methods))
        ee = np.concatenate(parts[first]["ee"], axis=0)[:, -15:].mean()
        log(f"cell {index + 1}/{len(experiment.conditions)}  {cond}  {first} steady EE {ee:.4f}")
    return out_dir


def static_shapes(count: int = 8, rms: float = 0.8, first_seed: int = 700) -> np.ndarray:
    """Coefficients (rad) of the static aberrations of Table 3 and Fig. 5(a), drawn by aosim."""
    from aosim import Simulator

    sim = Simulator(N=384, Dpix=44, nmodes=33)
    return np.stack(
        [sim.scale_to_rms(sim.turbulence_coeffs(4, seed=first_seed + s), rms) for s in range(count)]
    )


@torch.no_grad()
def evaluate_static(
    model: torch.nn.Module,
    cfg: ClosedLoopConfig,
    out_file: str | Path,
    device: torch.device,
    *,
    repeats: int = 16,
    frames: int = 11,
    photons: tuple[float, ...] = tuple(np.logspace(2, 6, 9)),
    read_noise: tuple[float, ...] = (0.0, 7.0),
    precision: str = "bf16",
    tf32: bool = False,
    log=print,
) -> Path:
    """Network trained on static aberrations, on the eight aberrations of Table 3 (Fig. 5(a)).

    For every aberration, photon level and read noise the loop runs repeats noise realizations with the network and
    with the true residual as estimate (same gain, delay and probe). The noise generator of aberration s is seeded
    by 2026102200 + s for every condition. The arrays in out_file have the shape
    (aberration, photons, read noise, realization, frame).
    """
    if cfg.reference.phase_mode != "static_modal_33":
        raise ValueError("evaluate_static expects the network trained on static aberrations")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = True
    else:
        precision = "fp32"
    model = model.to(device).eval()
    env = ClosedLoopEnvironment(cfg, device)

    def policy(previous, current, delta, log_exposure):
        use_bf16 = precision == "bf16" and device.type == "cuda"
        with torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext():
            return model(previous, current, delta, log_exposure).float()

    coefficients = static_shapes()
    keys = ("ee", "strehl", "residual_rms", "estimate_error_norm")
    shape = (len(coefficients), len(photons), len(read_noise), repeats, frames)
    out = {f"{m}_{k}": np.zeros(shape, dtype=np.float32) for m in ("network", "true_residual") for k in keys}

    def full(value):
        return torch.full((repeats,), float(value), device=device)

    for s, c in enumerate(coefficients):
        phase = env.basis.phase(torch.tensor(c, device=device, dtype=torch.float32)[None])
        env.screens = FixedShape(phase)
        for i, n in enumerate(photons):
            for j, sigma in enumerate(read_noise):
                for method in ("network", "true_residual"):
                    generator = torch.Generator(device=device).manual_seed(2026102200 + s)
                    truth = method == "true_residual"
                    _, st = env.rollout(
                        full(4),
                        full(0),
                        full(n),
                        frames,
                        generator,
                        None if truth else policy,
                        torch.full((repeats,), truth, device=device, dtype=torch.bool),
                        full(0),
                        collect=False,
                        static_rms=full(0.8),
                        read_noise_e=full(sigma),
                    )
                    for k in keys:
                        out[f"{method}_{k}"][s, i, j] = getattr(st, k).cpu().numpy()
        log(f"aberration {s + 1}/{len(coefficients)} done")
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_file,
        coefficients=coefficients,
        photons=np.asarray(photons),
        read_noise=np.asarray(read_noise),
        **out,
    )
    return out_file
