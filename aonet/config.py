"""Configuration of the simulated loop, the networks and the DAgger training.

A configuration file (configs/*.toml) has the top-level keys name and seed, one table per dataclass below, and a
[run] table with the input configuration and the batch settings used for the released weights. Weight files store
the same structure, without [run], as a dictionary.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


@dataclass(frozen=True)
class OpticsConfig:
    wavelength_m: float = 1.08e-6
    focal_length_m: float = 5.8
    pixel_pitch_m: float = 5.86e-6
    exit_aperture_m: float = 0.1
    fft_size: int = 512
    pupil_pixels: int = 0  # 0: derived from fft_size and lambda f / (D p)
    roi_half_lambda_over_d: float = 10.0
    network_image_size: int = 128
    diversity_defocus_rad: float = 0.0  # static defocus in the camera arm (0 = none)
    noll_start: int = 4
    nmodes: int = 33

    @property
    def target_pixels_per_lambda_over_d(self) -> float:
        return self.wavelength_m * self.focal_length_m / (self.exit_aperture_m * self.pixel_pitch_m)

    @property
    def resolved_pupil_pixels(self) -> int:
        if self.pupil_pixels > 0:
            return self.pupil_pixels
        return int(round(self.fft_size / self.target_pixels_per_lambda_over_d))

    @property
    def pixels_per_lambda_over_d(self) -> float:
        return self.fft_size / self.resolved_pupil_pixels


@dataclass(frozen=True)
class TurbulenceConfig:
    screen_size: int = 512
    outer_scale_over_d: float = 10.0
    inner_scale_over_d: float = 0.01
    subharmonics: bool = True
    d_over_r0_range: tuple[float, float] = (1.0, 8.0)  # at 1080 nm
    tau_ratio_range: tuple[float, float] = (0.03, 1.0)  # tau_loop / tau_0, log-uniform


@dataclass(frozen=True)
class CameraConfig:
    photons_range: tuple[float, float] = (1.0e3, 1.0e6)  # detected photons per frame, log-uniform
    read_noise_e: float = 7.0
    asinh_scale: float = 0.02  # input compression, relative to the diffraction-limited peak


@dataclass(frozen=True)
class ReferenceConfig:
    """Reference frame (previous or initial) and scene (frozen-flow turbulence or static 33-mode aberrations)."""

    mode: str = "previous"  # previous | initial
    phase_mode: str = "frozen_flow"  # frozen_flow | static_modal_33
    static_rms_range: tuple[float, float] = (0.4, 1.2)  # RMS of the static phase, rad
    static_shape_d_over_r0: float = 4.0  # strength of the screen that sets the static shape
    read_noise_choices_e: tuple[float, ...] = ()  # empty: camera.read_noise_e; otherwise drawn per sequence
    static_validation_rms: tuple[float, ...] = (0.4, 0.8, 1.2)
    static_validation_read_noise_e: tuple[float, ...] = (0.0, 7.0)
    static_validation_photons: tuple[float, ...] = (100.0, 1000.0, 10000.0, 100000.0, 1000000.0)

    def validate(self) -> None:
        if self.mode not in ("previous", "initial"):
            raise ValueError("reference.mode must be previous or initial")
        if self.phase_mode not in ("frozen_flow", "static_modal_33"):
            raise ValueError("reference.phase_mode must be frozen_flow or static_modal_33")
        if self.phase_mode == "static_modal_33" and self.mode != "initial":
            raise ValueError("static aberrations use the initial frame as reference")
        lo, hi = self.static_rms_range
        if not 0 < lo <= hi:
            raise ValueError("static_rms_range must be an ordered pair of positive values")


@dataclass(frozen=True)
class LoopConfig:
    gain: float = 0.5
    command_delay_frames: int = 1  # the estimate from frame k is applied from frame k + 1
    frames_per_episode: int = 20
    probe_defocus_rad: float = 0.2  # defocus added to the first command
    command_limit_rad: float = 3.0  # per-mode command limit


@dataclass(frozen=True)
class DaggerConfig:
    iterations: int = 2000
    episodes_per_iteration: int = 128
    rollout_batch: int = 128
    updates_per_iteration: int = 40
    batch_size: int = 128
    buffer_capacity: int = 60_000  # frames, stored in float16
    warmup_iterations: int = 100  # rollouts driven by the noisy true residual only
    warmup_oracle_noise_rad: float = 0.3
    oracle_noise_rad: float = 0.05
    beta_start: float = 0.8
    beta_end: float = 0.0
    beta_decay_iterations: int = 800
    curriculum_iterations: int = 600
    curriculum_start_d_over_r0_max: float = 3.0
    curriculum_start_tau_ratio_max: float = 0.3


@dataclass(frozen=True)
class ModelConfig:
    name: str = "film_resnet"  # film_resnet | plain_cnn | fno | vit
    widths: tuple[int, ...] = (32, 64, 128, 256)
    blocks: tuple[int, ...] = (2, 2, 2, 2)
    condition_dim: int = 128
    groups: int = 8


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    warmup_updates: int = 1000
    gradient_clip_norm: float = 1.0
    loss_floor_rad: float = 0.05  # per-mode floor of the loss denominator
    final_lr_fraction: float = 0.05
    device: str = "cuda"
    precision: str = "bf16"  # bf16 | fp32


@dataclass(frozen=True)
class EvaluationConfig:
    """Validation grid used to select the checkpoint during training."""

    interval: int = 50
    checkpoint_interval: int = 50
    seed: int = 900_000
    episodes_per_cell: int = 64
    frames_per_episode: int = 30
    steady_frames: int = 10
    d_over_r0: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0)
    tau_ratio: tuple[float, ...] = (0.1, 0.3, 1.0)
    photons: tuple[float, ...] = (1.0e4, 1.0e6)


@dataclass(frozen=True)
class ClosedLoopConfig:
    name: str = "closed_loop_dagger"
    seed: int = 20261020
    optics: OpticsConfig = field(default_factory=OpticsConfig)
    turbulence: TurbulenceConfig = field(default_factory=TurbulenceConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    dagger: DaggerConfig = field(default_factory=DaggerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        o, t, lp, ev = self.optics, self.turbulence, self.loop, self.evaluation
        self.reference.validate()
        p = o.resolved_pupil_pixels
        if o.nmodes + o.noll_start - 1 > 36:
            raise ValueError("the basis is limited to Noll j <= 36")
        if 2 * round(o.roi_half_lambda_over_d * o.pixels_per_lambda_over_d) > o.fft_size:
            raise ValueError("the camera window is larger than the focal plane")
        if lp.command_delay_frames < 1:
            raise ValueError("command_delay_frames must be >= 1")
        if self.reference.phase_mode == "static_modal_33":
            if (o.noll_start, o.nmodes) != (4, 33):
                raise ValueError("static aberrations use the modes Z4 to Z36")
            return
        # a frozen-flow trajectory must stay inside one screen (the subharmonics are not periodic)
        for label, frames, dr0_min, ratio_max in (
            ("training", lp.frames_per_episode, t.d_over_r0_range[0], t.tau_ratio_range[1]),
            ("evaluation", ev.frames_per_episode, min(ev.d_over_r0), max(ev.tau_ratio)),
        ):
            travel = (frames - 1) * 0.314 * p / dr0_min * ratio_max
            if travel + math.sqrt(2) * p + 4 > t.screen_size:
                raise ValueError(
                    f"{label}: a travel of {travel:.0f} px does not fit in screen_size {t.screen_size}"
                )


SECTIONS = {
    "optics": OpticsConfig,
    "turbulence": TurbulenceConfig,
    "camera": CameraConfig,
    "reference": ReferenceConfig,
    "loop": LoopConfig,
    "dagger": DaggerConfig,
    "model": ModelConfig,
    "training": TrainingConfig,
    "evaluation": EvaluationConfig,
}


def _section(cls, raw: dict[str, Any]):
    unknown = set(raw) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})


def config_from_dict(raw: dict[str, Any]) -> ClosedLoopConfig:
    """Build a configuration from a dictionary (a TOML file or the config stored in a checkpoint)."""
    config = ClosedLoopConfig(
        name=raw.get("name", "closed_loop_dagger"),
        seed=int(raw.get("seed", 20261020)),
        **{key: _section(cls, raw.get(key, {})) for key, cls in SECTIONS.items()},
    )
    config.validate()
    return config


def load_config(path: str | Path) -> ClosedLoopConfig:
    with open(path, "rb") as handle:
        return config_from_dict(tomllib.load(handle))


def load_run_settings(path: str | Path) -> dict[str, Any]:
    """The [run] table of a configuration file.

    inputs is full, two_frames or single_frame. rollout_batch and micro_batch default to dagger.rollout_batch
    and dagger.batch_size; max_iterations, when present, is the number of iterations after which training stopped.
    """
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)
    run = {"inputs": "full", "rollout_batch": None, "micro_batch": None, "max_iterations": None}
    unknown = set(raw.get("run", {})) - set(run)
    if unknown:
        raise ValueError(f"unknown keys in [run]: {sorted(unknown)}")
    run.update(raw.get("run", {}))
    return run
