"""Static aberrations in the 33 corrected modes, scaled once to a chosen RMS.

A von Karman screen sets the shape of the aberration. Every frame returns the same phase, so only the
correction changes the residual.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .optics import PupilBasis
from .turbulence import FrozenFlowScreens


@dataclass(frozen=True)
class StaticTrajectory:
    phase: torch.Tensor
    coefficients: torch.Tensor
    target_rms: torch.Tensor
    frames: int

    def frame(self, k: int) -> torch.Tensor:
        return self.phase


class FixedShape:
    """Stands in for FrozenFlowScreens and returns one given phase map (1, p, p) for every sequence.

    make_static_trajectory then projects it on the 33 modes and scales it to the requested RMS.
    """

    def __init__(self, phase: torch.Tensor):
        self.phase = phase

    def episode(self, d_over_r0, tau_ratio, frames, generator) -> StaticTrajectory:
        return StaticTrajectory(self.phase.expand(len(d_over_r0), -1, -1), None, None, frames)


@torch.no_grad()
def make_static_trajectory(
    basis: PupilBasis,
    screens: FrozenFlowScreens,
    batch: int,
    frames: int,
    generator: torch.Generator,
    rms_range: tuple[float, float],
    shape_d_over_r0: float,
    rms_override: float | torch.Tensor | None = None,
) -> StaticTrajectory:
    """Draw a von Karman shape, keep its Z4 to Z36 part and scale its RMS over the pupil.

    The RMS is sqrt(mean(phi^2)) over the pupil pixels. rms_override, a scalar or one value per sequence,
    replaces the random RMS (used by the validation grid).
    """
    device = basis.dm.device
    shape_strength = torch.full((batch,), float(shape_d_over_r0), device=device)
    source = screens.episode(shape_strength, torch.zeros(batch, device=device), 1, generator).frame(0)
    coefficients = basis.project(source)[:, basis.dm_slice]
    phase = basis.phase(coefficients)
    current_rms = phase[:, basis.mask].square().mean(-1).sqrt()
    if rms_override is None:
        lo, hi = rms_range
        target_rms = lo + (hi - lo) * torch.rand(batch, generator=generator, device=device)
    else:
        target_rms = torch.as_tensor(rms_override, device=device, dtype=phase.dtype)
        if target_rms.ndim == 0:
            target_rms = target_rms.expand(batch).clone()
    coefficients = coefficients * (target_rms / current_rms).unsqueeze(1)
    phase = basis.phase(coefficients)
    return StaticTrajectory(phase, coefficients, target_rms, frames)
