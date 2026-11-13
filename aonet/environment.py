"""Simulated closed loop used for training: turbulence, deformable mirror, far-field camera and integrator.

Frame k:  phi_k   turbulence (piston, tip and tilt removed by an ideal steering mirror)
          c_k     DM command (Noll 4 to 36, rad), residual r_k = phi_k - Z c_k, image I_k of r_k
          label   y_k, least-squares DM-mode content of r_k (the fitting error stays in r_k)
          input   reference image, I_k, command difference and log exposure; at k = 0 the reference is I_0
          update  c_{k+d} = c_{k+d-1} + g a_k, with a_k the estimate of y_k and d the command delay
The first update adds a known defocus, so the second frame carries an even probe.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable

import torch

from .config import ClosedLoopConfig
from .optics import FarFieldCamera, PupilBasis
from .static_scene import make_static_trajectory
from .turbulence import FrozenFlowScreens

Policy = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class EpisodeBatch:
    previous: torch.Tensor  # (N, H, W) float16, reference image
    current: torch.Tensor  # (N, H, W) float16
    delta_command: torch.Tensor  # (N, 33)
    log_exposure: torch.Tensor  # (N,)
    label: torch.Tensor  # (N, 33)


@dataclass
class EpisodeStats:
    strehl: torch.Tensor  # (B, K) Strehl ratio of the residual
    strehl_open: torch.Tensor  # (B, K) uncorrected (tip and tilt removed)
    residual_rms: torch.Tensor  # (B, K) rad, including the fitting error
    dm_residual_norm: torch.Tensor  # (B, K) rad, norm of the 33 residual coefficients
    estimate_error_norm: torch.Tensor  # (B, K) rad, norm of a_k - y_k
    ee: torch.Tensor  # (B, K) encircled energy within 1.22 lambda/D
    action: torch.Tensor  # (B, K, 33) estimate a_k
    label: torch.Tensor  # (B, K, 33) true residual y_k


class ClosedLoopEnvironment:
    def __init__(self, config: ClosedLoopConfig, device: torch.device):
        self.cfg, self.device = config, device
        self.basis = PupilBasis(config.optics, device)
        self.camera = FarFieldCamera(config.optics, config.camera, self.basis, device)
        self.screens = FrozenFlowScreens(config.turbulence, self.basis.pixels, device)
        self.nmodes = config.optics.nmodes
        self.defocus_index = 4 - config.optics.noll_start

    def sample_conditions(
        self,
        batch: int,
        generator: torch.Generator,
        d_over_r0_max: float | None = None,
        tau_ratio_max: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Turbulence strength (uniform), loop period and photons (log-uniform) of a batch of sequences."""
        t, cam = self.cfg.turbulence, self.cfg.camera
        if self.cfg.reference.phase_mode == "static_modal_33":
            # D/r0 only sets the shape of the static aberration
            d_over_r0 = torch.full((batch,), self.cfg.reference.static_shape_d_over_r0, device=self.device)
            tau = torch.zeros(batch, device=self.device)
            u = torch.rand(batch, generator=generator, device=self.device)
            plo, phi = cam.photons_range
            photons = torch.exp(math.log(plo) + (math.log(phi) - math.log(plo)) * u)
            return d_over_r0, tau, photons
        lo, hi = t.d_over_r0_range
        hi = hi if d_over_r0_max is None else min(hi, max(lo, d_over_r0_max))
        tlo, thi = t.tau_ratio_range
        thi = thi if tau_ratio_max is None else min(thi, max(tlo, tau_ratio_max))
        u = torch.rand((3, batch), generator=generator, device=self.device)
        d_over_r0 = lo + (hi - lo) * u[0]
        tau = torch.exp(math.log(tlo) + (math.log(thi) - math.log(tlo)) * u[1])
        plo, phi = cam.photons_range
        photons = torch.exp(math.log(plo) + (math.log(phi) - math.log(plo)) * u[2])
        return d_over_r0, tau, photons

    @torch.no_grad()
    def rollout(
        self,
        d_over_r0: torch.Tensor,
        tau_ratio: torch.Tensor,
        photons: torch.Tensor,
        frames: int,
        generator: torch.Generator,
        policy: Policy | None,
        expert_mask: torch.Tensor,
        expert_noise_rad: torch.Tensor,
        collect: bool = True,
        *,
        static_rms: float | torch.Tensor | None = None,
        read_noise_e: float | torch.Tensor | None = None,
    ) -> tuple[EpisodeBatch | None, EpisodeStats]:
        """Run a batch of sequences.

        expert_mask (B,) marks the sequences driven by the true residual plus Gaussian noise of standard
        deviation expert_noise_rad (B,) per mode; the other sequences are driven by the policy.
        """
        lp = self.cfg.loop
        batch = d_over_r0.numel()
        reference = self.cfg.reference
        if reference.phase_mode == "static_modal_33":
            traj = make_static_trajectory(
                self.basis,
                self.screens,
                batch,
                frames,
                generator,
                reference.static_rms_range,
                reference.static_shape_d_over_r0,
                rms_override=static_rms,
            )
        else:
            traj = self.screens.episode(d_over_r0, tau_ratio, frames, generator)
        # one read-noise level per sequence, kept for all of its frames
        if read_noise_e is None and reference.read_noise_choices_e:
            choices = torch.tensor(reference.read_noise_choices_e, device=self.device)
            indices = torch.randint(len(choices), (batch,), generator=generator, device=self.device)
            read_noise_e = choices[indices]
        c = torch.zeros(batch, self.nmodes, device=self.device)
        c_prev = c.clone()
        c_initial = c.clone()
        pending: deque[torch.Tensor] = deque()
        prev_img = initial_img = None
        store = {k: [] for k in ("previous", "current", "delta_command", "log_exposure", "label")}
        stats = {k: [] for k in EpisodeStats.__dataclass_fields__}
        limit = lp.command_limit_rad
        for k in range(frames):
            phi = traj.frame(k)
            coeffs = self.basis.project(phi)
            phi = self.basis.remove_low_order(phi, coeffs)
            label = coeffs[:, self.basis.dm_slice] - c
            residual = phi - self.basis.phase(c)
            intensity = self.camera.intensity(residual)
            if read_noise_e is None:
                counts = self.camera.expose(intensity, photons, generator)
            else:
                counts = self.camera.expose(intensity, photons, generator, read_noise_e=read_noise_e)
            image = self.camera.network_input(counts)
            log_exposure = torch.log10(counts.sum(dim=(-2, -1)).clamp_min(1.0)) - 4.0
            if initial_img is None:
                initial_img = image
            if reference.mode == "initial":
                previous = initial_img
                delta = c - c_initial
            else:
                previous = image if prev_img is None else prev_img
                delta = c - c_prev
            action = label + expert_noise_rad.view(-1, 1) * torch.randn(
                label.shape, generator=generator, device=self.device
            )
            if policy is not None and not bool(expert_mask.all()):
                estimate = policy(previous, image, delta, log_exposure).float()
                action = torch.where(expert_mask.view(-1, 1), action, estimate)
            if collect:
                store["previous"].append(previous.half())
                store["current"].append(image.half())
                store["delta_command"].append(delta)
                store["log_exposure"].append(log_exposure)
                store["label"].append(label)
            stats["strehl"].append(self.camera.strehl(residual))
            stats["strehl_open"].append(self.camera.strehl(phi))
            stats["residual_rms"].append(self.basis.rms(residual))
            stats["dm_residual_norm"].append(label.norm(dim=-1))
            stats["estimate_error_norm"].append((action - label).norm(dim=-1))
            stats["ee"].append(self.camera.encircled_energy(residual))
            stats["action"].append(action)
            stats["label"].append(label)
            update = lp.gain * action
            if k == 0 and 0 <= self.defocus_index < self.nmodes:
                update[:, self.defocus_index] += lp.probe_defocus_rad
            pending.append(update)
            c_prev = c
            if len(pending) >= lp.command_delay_frames:
                c = (c + pending.popleft()).clamp(-limit, limit)
            prev_img = image
        episode = EpisodeBatch(**{k: torch.cat(v) for k, v in store.items()}) if collect else None
        return episode, EpisodeStats(**{k: torch.stack(v, dim=1) for k, v in stats.items()})


def sign_accuracy(
    estimate: torch.Tensor, label: torch.Tensor, azimuthal_order: torch.Tensor, threshold: float = 0.05
) -> torch.Tensor:
    """(B, K) fraction of the even modes with |label| > threshold whose estimate has the correct sign."""
    even = azimuthal_order.abs() % 2 == 0
    mask = even & (label.abs() > threshold)
    correct = (torch.sign(estimate) == torch.sign(label)) & mask
    return correct.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
