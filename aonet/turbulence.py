"""Batched von Karman screens with frozen-flow translation (FFT method with three subharmonic levels, Schmidt 2010).

The spectrum and units are those of aosim.Simulator.vk_screen: D = 1, r0 and the scales in units of D, phase in
rad at 1080 nm. A screen with D/r0 = 1 is drawn and scaled by (D/r0)^(5/6), which is exact because the outer and
inner scales are fixed in units of D.

Frozen flow: the pupil is read from the screen at positions translated by v tau_loop per frame along a random
direction. With tau_0 = 0.314 r0 / v, the translation is 0.314 (r0 in pixels) tau_loop / tau_0. The subharmonics
are not periodic, so the trajectory stays inside the screen.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import TurbulenceConfig

GREENWOOD = 0.314


class FrozenFlowScreens:
    def __init__(self, turbulence: TurbulenceConfig, pupil_pixels: int, device: torch.device):
        self.cfg, self.p, self.device = turbulence, pupil_pixels, device
        n = turbulence.screen_size
        delta = 1.0 / pupil_pixels
        fm = 5.92 / turbulence.inner_scale_over_d / (2 * math.pi)
        f0 = 1.0 / turbulence.outer_scale_over_d
        del_f = 1.0 / (n * delta)
        fx = (torch.arange(n, device=device, dtype=torch.float64) - n // 2) * del_f
        FX, FY = torch.meshgrid(fx, fx, indexing="xy")
        f = torch.hypot(FX, FY)
        psd = 0.023 * torch.exp(-((f / fm) ** 2)) / (f**2 + f0**2) ** (11 / 6)
        psd[n // 2, n // 2] = 0.0
        self.amp_hi = (psd.sqrt() * del_f).float()  # r0 = D
        self.subharmonic_amp, self.subharmonic_basis = [], []
        if turbulence.subharmonics:
            xs = (torch.arange(n, device=device, dtype=torch.float64) - n // 2) * delta
            X, Y = torch.meshgrid(xs, xs, indexing="xy")
            for level in (1, 2, 3):
                dfp = 1.0 / (3**level * n * delta)
                fp = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=torch.float64) * dfp
                FXp, FYp = torch.meshgrid(fp, fp, indexing="xy")
                fpp = torch.hypot(FXp, FYp)
                psdp = 0.023 * torch.exp(-((fpp / fm) ** 2)) / (fpp**2 + f0**2) ** (11 / 6)
                psdp[1, 1] = 0.0
                self.subharmonic_amp.append((psdp.sqrt() * dfp).flatten().float())  # (9,)
                basis = torch.exp(
                    2j * math.pi * (FXp.flatten()[:, None, None] * X + FYp.flatten()[:, None, None] * Y)
                )
                self.subharmonic_basis.append(basis.to(torch.complex64))  # (9, n, n)

    def unit_screens(self, batch: int, generator: torch.Generator) -> torch.Tensor:
        """(B, n, n) screens with r0 = D."""
        n = self.cfg.screen_size
        shape = (batch, n, n)
        noise = torch.complex(
            torch.randn(shape, generator=generator, device=self.device),
            torch.randn(shape, generator=generator, device=self.device),
        )
        cn = noise * self.amp_hi
        screen = (
            torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(cn, dim=(-2, -1))), dim=(-2, -1)).real
            * n
            * n
        )
        for amp, basis in zip(self.subharmonic_amp, self.subharmonic_basis):
            c = (
                torch.complex(
                    torch.randn((batch, 9), generator=generator, device=self.device),
                    torch.randn((batch, 9), generator=generator, device=self.device),
                )
                * amp
            )
            low = torch.einsum("bk,khw->bhw", c, basis).real
            screen = screen + low - low.mean(dim=(-2, -1), keepdim=True)
        return screen

    def episode(
        self, d_over_r0: torch.Tensor, tau_ratio: torch.Tensor, frames: int, generator: torch.Generator
    ) -> "Trajectory":
        batch = d_over_r0.numel()
        screens = self.unit_screens(batch, generator) * d_over_r0.view(-1, 1, 1) ** (5.0 / 6.0)
        step = GREENWOOD * (self.p / d_over_r0) * tau_ratio  # pixels per frame
        angle = torch.rand(batch, generator=generator, device=self.device) * (2 * math.pi)
        direction = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)  # (B, 2) as (x, y)
        n = self.cfg.screen_size
        travel = (frames - 1) * step
        center = (n - 1) / 2.0
        start = center - 0.5 * travel.view(-1, 1) * direction
        return Trajectory(screens, start, step.view(-1, 1) * direction, self.p)


class Trajectory:
    """Pupil-sized phase maps along a frozen-flow trajectory (bilinear subpixel translation)."""

    def __init__(self, screens: torch.Tensor, start: torch.Tensor, velocity: torch.Tensor, pupil_pixels: int):
        self.screens = screens.unsqueeze(1)  # (B, 1, n, n)
        self.start, self.velocity = start, velocity
        n = screens.shape[-1]
        offs = (
            torch.arange(pupil_pixels, device=screens.device, dtype=torch.float32) - (pupil_pixels - 1) / 2.0
        )
        oy, ox = torch.meshgrid(offs, offs, indexing="ij")
        self.offsets = torch.stack([ox, oy], dim=-1)  # (p, p, 2)
        self.scale = 2.0 / (n - 1)

    def frame(self, k: int) -> torch.Tensor:
        center = self.start + k * self.velocity  # (B, 2)
        coords = center[:, None, None, :] + self.offsets[None]  # (B, p, p, 2), pixels
        grid = coords * self.scale - 1.0
        return F.grid_sample(
            self.screens, grid, mode="bilinear", padding_mode="border", align_corners=True
        ).squeeze(1)
