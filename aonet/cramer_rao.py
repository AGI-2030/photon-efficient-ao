"""Cramer-Rao reference of Figs. 3(b) and 3(c) (Section 3.4 and Note S5 of the paper).

At every frame the reference takes the true residual y of the 33 modes and adds Gaussian noise with the inverse
of the Fisher matrix of one frame, J^T diag(1 / (mean + read noise^2)) J, evaluated at y, with the noise limited
to 1 rad RMS over the pupil. The even sign is then decided between y + noise and its twin by the likelihood of the
previous raw frame, predicted for each candidate plus the last correction difference. At the first frame the even
sign is random. The reference uses the true residual and is neither an image-only estimator nor a bound.

The model, the Fisher matrix and the likelihood are computed in float64. Higher orders and the evolution of the
turbulence are present in the frames but absent from the model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .optics import FarFieldCamera, noll_indices


class CramerRaoReference:
    def __init__(
        self,
        camera: FarFieldCamera,
        ridge: float = 1e-9,
        noise_rms_cap_rad: float = 1.0,
        mode_chunk: int = 4,
        variance_floor: float = 1e-12,
    ):
        self.camera, self.basis = camera, camera.basis
        self.ridge, self.cap, self.chunk, self.floor = ridge, noise_rms_cap_rad, mode_chunk, variance_floor
        self.modes = self.basis.dm.double()
        self.amplitude = self.basis.amplitude.double()
        self.even = torch.tensor(
            [abs(m) % 2 == 0 for _, _, m in noll_indices(33, 4)], device=self.modes.device
        )
        self.twin_sign = torch.where(self.even, -1.0, 1.0).double()
        values = self.modes[:, self.basis.mask]
        centered = values - values.mean(1, keepdim=True)
        self.phase_gram = centered @ centered.T / self.basis.npupil
        self.normalizer = float(self.basis.npupil * camera.fft_size**2)

    def _crop(self, value):
        center, half = self.camera.fft_size // 2, self.camera.half
        return value[..., center - half : center + half, center - half : center + half]

    def _field(self, coefficients):
        phase = torch.einsum("bm,mhw->bhw", coefficients.double(), self.modes)
        pupil = self.amplitude * torch.exp(1j * phase)
        pad = self.camera.fft_size - self.basis.pixels
        field = torch.fft.fftshift(torch.fft.fft2(F.pad(pupil, (0, pad, 0, pad))), dim=(-2, -1))
        return pupil, field

    def expected_counts(self, coefficients, photons):
        _, field = self._field(coefficients)
        return self._crop(field).abs().square() / self.normalizer * photons.double()[:, None, None]

    def counts_and_jacobian(self, coefficients, photons):
        """Mean counts in the camera window (B, H, W) and their derivatives per mode (B, 33, H, W)."""
        pupil, field = self._field(coefficients)
        window = self._crop(field)
        flux = photons.double()[:, None, None]
        mean = window.abs().square() / self.normalizer * flux
        pad = self.camera.fft_size - self.basis.pixels
        rows = []
        for first in range(0, 33, self.chunk):
            product = pupil[:, None] * self.modes[None, first : first + self.chunk]
            derivative_field = torch.fft.fftshift(
                torch.fft.fft2(F.pad(product, (0, pad, 0, pad))), dim=(-2, -1)
            )
            derivative = -2 * (window[:, None].conj() * self._crop(derivative_field)).imag
            rows.append(derivative / self.normalizer * flux[:, None])
        return mean, torch.cat(rows, dim=1)

    def fisher(self, residual, photons):
        mean, jacobian = self.counts_and_jacobian(residual, photons)
        variance = (mean + self.camera.camera.read_noise_e**2).clamp_min(self.floor)
        weighted = jacobian.flatten(2) / variance.flatten(1).sqrt()[:, None]
        return weighted @ weighted.transpose(1, 2)

    def phase_rms(self, coefficients):
        c = coefficients.double()
        return torch.einsum("bm,mn,bn->b", c, self.phase_gram, c).clamp_min(0).sqrt()

    def sample_noise(self, fisher, generator):
        """Draw from the inverse of (fisher + ridge I) and scale draws above the RMS limit down to it."""
        eigenvalues, vectors = torch.linalg.eigh(0.5 * (fisher.double() + fisher.double().transpose(1, 2)))
        precision = eigenvalues.clamp_min(0) + self.ridge
        standard = torch.randn(
            eigenvalues.shape, generator=generator, dtype=torch.float64, device=eigenvalues.device
        )
        noise = (vectors @ (standard / precision.sqrt()).unsqueeze(-1)).squeeze(-1)
        scale = (self.cap / self.phase_rms(noise).clamp_min(1e-300)).clamp_max(1.0)
        return noise * scale[:, None]

    def log_likelihood(self, candidate, previous_counts, photons, delta):
        mean = self.expected_counts(candidate.double() + delta.double(), photons)
        variance = (mean + self.camera.camera.read_noise_e**2).clamp_min(self.floor)
        return -0.5 * (((previous_counts.double() - mean).square() / variance) + variance.log()).flatten(
            1
        ).sum(1)

    @torch.no_grad()
    def estimate(self, residual, photons, previous_counts, delta, generator):
        noise = self.sample_noise(self.fisher(residual, photons), generator)
        truth = residual.double()
        if previous_counts is None:
            u = torch.rand(truth.shape[0], device=truth.device, generator=generator)
            sign = torch.where(u < 0.5, 1.0, -1.0).double()
            estimate = torch.where(self.even, truth * sign[:, None], truth) + noise
        else:
            base = truth + noise
            twin = base * self.twin_sign
            margin = self.log_likelihood(twin, previous_counts, photons, delta) - self.log_likelihood(
                base, previous_counts, photons, delta
            )
            estimate = torch.where((margin > 0)[:, None], twin, base)
        return estimate.to(residual.dtype)
