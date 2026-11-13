"""Zernike basis on the pupil grid and the far-field camera (batched, torch).

The pupil spans pupil_pixels grid points. The focal plane is sampled at fft_size / pupil_pixels pixels per
lambda/D, chosen to match lambda f / (D p) of the far-field camera at 1080 nm.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .config import CameraConfig, OpticsConfig


def noll_indices(count: int, start: int = 1) -> list[tuple[int, int, int]]:
    """(j, n, m) in Noll ordering for j = start ... start + count - 1."""
    out, j, n = [], 1, 0
    while len(out) < count + start - 1:
        for am in range(n % 2, n + 1, 2):
            if am == 0:
                out.append((j, n, 0))
                j += 1
            else:
                for _ in range(2):
                    out.append((j, n, am if j % 2 == 0 else -am))
                    j += 1
        n += 1
    return [t for t in out if t[0] >= start][:count]


def zernike_map(n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    am = abs(m)
    radial = np.zeros_like(rho)
    for k in range((n - am) // 2 + 1):
        c = (
            (-1) ** k
            * math.factorial(n - k)
            / (math.factorial(k) * math.factorial((n + am) // 2 - k) * math.factorial((n - am) // 2 - k))
        )
        radial += c * rho ** (n - 2 * k)
    norm = math.sqrt(n + 1) if m == 0 else math.sqrt(2 * (n + 1))
    if m > 0:
        return norm * radial * np.cos(am * theta)
    if m < 0:
        return norm * radial * np.sin(am * theta)
    return norm * radial


class PupilBasis:
    """Noll modes j = 1 ... 36 on a centrosymmetric grid; the DM corrects j = noll_start ... noll_start + nmodes - 1."""

    def __init__(self, optics: OpticsConfig, device: torch.device):
        p = optics.resolved_pupil_pixels
        self.pixels = p
        x = (np.arange(p) - (p - 1) / 2.0) / (p / 2.0)
        X, Y = np.meshgrid(x, x)
        r, theta = np.hypot(X, Y), np.arctan2(Y, X)
        mask = r <= 1.0
        rho = np.clip(r, 0.0, 1.0)
        total = optics.noll_start - 1 + optics.nmodes
        full = np.stack([zernike_map(n, m, rho, theta) * mask for (_, n, m) in noll_indices(total, 1)])
        self.mask = torch.from_numpy(mask).to(device)
        self.amplitude = self.mask.float()
        self.npupil = int(mask.sum())
        self.dm_slice = slice(optics.noll_start - 1, total)
        self.full = torch.from_numpy(full).float().to(device)  # (36, p, p)
        self.dm = self.full[self.dm_slice].contiguous()  # (33, p, p)
        z = torch.from_numpy(full[:, mask]).double()  # (36, npupil)
        self.projector = torch.linalg.pinv(z.T).float().to(device)  # (36, npupil)
        self.defocus = self.full[3]

    def phase(self, dm_coeffs: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bm,mhw->bhw", dm_coeffs, self.dm)

    def project(self, phase: torch.Tensor) -> torch.Tensor:
        """Least-squares coefficients of Noll modes 1 ... 36 for (B, p, p) phase maps."""
        return phase[:, self.mask] @ self.projector.T

    def remove_low_order(self, phase: torch.Tensor, coeffs: torch.Tensor | None = None) -> torch.Tensor:
        """Remove piston, tip and tilt (an ideal steering mirror)."""
        coeffs = self.project(phase) if coeffs is None else coeffs
        return phase - torch.einsum("bm,mhw->bhw", coeffs[:, :3], self.full[:3])

    def rms(self, phase: torch.Tensor) -> torch.Tensor:
        values = phase[:, self.mask]
        return (values - values.mean(1, keepdim=True)).pow(2).mean(1).sqrt()


class FarFieldCamera:
    """Far-field image of the residual with Poisson noise and Gaussian read noise."""

    def __init__(self, optics: OpticsConfig, camera: CameraConfig, basis: PupilBasis, device: torch.device):
        self.optics, self.camera, self.basis = optics, camera, basis
        self.fft_size = optics.fft_size
        self.half = int(round(optics.roi_half_lambda_over_d * optics.pixels_per_lambda_over_d))
        self.out_size = optics.network_image_size
        self.diversity = optics.diversity_defocus_rad * basis.defocus
        flat = self.intensity(torch.zeros(1, basis.pixels, basis.pixels, device=device), diversity=False)
        self.peak0 = float(flat.max())  # diffraction-limited peak, fraction of the photons per pixel

    def intensity(self, residual: torch.Tensor, diversity: bool = True) -> torch.Tensor:
        """Noise-free image in the camera window, normalized so that the whole focal plane sums to one."""
        p, n = self.basis.pixels, self.fft_size
        phase = residual + self.diversity if diversity else residual
        field = self.basis.amplitude * torch.exp(1j * phase)
        pad = n - p
        field = F.pad(field, (0, pad, 0, pad))
        image = torch.fft.fftshift(torch.fft.fft2(field), dim=(-2, -1)).abs().pow(2)
        image = image / float(self.basis.npupil * n * n)
        c = n // 2
        return image[:, c - self.half : c + self.half, c - self.half : c + self.half]

    def strehl(self, residual: torch.Tensor) -> torch.Tensor:
        """Center pixel of the image of the residual divided by the diffraction-limited peak."""
        image = self.intensity(residual, diversity=False)
        return image[:, self.half, self.half] / self.peak0

    def encircled_energy(self, residual: torch.Tensor, diversity: bool = False) -> torch.Tensor:
        """Fraction of the energy within 1.22 lambda/D of the axis (0.838 for a flat wavefront)."""
        image = self.intensity(residual, diversity=diversity)
        half = self.half
        y, x = torch.meshgrid(
            torch.arange(2 * half, device=image.device, dtype=torch.float32) - half,
            torch.arange(2 * half, device=image.device, dtype=torch.float32) - half,
            indexing="ij",
        )
        mask = torch.hypot(y, x) <= 1.22 * self.optics.pixels_per_lambda_over_d
        return image[..., mask].sum(-1)

    def expose(
        self,
        intensity: torch.Tensor,
        photons: torch.Tensor,
        generator: torch.Generator,
        *,
        read_noise_e: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Detected electrons, Poisson counts plus Gaussian read noise, without clipping."""
        counts = torch.poisson(intensity * photons.view(-1, 1, 1), generator=generator)
        sigma = self.camera.read_noise_e
        if read_noise_e is not None:
            sigma = torch.as_tensor(read_noise_e, device=counts.device, dtype=counts.dtype)
            if sigma.ndim == 1:
                sigma = sigma[:, None, None]
        noise = torch.randn(counts.shape, generator=generator, device=counts.device) * sigma
        return counts + noise

    def network_input(self, counts: torch.Tensor) -> torch.Tensor:
        """Normalize by the measured frame sum and the diffraction-limited peak, compress, resample to the network grid."""
        total = counts.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        scale = self.camera.asinh_scale
        x = torch.asinh(counts / (total * self.peak0 * scale)) / math.asinh(1.0 / scale)
        return F.interpolate(x.unsqueeze(1), size=(self.out_size, self.out_size), mode="area").squeeze(1)
