"""Scalar Fraunhofer model of the image-based loop.

Zernike basis (Noll), far-field images, the twin transform, discriminability per photon, von Karman
phase screens and the ideal-observer test. Units: pupil diameter D = 1, phase in radians, focal plane
in lambda/D. The grid is symmetric under point reflection, so the twin transform
T(phi)(x) = -phi(-x) is exact on the discrete grid.
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from math import factorial, sqrt, pi


def noll_indices(nmodes, start=2):
    """(j, n, m) triples in Noll ordering for j = start ... start+nmodes-1 (j=1 is piston)."""
    out, j, n = [], 1, 0
    while len(out) < nmodes + (start - 1):
        for am in range(n % 2, n + 1, 2):
            if am == 0:
                out.append((j, n, 0))
                j += 1
            else:
                for _ in range(2):
                    m = am if (j % 2 == 0) else -am
                    out.append((j, n, m))
                    j += 1
        n += 1
    return [t for t in out if t[0] >= start][:nmodes]


def radial_poly(n, am, rho):
    R = np.zeros_like(rho)
    for k in range((n - am) // 2 + 1):
        c = (
            (-1) ** k
            * factorial(n - k)
            / (factorial(k) * factorial((n + am) // 2 - k) * factorial((n - am) // 2 - k))
        )
        R += c * rho ** (n - 2 * k)
    return R


def zernike(n, m, rho, theta):
    am = abs(m)
    R = radial_poly(n, am, rho)
    norm = sqrt(n + 1) if m == 0 else sqrt(2 * (n + 1))
    if m > 0:
        return norm * R * np.cos(am * theta)
    if m < 0:
        return norm * R * np.sin(am * theta)
    return norm * R


ZERNIKE_NAMES = {
    (1, 1): "tilt x",
    (1, -1): "tilt y",
    (2, 0): "defocus",
    (2, -2): "astigmatism 45",
    (2, 2): "astigmatism 0",
    (3, -1): "coma y",
    (3, 1): "coma x",
    (3, -3): "trefoil y",
    (3, 3): "trefoil x",
    (4, 0): "spherical",
    (4, 2): "secondary astig 0",
    (4, -2): "secondary astig 45",
    (4, 4): "quadrafoil x",
    (4, -4): "quadrafoil y",
    (5, 1): "secondary coma x",
    (5, -1): "secondary coma y",
    (5, 3): "secondary trefoil x",
    (5, -3): "secondary trefoil y",
    (5, 5): "pentafoil x",
    (5, -5): "pentafoil y",
    (6, 0): "secondary spherical",
    (6, 2): "tertiary astig 0",
    (6, -2): "tertiary astig 45",
    (6, 4): "secondary quadrafoil x",
    (6, -4): "secondary quadrafoil y",
    (6, 6): "hexafoil x",
    (6, -6): "hexafoil y",
    (7, 1): "tertiary coma x",
    (7, -1): "tertiary coma y",
    (7, 3): "tertiary trefoil x",
    (7, -3): "tertiary trefoil y",
    (7, 5): "secondary pentafoil x",
    (7, -5): "secondary pentafoil y",
    (7, 7): "heptafoil x",
    (7, -7): "heptafoil y",
}


class Simulator:
    def __init__(self, N=512, Dpix=128, nmodes=33, jstart=4):
        """Zernike modes j = jstart ... jstart + nmodes - 1 (Noll) on an N x N grid with a pupil of Dpix pixels.

        The default set, Noll j = 4 to 36 (radial orders 2 to 7), is the set corrected in the paper; tip and
        tilt are handled by the steering mirror."""
        self.N, self.Dpix, self.nmodes, self.jstart = N, Dpix, nmodes, jstart
        self.ppl = N / Dpix  # focal-plane pixels per lambda/D
        x = np.arange(N) - (N - 1) / 2.0  # symmetric grid: flip is exact point reflection
        X, Y = np.meshgrid(x, x)
        r = np.hypot(X, Y) / (Dpix / 2.0)
        self.pupil = r <= 1.0
        self.A = self.pupil.astype(float)
        theta = np.arctan2(Y, X)
        rho = np.clip(r, 0.0, 1.0)
        self.modes_info = noll_indices(nmodes, start=jstart)
        self.Z = np.zeros((nmodes, N, N))
        for i, (j, n, m) in enumerate(self.modes_info):
            self.Z[i] = zernike(n, m, rho, theta) * self.pupil
        self.even_mask = np.array([abs(m) % 2 == 0 for (_, _, m) in self.modes_info])
        self.odd_mask = ~self.even_mask
        self.Zp = self.Z[:, self.pupil]  # (nmodes, npupil)
        self.npup = int(self.pupil.sum())
        # focal-plane coordinates (lambda/D)
        u = (np.arange(N) - N / 2) / self.ppl
        self.U, self.V = np.meshgrid(u, u)
        self.Rf = np.hypot(self.U, self.V)

    def phase(self, coeffs):
        return np.tensordot(coeffs, self.Z, axes=(0, 0))

    def rms(self, phase):
        p = phase[self.pupil]
        return float(np.sqrt(np.mean(p**2)))

    def scale_to_rms(self, coeffs, target):
        cur = self.rms(self.phase(coeffs))
        return coeffs * (target / cur) if cur > 0 else coeffs

    def even(self, coeffs):
        return coeffs * self.even_mask

    def odd(self, coeffs):
        return coeffs * self.odd_mask

    @staticmethod
    def twin_map(phase):
        """T(phi)(x) = -phi(-x) on the symmetric grid."""
        return -np.flip(phase, axis=(0, 1))

    def twin_coeffs(self, coeffs):
        c = coeffs.copy()
        c[self.even_mask] *= -1.0
        return c

    def project(self, phase_map):
        return np.linalg.lstsq(self.Zp.T, phase_map[self.pupil], rcond=None)[0]

    def intensity(self, phase, normalize=True):
        Uf = self.A * np.exp(1j * phase)
        I = np.abs(fftshift(fft2(Uf))) ** 2
        return I / I.sum() if normalize else I

    def crop(self, img, half_ld):
        """Central crop of +/- half_ld lambda/D."""
        h = int(round(half_ld * self.ppl))
        c = self.N // 2
        return img[c - h : c + h, c - h : c + h]

    def encircled(self, I, radius_ld):
        return float(I[self.Rf <= radius_ld].sum())

    def J(self, psi, eps, t=1e-3):
        """Linear response of the image to a phase change eps about psi (central difference)."""
        return (self.intensity(psi + t * eps) - self.intensity(psi - t * eps)) / (2 * t)

    @staticmethod
    def l1(dI):
        return float(np.abs(dI).sum())

    @staticmethod
    def F_shot(Ia, Ib):
        """Shot-noise discriminability per photon, sum (Ia - Ib)^2 / I, with I the mean of the two images."""
        m = 0.5 * (Ia + Ib)
        ok = m > 0
        return float(((Ia - Ib)[ok] ** 2 / m[ok]).sum())

    @staticmethod
    def S2(Ia, Ib):
        """Read-noise discriminability, sum (Ia - Ib)^2."""
        return float(((Ia - Ib) ** 2).sum())

    def vk_screen(self, Dr0, L0_D=10.0, l0_D=0.01, rng=None):
        """Von Karman phase screen (FFT method with three subharmonic levels), units D = 1."""
        rng = np.random.default_rng() if rng is None else rng
        N = self.N
        delta = 1.0 / self.Dpix
        r0, L0, l0 = 1.0 / Dr0, L0_D, l0_D
        fm, f0 = 5.92 / l0 / (2 * pi), 1.0 / L0
        del_f = 1.0 / (N * delta)
        fx = (np.arange(N) - N / 2) * del_f
        FX, FY = np.meshgrid(fx, fx)
        f = np.hypot(FX, FY)
        PSD = 0.023 * r0 ** (-5 / 3) * np.exp(-((f / fm) ** 2)) / (f**2 + f0**2) ** (11 / 6)
        PSD[N // 2, N // 2] = 0.0
        cn = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) * np.sqrt(PSD) * del_f
        phz_hi = np.real(ifftshift(ifft2(ifftshift(cn))) * N**2)  # ift2 with delta_f = 1 (Schmidt 2010)
        xs = (np.arange(N) - N / 2) * delta
        X, Y = np.meshgrid(xs, xs)
        phz_lo = np.zeros((N, N))
        for p in (1, 2, 3):
            dfp = 1.0 / (3**p * N * delta)
            fxp = np.array([-1.0, 0.0, 1.0]) * dfp
            FXp, FYp = np.meshgrid(fxp, fxp)
            fp = np.hypot(FXp, FYp)
            PSDp = 0.023 * r0 ** (-5 / 3) * np.exp(-((fp / fm) ** 2)) / (fp**2 + f0**2) ** (11 / 6)
            PSDp[1, 1] = 0.0
            cnp = (rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))) * np.sqrt(PSDp) * dfp
            SH = np.zeros((N, N), complex)
            for ii in range(3):
                for jj in range(3):
                    SH += cnp[ii, jj] * np.exp(2j * pi * (FXp[ii, jj] * X + FYp[ii, jj] * Y))
            phz_lo += np.real(SH)
        phz_lo -= phz_lo.mean()
        return phz_hi + phz_lo

    def turbulence_coeffs(self, Dr0, seed):
        rng = np.random.default_rng(seed)
        scr = self.vk_screen(Dr0, rng=rng)
        return self.project(scr)


def ideal_observer_accuracy(Ia, Ib, N_photons, trials, rng):
    """Fraction of correct decisions of the Poisson likelihood-ratio test between means N Ia and N Ib.

    Half of the trials are drawn from each hypothesis."""
    muA, muB = N_photons * Ia, N_photons * Ib
    floor = 1e-15
    w = np.log((muA + floor) / (muB + floor))
    off = (muA - muB).sum()
    correct = 0
    nA = trials // 2
    for k in range(trials):
        from_A = k < nA
        n = rng.poisson(muA if from_A else muB)
        llr = (n * w).sum() - off
        correct += int((llr > 0) == from_A)
    return correct / trials
