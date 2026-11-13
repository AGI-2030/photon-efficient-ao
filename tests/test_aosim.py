import numpy as np

from aosim import Simulator


def test_flat_wavefront_encircled_energy():
    sim = Simulator(N=384, Dpix=44, nmodes=33)
    image = sim.intensity(np.zeros((sim.N, sim.N)))
    assert abs(sim.encircled(image, 1.22) - 0.838) < 0.01


def test_twin_wavefronts_give_identical_images():
    sim = Simulator(N=256, Dpix=64, nmodes=33)
    c = np.random.default_rng(0).normal(scale=0.2, size=sim.nmodes)
    twin = sim.twin_coeffs(c)
    assert np.abs(sim.intensity(sim.phase(c)) - sim.intensity(sim.phase(twin))).max() < 1e-12
    assert np.allclose(sim.phase(twin), sim.twin_map(sim.phase(c)))


def test_known_correction_separates_the_twins():
    sim = Simulator(N=256, Dpix=64, nmodes=33)
    rng = np.random.default_rng(1)
    phi = sim.scale_to_rms(rng.normal(size=sim.nmodes), 0.8)
    twin = sim.twin_coeffs(phi)
    c = phi / 2
    separation = np.abs(sim.intensity(sim.phase(phi - c)) - sim.intensity(sim.phase(twin - c))).sum()
    assert separation > 1e-3


def test_scale_to_rms():
    sim = Simulator(N=256, Dpix=64, nmodes=33)
    c = sim.scale_to_rms(sim.turbulence_coeffs(4, seed=700), 0.8)
    assert abs(sim.rms(sim.phase(c)) - 0.8) < 1e-12
