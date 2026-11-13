"""Numerical model of image-based wavefront sensing with known corrections (numpy)."""

from pathlib import Path

from .simulator import ZERNIKE_NAMES, Simulator, ideal_observer_accuracy, noll_indices, zernike

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

__all__ = [
    "ZERNIKE_NAMES",
    "Simulator",
    "ideal_observer_accuracy",
    "noll_indices",
    "zernike",
    "DATA",
    "data_dir",
]


def data_dir(name):
    """Folder data/<name> of the repository, created if needed."""
    path = DATA / name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
