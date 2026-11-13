"""Statistics over the sequences of an evaluation (numpy).

The statistical unit is the sequence. A steady-state value is first averaged over the frames of the steady window
within each sequence; means and sample standard deviations (ddof = 1) are then taken over the sequences.
"""

from __future__ import annotations

import warnings

import numpy as np

from .optics import noll_indices


def even_mode_mask(nmodes: int = 33, noll_start: int = 4) -> np.ndarray:
    """True for the modes with an even azimuthal order |m| among Noll j = noll_start ... (15 of Z4 to Z36)."""
    order = np.array([m for _, _, m in noll_indices(noll_start - 1 + nmodes)[noll_start - 1 :]])
    return np.abs(order) % 2 == 0


def mean_sd(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()), float(values.std(ddof=1))


def steady(values: np.ndarray, first: int, last: int | None = None) -> np.ndarray:
    """Mean of each sequence over frames first ... last (inclusive; default the last frame)."""
    values = np.asarray(values, dtype=np.float64)
    stop = values.shape[1] if last is None else last + 1
    return values[:, first:stop].mean(axis=1)


def sign_ratio(correct: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Per-frame fraction of correct signs, NaN where no mode exceeds the threshold."""
    correct = np.asarray(correct, dtype=np.float64)
    count = np.asarray(count, dtype=np.float64)
    return np.where(count > 0, correct / np.maximum(count, 1), np.nan)


def steady_sign(correct: np.ndarray, count: np.ndarray, first: int, last: int | None = None) -> np.ndarray:
    """Per-frame sign accuracy averaged over the steady window within each sequence (NaN without eligible modes)."""
    ratio = sign_ratio(correct, count)
    stop = ratio.shape[1] if last is None else last + 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(ratio[:, first:stop], axis=1)


def parity_rms(label: np.ndarray, even: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(B, K) root mean square of the residual coefficients over the even and over the odd modes."""
    label = np.asarray(label, dtype=np.float64)
    return np.sqrt((label[..., even] ** 2).mean(-1)), np.sqrt((label[..., ~even] ** 2).mean(-1))


def summary_row(trace: dict[str, np.ndarray], first: int) -> dict[str, float]:
    """Steady-state means and standard deviations over the sequences of one evaluation cell."""
    row = {"sequences": int(trace["ee"].shape[0]), "first_steady_frame": first}
    for key in ("ee", "strehl", "residual_rms", "ee_open", "ee_fitting"):
        row[f"{key}_mean"], row[f"{key}_sd"] = mean_sd(steady(trace[key], first))
    for parity in ("even", "odd"):
        values = steady_sign(trace[f"{parity}_correct"], trace[f"{parity}_count"], first)
        row[f"{parity}_sign_mean"], row[f"{parity}_sign_sd"] = mean_sd(values)
    return row
