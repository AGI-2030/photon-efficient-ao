"""Frames in which the estimate reverses the even part of the residual (Fig. 3(d), Section 3.1, Note S3).

Only the even modes whose true coefficient exceeds 0.05 rad in magnitude enter. With G the Gram matrix of the 33
modes over the discrete pupil (piston removed), the alignment of the estimate a with the true residual y on these
modes is the cosine of the two even phase maps,

    q = a^T G y / sqrt(a^T G a  y^T G y).

A frame is reversed for q <= -0.2 and correct for q >= 0.2. Frames in between, frames without such modes and frames
whose even estimate is below 0.01 rad RMS are undecided.

An event starts at the first reversed frame, from frame 1 on, and ends at the second of two consecutive correct
frames. Reversed and undecided frames in between belong to the same event. Event-aligned averages use the events
whose frames -2 to +5 lie within frames 1 to 29. The events of each sequence are averaged first and the sequences
are then weighted equally. The standard error is taken over these sequences, and the 95% intervals come from 2000
bootstrap resamples of the sequences.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import OpticsConfig
from .optics import PupilBasis

MODE_THRESHOLD = 0.05  # rad, true coefficient of an even mode that takes part
ESTIMATE_FLOOR = 0.01  # rad, smallest RMS of the even estimate for a decided frame
Q_THRESHOLD = 0.2
LAGS = np.arange(-2, 6)


def pupil_gram(optics: OpticsConfig) -> np.ndarray:
    """(33, 33) Gram matrix of the corrected modes over the discrete pupil, piston removed."""
    basis = PupilBasis(optics, torch.device("cpu"))
    maps = basis.dm[:, basis.mask].double().numpy()
    maps = maps - maps.mean(axis=1, keepdims=True)
    return maps @ maps.T / maps.shape[1]


def even_alignment(estimate, label, gram, even):
    """q, a mask of the decided frames and the RMS of the true even part on the selected modes, each (B, K)."""
    estimate, label = np.asarray(estimate, dtype=float), np.asarray(label, dtype=float)
    selected = (np.abs(label) > MODE_THRESHOLD) & even
    a, y = estimate * selected, label * selected

    def quadratic(u, v):
        return np.einsum("...i,ij,...j->...", u, gram, v)

    a_rms, y_rms = np.sqrt(np.maximum(quadratic(a, a), 0)), np.sqrt(np.maximum(quadratic(y, y), 0))
    norm = a_rms * y_rms
    q = np.clip(np.divide(quadratic(a, y), norm, out=np.full(norm.shape, np.nan), where=norm > 1e-15), -1, 1)
    decided = (selected.sum(-1) > 0) & (a_rms >= ESTIMATE_FLOOR) & (y_rms > 1e-15)
    return q, decided, y_rms


def frame_states(q, decided):
    """-1 for a reversed frame, +1 for a correct frame and 0 for an undecided frame."""
    state = np.zeros(np.shape(q), dtype=np.int8)
    state[decided & (q <= -Q_THRESHOLD)] = -1
    state[decided & (q >= Q_THRESHOLD)] = 1
    return state


def find_events(states, first: int = 1, streak: int = 2) -> list[tuple[int, int, bool]]:
    """(first frame, last frame, recovered) of every event in one sequence of frame states."""
    events, start, run = [], None, 0
    for k in range(first, len(states)):
        if start is None:
            if states[k] != -1:
                continue
            start, run = k, 0
        run = run + 1 if states[k] == 1 else 0
        if run >= streak:
            events.append((start, k, True))
            start, run = None, 0
    if start is not None:
        events.append((start, len(states) - 1, False))
    return events


def complete_events(events_by_sequence, frames: int, first: int = 1) -> list[tuple[int, int]]:
    """(sequence, start) of the events whose frames -2 to +5 lie within frames first to frames - 1."""
    return [
        (s, start)
        for s, events in enumerate(events_by_sequence)
        for start, _, _ in events
        if start >= first - LAGS[0] and start + LAGS[-1] < frames
    ]


def event_average(
    values, events: list[tuple[int, int]], seed: int, samples: int = 2000
) -> dict[str, np.ndarray]:
    """Mean of values (B, K) at each lag around the events, with the standard error and 95% bootstrap intervals
    over the sequences."""
    n = values.shape[0]
    per_sequence: list[list[np.ndarray]] = [[] for _ in range(n)]
    for s, start in events:
        per_sequence[s].append(values[s, start + LAGS])
    means = np.full((n, len(LAGS)), np.nan)
    for s, rows in enumerate(per_sequence):
        if rows:
            means[s] = np.mean(rows, axis=0)
    weights = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=samples)
    mean, low, high = np.zeros(len(LAGS)), np.zeros(len(LAGS)), np.zeros(len(LAGS))
    for j in range(len(LAGS)):
        has = np.isfinite(means[:, j]).astype(float)
        column = np.nan_to_num(means[:, j])
        mean[j] = column.sum() / has.sum()
        counts = weights @ has
        draws = np.divide(weights @ column, counts, out=np.full(samples, np.nan), where=counts > 0)
        low[j], high[j] = np.quantile(draws[np.isfinite(draws)], [0.025, 0.975])
    rows = means[np.isfinite(means[:, 0])]
    se = rows.std(axis=0, ddof=1) / np.sqrt(len(rows))
    return {"lag": LAGS, "mean": mean, "se": se, "low": low, "high": high, "sequences": len(rows)}


def analyze(trace: dict[str, np.ndarray], gram: np.ndarray, even: np.ndarray, seed: int) -> dict:
    """Events of one evaluation cell and the event-aligned encircled energy."""
    q, decided, y_rms = even_alignment(trace["estimate"], trace["label"], gram, even)
    states = frame_states(q, decided)
    events_by_sequence = [find_events(row) for row in states]
    complete = complete_events(events_by_sequence, states.shape[1])
    ee = trace["ee"]
    change_from_previous = np.array([ee[s, start + 1] - ee[s, start - 1] for s, start in complete])
    return {
        "events": sum(len(e) for e in events_by_sequence),
        "complete_events": len(complete),
        "reversed_fraction": float((states[:, 1:] == -1).mean()),
        "ee": event_average(ee, complete, seed),
        "drop_le_0.05": float(np.mean(change_from_previous > -0.05)),
        "change_le_0.05": float(np.mean(np.abs(change_from_previous) <= 0.05)),
        "q": q,
        "states": states,
        "even_rms": y_rms,
        "starts": [(s, start) for s, events in enumerate(events_by_sequence) for start, _, _ in events],
    }
