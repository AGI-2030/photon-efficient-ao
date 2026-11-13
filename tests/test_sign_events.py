import numpy as np
import pytest

pytest.importorskip("torch")

from aonet.analysis import even_mode_mask  # noqa: E402
from aonet.sign_events import even_alignment, find_events, frame_states  # noqa: E402


def test_alignment_of_the_twin_and_of_the_truth():
    even = even_mode_mask()
    gram = np.eye(33)
    y = np.random.default_rng(0).normal(scale=0.3, size=(1, 2, 33))
    twin = np.where(even, -y, y)
    q, decided, _ = even_alignment(np.concatenate([y, twin], 1), np.concatenate([y, y], 1), gram, even)
    assert decided.all()
    assert np.allclose(q, [[1, 1, -1, -1]])
    assert frame_states(q, decided).tolist() == [[1, 1, -1, -1]]


def test_small_modes_and_small_estimates_are_undecided():
    even = even_mode_mask()
    y = np.where(even, 0.01, 0.5)[None, None]
    q, decided, _ = even_alignment(y, y, np.eye(33), even)
    assert not decided.any()


def test_events():
    states = np.array([1, -1, 0, 1, -1, 1, 1, -1, -1])
    assert find_events(states) == [(1, 6, True), (7, 8, False)]
    assert find_events(np.array([-1, 1, 1])) == []
