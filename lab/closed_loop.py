from __future__ import annotations

import time

import numpy as np

NMODES = 33
DEFOCUS = 0
PROBE_RAD = 0.2


class Camera:
    def grab(self) -> np.ndarray:
        raise NotImplementedError


class Mirror:
    def apply(self, c33: np.ndarray) -> None:
        raise NotImplementedError


def crop_centroid(img: np.ndarray, half: int = 107, threshold: float = 0.1) -> np.ndarray:
    img = np.asarray(img)
    if img.shape[0] < 2 * half or img.shape[1] < 2 * half:
        raise ValueError(f"frame {img.shape} is smaller than the window {2 * half} x {2 * half}")
    s = img.astype(np.float64) - float(img.min())
    w = np.where(s > threshold * s.max(), s, 0.0)
    tot = w.sum()
    if tot > 0:
        cy = float(w.sum(1) @ np.arange(img.shape[0])) / tot
        cx = float(w.sum(0) @ np.arange(img.shape[1])) / tot
    else:
        cy, cx = (img.shape[0] - 1) / 2, (img.shape[1] - 1) / 2
    y0 = int(np.clip(round(cy) - half, 0, img.shape[0] - 2 * half))
    x0 = int(np.clip(round(cx) - half, 0, img.shape[1] - 2 * half))
    return img[y0 : y0 + 2 * half, x0 : x0 + 2 * half]


class NetworkEstimator:
    def __init__(self, weights: str, device: str = "cpu", counts_to_photons: float = 1.0):
        import torch
        from aonet.models import load_model
        from aonet.optics import FarFieldCamera, PupilBasis

        self.torch = torch
        self.device = torch.device(device)
        model, cfg = load_model(weights)
        self.model, self.cfg = model.to(self.device).eval(), cfg
        self.camera = FarFieldCamera(cfg.optics, cfg.camera, PupilBasis(cfg.optics, self.device), self.device)
        self.size = 2 * self.camera.half
        self.counts_to_photons = float(counts_to_photons)
        self.reference = cfg.reference.mode

    def preprocess(self, frame: np.ndarray):
        torch = self.torch
        if frame.shape != (self.size, self.size):
            raise ValueError(f"expected a {self.size} x {self.size} frame, got {frame.shape}")
        counts = torch.as_tensor(np.asarray(frame, np.float32) * self.counts_to_photons, device=self.device)[
            None
        ]
        image = self.camera.network_input(counts)
        log_exposure = torch.log10(counts.sum(dim=(-2, -1)).clamp_min(1.0)) - 4.0
        return image, log_exposure

    def __call__(self, previous: np.ndarray, current: np.ndarray, d: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            prev, _ = self.preprocess(previous)
            cur, log_exposure = self.preprocess(current)
            dd = torch.as_tensor(np.asarray(d, np.float32), device=self.device)[None]
            return self.model(prev, cur, dd, log_exposure).float().cpu().numpy()[0]


def _pace(t0: float, period: float) -> None:
    if period > 0:
        rest = period - (time.perf_counter() - t0)
        if rest > 0:
            time.sleep(rest)


def _grab_n(camera, n, half, period):
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        out.append(crop_centroid(camera.grab(), half))
        _pace(t0, period)
    return out


def run_loop(
    camera: Camera,
    mirror: Mirror,
    estimator,
    *,
    gain: float = 0.25,
    leak: float = 1.0,
    loop_modes=None,
    c_limit: float = 3.0,
    rate_hz: float = 20.0,
    n_ref: int = 100,
    n_loop: int = 500,
    n_hold: int = 100,
    n_after: int = 100,
    half: int = 107,
) -> dict:
    if n_ref < 2:
        raise ValueError("n_ref must be at least 2")
    period = 1.0 / rate_hz if rate_hz > 0 else 0.0
    mask = np.zeros(NMODES, bool)
    mask[list(range(NMODES)) if loop_modes is None else [int(m) for m in loop_modes]] = True
    zero = np.zeros(NMODES, np.float32)

    mirror.apply(zero)
    ref = _grab_n(camera, n_ref, half, period)

    c = zero.copy()
    d = zero.copy()
    previous, current = ref[-2], ref[-1]
    initial = ref[-1]
    c_initial = zero.copy()
    frames, estimates, corrections = [], [], []
    for i in range(n_loop):
        t0 = time.perf_counter()
        if getattr(estimator, "reference", "previous") == "initial":
            a = estimator(initial, current, c - c_initial)
        else:
            a = estimator(previous, current, d)
        update = np.where(mask, gain * np.asarray(a, np.float32), 0.0).astype(np.float32)
        if i == 0 and gain > 0 and mask[DEFOCUS]:
            update[DEFOCUS] += PROBE_RAD
        c_prev = c
        c = np.clip(leak * c + update, -c_limit, c_limit).astype(np.float32)
        d = c - c_prev
        mirror.apply(c)
        previous, current = current, crop_centroid(camera.grab(), half)
        frames.append(current)
        estimates.append(np.asarray(a, np.float32))
        corrections.append(c)
        _pace(t0, period)

    c_hold = np.mean(corrections[n_loop // 2 :], axis=0).astype(np.float32) if n_loop else zero
    mirror.apply(c_hold)
    hold = _grab_n(camera, n_hold, half, period)
    mirror.apply(zero)
    after = _grab_n(camera, n_after, half, period)
    return dict(
        ref=np.stack(ref),
        loop=np.stack(frames) if frames else np.zeros((0, 2 * half, 2 * half)),
        hold=np.stack(hold) if hold else None,
        after=np.stack(after) if after else None,
        estimates=np.stack(estimates) if estimates else None,
        corrections=np.stack(corrections) if corrections else None,
        hold_correction=c_hold,
    )
