import datetime as dt
import math
import os
import re

import numpy as np

FRAC = 1 - math.exp(-2)
UTC = dt.timezone.utc


def add_data_argument(parser):
    parser.add_argument(
        "--data",
        required=True,
    )


def epoch(t):
    return t.replace(tzinfo=UTC).timestamp()


def clock(t, fmt="%H:%M:%S"):
    return dt.datetime.fromtimestamp(t, UTC).strftime(fmt)


_grid = {}


def r865(img, nbin):
    s = np.maximum(img.astype(np.float64), 0)
    if s.shape not in _grid:
        _grid[s.shape] = np.indices(s.shape).astype(np.float64)
    yy, xx = _grid[s.shape]
    tot = s.sum()
    if tot <= 0:
        return math.nan
    cy = (s * yy).sum() / tot
    cx = (s * xx).sum() / tot
    r = np.hypot(yy - cy, xx - cx)
    idx = (r * nbin).astype(np.int64)
    h = np.bincount(idx.ravel(), weights=s.ravel())
    c = np.cumsum(h)
    tgt = FRAC * tot
    k = int(np.searchsorted(c, tgt))
    prev = c[k - 1] if k > 0 else 0.0
    return (k + (tgt - prev) / h[k]) / nbin


ATM_FILES = {
    "r0": "dimm_r0.txt",
    "heater": "heater_power.txt",
    "fan": "fan_power.txt",
    "T1": "temp_humidity_1.txt",
    "T2": "temp_humidity_2.txt",
    "T3": "temp_humidity_3.txt",
    "W1": "wind_speed_1.txt",
    "W2": "wind_speed_2.txt",
    "W3": "wind_speed_3.txt",
}


def _t(day, s):
    d = dt.datetime.strptime(day, "%Y%m%d")
    h, m, sec = map(int, s.split(":"))
    return d + dt.timedelta(hours=h, minutes=m, seconds=sec)


def read_series(path, day):
    ts = []
    vals = []
    nbad = 0
    for l in open(path, encoding="utf-8-sig", errors="replace").read().splitlines()[1:]:
        l = re.sub(r"(\d+\.\d{3})(\d\.\d{3}e[+-]\d+)", r"\1 \2", l)
        t = l.split()
        if len(t) < 2:
            continue
        try:
            v = [float(x) for x in t[1:]]
            tt = _t(day, t[0])
        except ValueError:
            nbad += 1
            continue
        ts.append(tt)
        vals.append(v)
    if nbad:
        print(f"  [{os.path.basename(path)}: skipped {nbad} corrupted lines]")
    return np.array([epoch(x) for x in ts]), np.array(vals)


def load_atm(data, day, keys=tuple(ATM_FILES)):
    return {k: read_series(os.path.join(data, "atmosphere", day, ATM_FILES[k]), day) for k in keys}


def window(series, t0, t1, col=0):
    t, v = series
    m = (t >= t0) & (t <= t1)
    return v[m, col]


def interp(series, t, col=0):
    tt, v = series
    return float(np.interp(t, tt, v[:, col]))
