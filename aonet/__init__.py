"""Simulated closed loop, networks and DAgger training (torch).

Same physics as aosim: pupil diameter D = 1, phase in rad at 1080 nm, Noll j = 4 to 36 on the deformable mirror,
piston, tip and tilt removed by an ideal steering mirror, coherence time tau_0 = 0.314 r0 / v (frozen flow).
"""

from .config import ClosedLoopConfig, config_from_dict, load_config, load_run_settings
from .environment import ClosedLoopEnvironment
from .models import build_model, load_model

__all__ = [
    "ClosedLoopConfig",
    "ClosedLoopEnvironment",
    "build_model",
    "config_from_dict",
    "load_config",
    "load_model",
    "load_run_settings",
]
