"""Estimators of the residual wavefront: FiLM-ResNet, plain CNN, Fourier neural operator and vision transformer.

Every model takes the reference frame, the current frame (128 x 128 each), the command difference d (33 modes,
rad) and the log exposure, and returns the 33 residual coefficients Z4 to Z36 in rad. Group normalization only,
so the network behaves the same in the loop and in training.

The input configuration (full, two_frames or single_frame) restricts what the network sees: two_frames sets d to
zero, single_frame also replaces the reference frame by the current frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ClosedLoopConfig, ModelConfig, config_from_dict

INPUTS = ("full", "two_frames", "single_frame")


class FiLMBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, cond_dim: int, groups: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.norm1 = nn.GroupNorm(groups, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(groups, cout)
        self.film = nn.Linear(cond_dim, 2 * cout)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.GroupNorm(groups, cout))
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return self.act(h + (x if self.skip is None else self.skip(x)))


class FiLMResNet(nn.Module):
    """Residual network whose blocks are scaled and shifted by a condition vector computed from d and the exposure."""

    def __init__(self, cfg: ModelConfig, nmodes: int = 33):
        super().__init__()
        self.nmodes = nmodes
        self.cond = nn.Sequential(
            nn.Linear(nmodes + 1, cfg.condition_dim),
            nn.SiLU(),
            nn.Linear(cfg.condition_dim, cfg.condition_dim),
            nn.SiLU(),
        )
        w0 = cfg.widths[0]
        self.stem = nn.Sequential(
            nn.Conv2d(3, w0, 3, 1, 1, bias=False), nn.GroupNorm(cfg.groups, w0), nn.SiLU()
        )
        blocks, cin = [], w0
        for stage, (width, count) in enumerate(zip(cfg.widths, cfg.blocks)):
            for b in range(count):
                stride = 2 if (b == 0 and stage > 0) else 1
                blocks.append(FiLMBlock(cin, width, stride, cfg.condition_dim, cfg.groups))
                cin = width
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(nn.Linear(cin + cfg.condition_dim, 512), nn.SiLU(), nn.Linear(512, nmodes))

    def forward(self, previous, current, delta_command, log_exposure):
        x = torch.stack([previous, current, current - previous], dim=1)
        cond = self.cond(torch.cat([delta_command, log_exposure.view(-1, 1)], dim=-1))
        h = self.stem(x)
        for block in self.blocks:
            h = block(h, cond)
        h = h.mean(dim=(-2, -1))
        return self.head(torch.cat([h, cond], dim=-1))


class _ResidualBlock(nn.Module):
    def __init__(self, conv: nn.Module, skip: nn.Module):
        super().__init__()
        self.conv = conv
        self.skip = skip

    def forward(self, x):
        return F.silu(self.conv(x) + self.skip(x))


class PlainCNN(nn.Module):
    """Residual network without conditioning in the blocks; d and the exposure join the pooled features."""

    def __init__(self, cfg: ModelConfig, nmodes: int = 33):
        super().__init__()
        self.nmodes = nmodes
        w0 = cfg.widths[0]
        self.stem = nn.Sequential(
            nn.Conv2d(3, w0, 3, 1, 1, bias=False), nn.GroupNorm(cfg.groups, w0), nn.SiLU()
        )
        blocks, cin = [], w0
        for stage, (width, count) in enumerate(zip(cfg.widths, cfg.blocks)):
            for b in range(count):
                stride = 2 if (b == 0 and stage > 0) else 1
                blocks.append(self._residual_block(cin, width, stride, cfg.groups))
                cin = width
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(cin + nmodes + 1, 512), nn.SiLU(), nn.Linear(512, nmodes))

    @staticmethod
    def _residual_block(cin: int, cout: int, stride: int, groups: int) -> nn.Module:
        conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.GroupNorm(groups, cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
            nn.GroupNorm(groups, cout),
        )
        if stride == 1 and cin == cout:
            skip = nn.Identity()
        else:
            skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.GroupNorm(groups, cout))
        return _ResidualBlock(conv, skip)

    def forward(self, previous, current, delta_command, log_exposure):
        x = torch.stack([previous, current, current - previous], dim=1)
        h = self.blocks(self.stem(x)).mean(dim=(-2, -1))
        cond = torch.cat([delta_command, log_exposure.view(-1, 1)], dim=-1)
        return self.head(torch.cat([h, cond], dim=-1))


class SpectralConv2d(nn.Module):
    """Convolution in the Fourier domain, truncated to the lowest modes x modes frequencies."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels, self.out_channels, self.modes = in_channels, out_channels, modes
        scale = 1 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, modes))
        self.weights_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        with torch.autocast(x.device.type, enabled=False):  # the FFT runs in float32
            x = x.float()
            x_ft = torch.fft.rfft2(x, norm="ortho")
            out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
            m = min(self.modes, H, W // 2 + 1)
            weights = torch.complex(self.weights_real[:, :, :m, :m], self.weights_imag[:, :, :m, :m])
            out_ft[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m, :m], weights)
            return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNO(nn.Module):
    """Fourier neural operator with four spectral layers on the lowest 16 x 16 frequencies."""

    def __init__(self, cfg: ModelConfig, nmodes: int = 33):
        super().__init__()
        self.nmodes = nmodes
        hidden = cfg.widths[0] if cfg.widths else 64
        modes = 16
        self.lift = nn.Conv2d(3, hidden, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(hidden, hidden, modes) for _ in range(4)])
        self.local = nn.ModuleList([nn.Conv2d(hidden, hidden, 1) for _ in range(4)])
        self.project = nn.Conv2d(hidden, hidden, 1)
        self.cond_embed = nn.Sequential(nn.Linear(nmodes + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.head = nn.Sequential(nn.Linear(2 * hidden, 512), nn.SiLU(), nn.Linear(512, nmodes))

    def forward(self, previous, current, delta_command, log_exposure):
        x = torch.stack([previous, current, current - previous], dim=1)
        h = self.lift(x)
        for spec, loc in zip(self.spectral, self.local):
            h = F.silu(spec(h) + loc(h) + h)
        h = self.project(h).mean(dim=(-2, -1))
        cond = self.cond_embed(torch.cat([delta_command, log_exposure.view(-1, 1)], dim=-1))
        return self.head(torch.cat([h, cond], dim=-1))


class ViT(nn.Module):
    """Vision transformer on 16 x 16 patches; a token formed from d and the exposure is read out."""

    def __init__(self, cfg: ModelConfig, nmodes: int = 33, image_size: int = 128):
        super().__init__()
        self.nmodes = nmodes
        hidden = cfg.widths[0] if cfg.widths else 256
        patch_size = 16
        n_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, hidden, patch_size, patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, hidden) * 0.02)
        self.cond_embed = nn.Sequential(nn.Linear(nmodes + 1, hidden), nn.SiLU())
        depth = cfg.blocks[0] if cfg.blocks else 4
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    hidden,
                    nhead=8,
                    dim_feedforward=4 * hidden,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, 512), nn.SiLU(), nn.Linear(512, nmodes))

    def forward(self, previous, current, delta_command, log_exposure):
        x = torch.stack([previous, current, current - previous], dim=1)
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        cond = self.cond_embed(torch.cat([delta_command, log_exposure.view(-1, 1)], dim=-1)).unsqueeze(1)
        tokens = torch.cat([cond, tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.norm(tokens[:, 0]))


class _RestrictedInputs:
    """Mixin that applies the input configuration before the forward pass of the backbone."""

    def __init__(self, *args, inputs: str = "full", **kwargs):
        if inputs not in INPUTS:
            raise ValueError(f"inputs must be one of {INPUTS}")
        super().__init__(*args, **kwargs)
        self.inputs = inputs

    def forward(self, previous, current, delta_command, log_exposure):
        if self.inputs == "single_frame":
            previous = current
            delta_command = torch.zeros_like(delta_command)
        elif self.inputs == "two_frames":
            delta_command = torch.zeros_like(delta_command)
        return super().forward(previous, current, delta_command, log_exposure)


class FiLMResNetModel(_RestrictedInputs, FiLMResNet):
    pass


class PlainCNNModel(_RestrictedInputs, PlainCNN):
    pass


class FNOModel(_RestrictedInputs, FNO):
    pass


class ViTModel(_RestrictedInputs, ViT):
    pass


ARCHITECTURES = {
    "film_resnet": FiLMResNetModel,
    "plain_cnn": PlainCNNModel,
    "fno": FNOModel,
    "vit": ViTModel,
}


def build_model(cfg: ModelConfig, nmodes: int = 33, image_size: int = 128, inputs: str = "full") -> nn.Module:
    """Fresh network with the given input configuration (initialized from the global torch RNG)."""
    cls = ARCHITECTURES[cfg.name]
    if cls is ViTModel:
        return cls(cfg, nmodes, image_size, inputs=inputs)
    return cls(cfg, nmodes, inputs=inputs)


def load_model(source: str | Path | Mapping[str, Any]) -> tuple[nn.Module, ClosedLoopConfig]:
    """Network and configuration from a weight file (weights/<name>.pt or a best.pt of training).

    The model is returned on the CPU in evaluation mode.
    """
    state = (
        source if isinstance(source, Mapping) else torch.load(source, map_location="cpu", weights_only=True)
    )
    cfg = config_from_dict(state["config"])
    model = build_model(
        cfg.model, cfg.optics.nmodes, cfg.optics.network_image_size, state.get("inputs", "full")
    )
    model.load_state_dict(state["model"], strict=True)
    return model.eval(), cfg
