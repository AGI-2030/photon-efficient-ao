from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aonet.analysis import even_mode_mask  # noqa: E402
from aonet.config import load_config, load_run_settings  # noqa: E402
from aonet.environment import ClosedLoopEnvironment  # noqa: E402
from aonet.models import INPUTS, build_model, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs").glob("*.toml"))
PARAMETERS = {"film_resnet": 3_277_761, "plain_cnn": 3_991_633, "fno": 4_801_169, "vit": 3_841_697}
CPU = torch.device("cpu")


def test_configurations():
    assert len(CONFIGS) == 14
    for path in CONFIGS:
        cfg = load_config(path)
        assert cfg.name == path.stem
        assert load_run_settings(path)["inputs"] in INPUTS


@pytest.mark.parametrize("name", sorted(PARAMETERS))
def test_models(name):
    cfg = next(load_config(p) for p in CONFIGS if load_config(p).model.name == name)
    torch.manual_seed(0)
    model = build_model(cfg.model).eval()
    assert sum(p.numel() for p in model.parameters()) == PARAMETERS[name]
    x = torch.rand(2, 128, 128)
    with torch.no_grad():
        out = model(x, x, torch.zeros(2, 33), torch.zeros(2))
    assert out.shape == (2, 33)


def test_input_configurations():
    cfg = load_config(ROOT / "configs" / "film_resnet_full.toml")
    previous, current = torch.rand(2, 128, 128), torch.rand(2, 128, 128)
    delta, exposure = torch.randn(2, 33), torch.zeros(2)
    outputs = {}
    for inputs in INPUTS:
        torch.manual_seed(0)
        model = build_model(cfg.model, inputs=inputs).eval()
        with torch.no_grad():
            outputs[inputs] = (
                model(previous, current, delta, exposure),
                model(previous, current, torch.zeros_like(delta), exposure),
                model(current, current, torch.zeros_like(delta), exposure),
            )
    full, no_delta, single = outputs["full"]
    assert torch.equal(outputs["two_frames"][0], no_delta)
    assert torch.equal(outputs["single_frame"][0], single)


@pytest.mark.parametrize("name", ["film_resnet_full", "film_resnet_static"])
def test_camera(name):
    env = ClosedLoopEnvironment(load_config(ROOT / "configs" / f"{name}.toml"), CPU)
    flat = torch.zeros(1, env.basis.pixels, env.basis.pixels)
    assert abs(float(env.camera.encircled_energy(flat)) - 0.838) < 0.01
    c = 0.2 * torch.randn(1, 33, generator=torch.Generator().manual_seed(0))
    twin = torch.where(torch.from_numpy(even_mode_mask()), -c, c)
    a = env.camera.intensity(env.basis.phase(c), diversity=False)
    b = env.camera.intensity(env.basis.phase(twin), diversity=False)
    assert float((a - b).abs().max()) < 1e-5 * float(a.max())


def test_basis_matches_aosim():
    from aosim import Simulator

    env = ClosedLoopEnvironment(load_config(ROOT / "configs" / "film_resnet_static.toml"), CPU)
    sim = Simulator(N=384, Dpix=44, nmodes=33)
    assert np.abs(sim.Z[:, 170:214, 170:214] - env.basis.dm.numpy()).max() < 1e-5


@pytest.mark.skipif(not (ROOT / "weights").is_dir(), reason="weights not downloaded")
def test_weights_match_configurations():
    for path in CONFIGS:
        model, cfg = load_model(ROOT / "weights" / f"{path.stem}.pt")
        assert cfg.to_dict() == load_config(path).to_dict()
        assert model.inputs == load_run_settings(path)["inputs"]


def test_training_smoke(tmp_path):
    from aonet.training import smoke_config, train

    cfg = smoke_config(load_config(ROOT / "configs" / "film_resnet_static.toml"))
    train(cfg, "full", tmp_path / "run", CPU, micro_batch=4, log=lambda s: None)
    model, loaded = load_model(tmp_path / "run" / "best.pt")
    assert loaded.to_dict() == cfg.to_dict()
    assert (tmp_path / "run" / "latest.pt").exists()
