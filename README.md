# Photon-efficient adaptive optics without a wavefront sensor

Simulation code for the paper *Photon-Efficient Adaptive Optics in Turbulence without a Wavefront Sensor*
(submitted).

The adaptive optics loop corrects 33 Zernike modes (Noll j = 4 to 36) from far-field images alone. The network
receives the current frame, a reference frame, the known change of the correction between the two frames, and the
photon count, and returns the residual coefficients. The repository has two parts.

- `aosim/`, `simulations/`, `figures/`: a numpy model of image formation with the ideal observer and the ideal
  estimators, used for the theory checks, the loop design and the comparison of sensing methods (Notes S1 to S5,
  Table 3, Figs. S1 to S5).
- `aonet/`, `configs/`, `scripts/`: a torch simulator of the closed loop, the four network backbones, the DAgger
  training and the evaluations behind the network results (Figs. 2, 3 and 5(a), Table S1).

Results that take long to compute are stored in `data/`, and the figure scripts read them from there.

## Installation

```bash
pip install -e .              # aosim, simulations, figures (numpy, scipy, matplotlib)
pip install -e ".[network]"   # adds torch for aonet
```

Python 3.9 or later and numpy 2 are required (the cached results in `data/` are numpy 2 pickles). The networks were
trained and evaluated with Python 3.11, torch 2.5.1 and CUDA 12.1 on NVIDIA A100 GPUs.

The trained weights of the 14 networks are published on Zenodo ([10.5281/zenodo.24183126](https://doi.org/10.5281/zenodo.24183126)) as `weights.zip`
(200 MB). Unpack it in the repository folder to obtain `weights/<name>.pt`. `environment.yml` reproduces the
environment of the network runs.

## Layout

| Path | Content |
|---|---|
| `aosim/simulator.py` | Zernike basis, far-field images, twin transform, von Kármán screens, ideal observer |
| `simulations/theory_checks.py` | Monte Carlo sign accuracy, pupil asymmetry, alternating projection (Note S5) |
| `simulations/loop_design.py` | camera window, defocus probe, scaling laws, reversion rule, boiling, scintillation, scans of loop period, D/r0 and photons with ideal estimators |
| `simulations/reference_frames.py` | sign decision with the previous or the initial frame as reference (Fig. 1(e), Fig. S3(e)) |
| `simulations/photon_comparison.py` | image-based loop, phase diversity and Shack-Hartmann sensing with ideal estimators (Fig. 5(a), Table 3) |
| `figures/` | Figs. S1 to S5 and Table 3, written to `figures/output/` |
| `aonet/` | closed-loop simulator, networks (`models.py`), training (`training.py`), evaluation (`evaluation.py`), Cramér–Rao reference (`cramer_rao.py`), frames with a reversed even part (`sign_events.py`) |
| `configs/` | the 14 networks of Table S1 |
| `scripts/` | `train.py`, `evaluate.py`, `network_tables.py`, `sign_events.py` |
| `data/` | cached results read by the figure scripts |

## Figures and tables

### Simulations with ideal estimators (numpy)

The results of these simulations are stored in `data/`, so the figure scripts run without them. Each script is run
as a module from the repository folder and lists its stages in its docstring. `loop_design` takes one stage per
call, and `theory_checks` takes several.

| Item | Data | Plot |
|---|---|---|
| Fig. 1(e) | `python -m simulations.reference_frames run <ratio>` for ratios 0.03, 0.1, 0.3 and 1 | |
| Fig. 5(a), Table 3 | `python -m simulations.photon_comparison run <method> <read\|shot> [--flow 0.3\|1.0]` for the four methods | `python figures/table3.py` |
| Fig. S1 | `python -m simulations.theory_checks sign_accuracy` and `python -m simulations.loop_design scaling` | `python figures/figure_s1.py` |
| Fig. S2 | `python -m simulations.loop_design probe` and `... window` | `python figures/figure_s2.py` |
| Fig. S3 | `python -m simulations.loop_design loop_period`, `... photons`, `... turbulence_strength`, then `... tables`, and `python -m simulations.reference_frames gain` | `python figures/figure_s3.py` |
| Fig. S4 | `python -m simulations.loop_design boiling`, `... scintillation` and `python -m simulations.theory_checks pupil_asymmetry` | `python figures/figure_s4.py` |
| Note S5 | `python -m simulations.theory_checks alternating_projection alternating_projection_noise summary` | |

Table S2 is stored in `data/turbulence_strength.csv`.

### Networks (torch)

`scripts/evaluate.py` runs a network under the conditions of a figure and writes the per-frame records of every
sequence (`cell_XXX_<method>.npz`) and a `summary.csv` of steady-state means. The tables of `data/network/` are
built from these records.

```bash
# Fig. 2, Table S1 and Fig. S5: the 12 networks of the input ablation on the grid of 36 cells
for net in film_resnet vit plain_cnn fno; do
  for inputs in full two_frames single_frame; do
    python scripts/evaluate.py weights/${net}_${inputs}.pt grid --out runs/grid/${net}_${inputs}
  done
done
# Fig. S3(f) and Fig. 5(a)
python scripts/evaluate.py weights/film_resnet_full.pt gain --out runs/gain/film_resnet_full
python scripts/evaluate.py weights/film_resnet_static.pt static --out runs/static
python scripts/network_tables.py --runs runs --weights weights --out data/network

# Fig. 3(a) to 3(c)
python scripts/evaluate.py weights/film_resnet_full.pt strength --out runs/strength
python scripts/evaluate.py weights/film_resnet_full.pt photons --methods network cramer_rao --out runs/photons
python scripts/evaluate.py weights/film_resnet_full.pt loop_period --methods network oracle cramer_rao \
    --out runs/loop_period/film_resnet_full
python scripts/evaluate.py weights/film_resnet_initial_frame.pt loop_period --out runs/loop_period/film_resnet_initial_frame

# Fig. 3(d): frames in which the network reverses the even part of the residual
python scripts/sign_events.py runs/loop_period/film_resnet_full --out data/network
```

`network_tables.py` uses cell 26 of the grid (or the `main` experiment, which runs only that cell) for Fig. 2(b, c),
Fig. S5 and Table S1. The steady state is frames 15 to 29 for Fig. 2 and Table S1, frames 20 to 29 for Fig. S3(f)
and frames 8 to 10 for Fig. 5(a). `aonet/sign_events.py` defines the frames and events of Fig. 3(d).
`data/network/training_log.csv` is taken from the training logs. The plotting scripts of the main-text figures are
not part of this repository.

## Training

```bash
python scripts/train.py configs/film_resnet_full.toml --out runs/train/film_resnet_full
```

One network takes 1 to 23 hours on one A100 GPU (`data/network/training_log.csv`). The `[run]` table of each
configuration gives the input configuration and the batch settings used for the released weights. The rollout
batch changes the random stream, so it should be kept to repeat a run. GPU kernels are not deterministic, and a
repeated run therefore follows the original only approximately. `--smoke` runs two short iterations to check the
installation.

## Reproducibility

Every random draw in `aonet` comes from a generator seeded by the condition, the batch and the frame, so a
network, its input variants and the references see the same turbulence. With the released weights,
`scripts/evaluate.py` reproduces the traces of the paper exactly on an A100 GPU with torch 2.5.1, and
`scripts/network_tables.py` and `scripts/sign_events.py` reproduce `data/network/` up to the last digit. Other GPUs and versions give small
differences that grow over the frames of a closed loop.

## Tests

```bash
pip install -e ".[test]"
pytest tests
```

The torch tests are skipped when torch is not installed, and the test of the weights runs when `weights/` exists.

## Laboratory data

The laboratory data are published on Zenodo ([10.5281/zenodo.24183126](https://doi.org/10.5281/zenodo.24183126)): the far-field frames of the 190 network-loop
measurements (Fig. 4(a-c), Table S3, Video S1) with a table of the measurements (`runs.csv`), the far-field frames of
the comparison session with the installed Shack-Hartmann (SH) loop (Fig. 4(d)), and the atmospheric records of both
days. The deposit's `README.md` describes every file. The control software and the calibration of the laboratory
setup are not distributed; `lab/closed_loop.py` is a hardware-independent version of the control loop.

`lab/` needs numpy only (`closed_loop.py` also needs torch). Unpack `lab_network_loop.zip`, `lab_sh_comparison.zip`
and `atmosphere.zip` into one folder and run, from the repository folder,

```bash
python -m lab.network_loop --data <folder> --out data/lab/r865_all_runs.csv   # Fig. 4(a-c)
python -m lab.sh_summary   --data <folder> --out data/lab/sh_comparison.csv   # Fig. 4(d)
python -m lab.table_s3     --runs data/lab/r865_all_runs.csv --sh data/lab/sh_comparison.csv --out data/lab/table_s3.csv
```

| Path | Content |
|---|---|
| `lab/common.py` | R86.5 metric, reading of the atmospheric records, time windows |
| `lab/network_loop.py` | R86.5 of the ref, late (second half of the loop), hold and after phases of every measurement, with r0, temperature, wind, heater and fan power averaged over the time window of the measurement |
| `lab/sh_frames.py`, `lab/sh_summary.py` | per-frame far-field R86.5 and per-run summary of the SH comparison |
| `lab/table_s3.py` | Table S3 from the two tables above |
| `lab/closed_loop.py` | control loop of Section 3.1 with abstract camera and mirror interfaces (`Camera.grab`, `Mirror.apply`) and the network estimator with the preprocessing of `aonet`; `run_loop(camera, mirror, NetworkEstimator("weights/film_resnet_full.pt"), gain=0.25)` |
| `data/lab/r865_all_runs.csv` | one row per network-loop measurement |
| `data/lab/sh_comparison.csv` | one row per SH comparison run |
| `data/lab/table_s3.csv` | Table S3 before rounding |

R86.5 is the radius around the energy centroid that contains 1 − e⁻² of the energy of a 214 × 214 frame, after
subtraction of the background and with negative pixels set to zero. The R86.5 of a phase in the paper is the mean of
the per-frame values, `Rinst_<phase>` in `r865_all_runs.csv`; `R_<phase>` is the R86.5 of the mean frame (long
exposure) and is given for comparison. The network-loop frames were background-subtracted at acquisition. The
far-field frames of the SH session were saved as full 8-bit frames; the background is the median of each full frame
(`lab/sh_frames.py`). The 57 measurements at a gain of 0.25 reduce `Rinst_late` relative to `Rinst_ref` by 1.4% to
24.5%.

## Citation

If you use this code or the accompanying data, please cite:

*Photon-Efficient Adaptive Optics in Turbulence without a Wavefront Sensor* (submitted to *Engineering*).
Zenodo: [10.5281/zenodo.24183126](https://doi.org/10.5281/zenodo.24183126).
