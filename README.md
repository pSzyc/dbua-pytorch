# Differentiable Beamforming for Ultrasound Autofocusing
### [Project Page](https://www.waltersimson.com/dbua) | [Paper](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_41) | [Pre-Print](https://waltersimson.com/dbua/static/pdfs/SimsonMICCAI2023.pdf) | [Data](https://github.com/waltsims/dbua/releases/tag/miccai2023)


[Walter Simson](https://waltersimson.com/),
[Louise Zhuang](https://profiles.stanford.edu/louise-zhuang),
[Sergio Sanabria](https://scholar.google.es/citations?hl=es&user=E7h77bAAAAAJ),
[Neha Antil](https://med.stanford.edu/profiles/neha-antil),
[Jeremy Dahl](https://med.stanford.edu/profiles/jeremy-dahl),
[Dongwoon Hyun](https://profiles.stanford.edu/dongwoon-hyun)<br>
Stanford University


This is the official implementation of the paper "Differentiable Beamforming for Ultrasound Autofocusing."

[![dbua_video](https://img.youtube.com/vi/cUoAsEA5snE/0.jpg)](https://www.youtube.com/watch?v=cUoAsEA5snE)

> **This fork adds a PyTorch port and migrates dependency management to [uv](https://docs.astral.sh/uv/).**
> The original **JAX** reference implementation lives in `JaxDbua/`, and a numerically
> faithful **PyTorch** port lives in `TorchDbua/`. The two mirror each other file-for-file
> (`dbua.py`, `das.py`, `paths.py`, `losses.py`), so you can run either backend.

## High-Level Overview

Both backends share the same file layout (`JaxDbua/` = JAX reference, `TorchDbua/` = PyTorch port):

 * `dbua.py` - main experiment file. Adjust the configuration to run experiments (JAX: module-level globals; PyTorch: the `DBUAConfig` dataclass in `TorchDbua/conf.py`).
 * `das.py` -  Delay-and-sum IQ data according to a given time delay profile.
 * `paths.py` - Calculates the time-of-flight between two points  according to a speed-of-sound map.
 * `losses.py` - Contains the proposed phase-error and auxiliary loss functions.

## Getting Started

This project requires ffmpeg to be installed to save mp4 files. If you have not already, install ffmpeg on Ubuntu by running:

```bash
sudo apt-get install ffmpeg
```

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Install uv (if you
haven't already) and let it create the virtual environment (`.venv/`) from
`pyproject.toml` / `uv.lock`:

```bash
# Install uv (see https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the environment and install all dependencies (PyTorch + JAX)
uv sync
```

On Linux this pulls `torch` from the CUDA 12.8 wheel index and installs `jax[cuda12]`
for the reference/benchmark; both backends are set up by a single `uv sync`. Prefix
any command with `uv run` to execute it inside the managed environment — there is no
separate "activate" step required.

## Reproducing results

The hyperparameters set in dbua can be used to reproduce the results from the paper. The data from the paper can be found in the release on GitHub, and the mat files should be placed in the `data/` directory of this repository:

```
data
├── 1420.mat
├── 1465.mat
├── 1480.mat
├── 1510.mat
├── 1540.mat
├── 1555.mat
├── 1570.mat
├── checker2.mat
├── checker8.mat
├── four_layer.mat
├── inclusion_layer.mat
├── inclusion.mat
├── README.md
└── two_layer.mat
```

Run either backend as a module (module form is required for the package-relative
imports):

```bash
uv run python -m TorchDbua.dbua      # PyTorch port
uv run python -m JaxDbua.dbua        # JAX reference
```

Each run sweeps a global sound speed to initialize the map, optimizes for `N_ITERS`,
and writes diagnostic figures to `images/` / `scratch/` plus an optimization video to
`videos/` (pass `plot=False` — `DBUAConfig(plot=False)` for PyTorch, `main(..., plot=False)`
for JAX — to skip all figure/video output). Converged sound-speed maps and final metrics
are written to `results/`.

To reproduce the two side-by-side comparison experiments across both backends, run the
harness (it disables per-iteration plotting for speed):

```bash
uv run python run.py
```

Experiment knobs live at the top of each `dbua.py` (JAX globals) or in
`TorchDbua/conf.py` (the `DBUAConfig` dataclass): `LOSS`/`loss_name`
(`"pe"`/`"sb"`/`"cf"`/`"lc"`), `SAMPLE`/`sample`, `N_ITERS`/`n_iters`,
`LEARNING_RATE`/`learning_rate`.

### Citation

```Bibtex
@inproceedings{simson2023dbua,
            title={Differentiable Beamforming for Ultrasound Autofocusing},
            author={Simson, Walter and Zhuang, Louise and Sanabria, Sergio J and Antil, Neha and Dahl, Jeremy J and Hyun, Dongwoon},
            booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
            pages={428--437},
            year={2023},
            organization={Springer}
          }
```

## FAQ:

- **Q:** What computer configuration was used to develop this program?
- **A:** This code was developed using the following configuration:
  
| Attribute   | Detail                                |
|-------------|---------------------------------------|
| OS          |            Ubuntu Linux               |
| RAM         | 32GB                                  |
| GPU         | NVIDIA RTX A6000  (48 GB VRAM)        |
| CUDA Version| 12.1                                  |

