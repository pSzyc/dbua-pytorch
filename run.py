"""Run the DBUA optimization for two experiments and save the converged maps.

Experiments
-----------
1. accuracy  : every focusing loss driving reconstruction on every constant-speed
               (homogeneous-SOS) phantom, both backends, ``N_ITERS`` iterations.
               We record only what identifies the run and how accurate it was:
               the driving cost function, the phantom, and the MAE (± standard
               error) of the recovered sound speed against the phantom's known
               constant. The per-cost-function scores on the converged map are
               intentionally dropped -- this experiment is about reconstruction
               accuracy, not focusing quality.
2. recovery  : the proposed loss (``EXP2_LOSS``) on every heterogeneous phantom,
               both backends. These have no single ground-truth speed, so there
               is no MAE -- we just save the recovered sound-speed map for
               qualitative comparison.

Both experiments run with ``plot=True`` so each backend saves its own SoS /
B-mode figure (``scratch/{sample}_{loss}_{backend}.png``) and loss-vs-speed
survey -- we don't re-render anything here.

Results layout (under ``results/``)
    {sample}-{loss}-{backend}.npy   converged sound-speed map
    mae-{backend}.json              accuracy experiment -> {sample: {loss: {mae, mae_se}}}

``analyse.py`` turns ``mae-{backend}.json`` into the MAE comparison table.
"""

import gc
import json
from pathlib import Path

import numpy as np

# Cost functions driving the optimization (each run is driven by exactly one).
LOSSES = ["sb", "lc", "cf", "pe"]

# Gradient-descent iterations per run.
N_ITERS = 300

# Driving loss for the heterogeneous-phantom recovery experiment (the proposed
# method, phase error).
EXP2_LOSS = "pe"

BACKENDS = ["torch", "jax"]

RESULTS_DIR = Path("results")


def _run_one_torch(sample: str, loss: str, n_iters: int) -> dict:
    from TorchDbua.conf import DBUAConfig
    from TorchDbua.dbua import main

    config = DBUAConfig(sample=sample, loss_name=loss, n_iters=n_iters, plot=True)
    c, metrics = main(config)
    np.save(RESULTS_DIR / f"{sample}-{loss}-torch.npy", c.cpu().numpy())
    return metrics


def _run_one_jax(sample: str, loss: str, n_iters: int) -> dict:
    import JaxDbua.dbua as jax_dbua

    # main() reads N_ITERS as a module global at call time; override it here.
    jax_dbua.N_ITERS = n_iters
    c, metrics = jax_dbua.main(sample, loss, plot=True)
    np.save(RESULTS_DIR / f"{sample}-{loss}-jax.npy", np.array(c))
    return metrics


_RUNNERS = {"torch": _run_one_torch, "jax": _run_one_jax}


def _safe(runner, sample: str, loss: str, n_iters: int) -> dict:
    """Run one optimization, capturing failures so the sweep keeps going."""
    try:
        return runner(sample, loss, n_iters)
    except Exception as e:
        return {"error": str(e)}


def _mae_only(metrics: dict) -> dict:
    """Keep just the reconstruction accuracy, dropping cost-function scores."""
    if "error" in metrics:
        return metrics
    return {"mae": metrics.get("mae"), "mae_se": metrics.get("mae_se")}


def run_accuracy(backend: str, samples: list[str]) -> dict:
    """Experiment 1: full loss x homogeneous-phantom grid for one backend.

    Returns ``{sample: {loss: {mae, mae_se}}}``.
    """
    runner = _RUNNERS[backend]
    results = {}
    for sample in samples:
        results[sample] = {
            loss: _mae_only(_safe(runner, sample, loss, N_ITERS))
            for loss in LOSSES
        }
    return results


def run_recovery(backend: str, samples: list[str]) -> None:
    """Experiment 2: the proposed loss on each heterogeneous phantom.

    No ground-truth constant exists, so nothing is recorded -- ``_safe`` still
    writes the converged sound-speed map (``.npy``) as a side effect.
    """
    runner = _RUNNERS[backend]
    for sample in samples:
        _safe(runner, sample, EXP2_LOSS, N_ITERS)


def free_gpu() -> None:
    """Release Torch's cached GPU memory so the next backend can allocate."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)

    from TorchDbua.conf import DBUAConfig

    ctrue = DBUAConfig().ctrue
    # Homogeneous phantoms carry a known constant ground-truth speed (> 0);
    # heterogeneous phantoms record 0.
    homogeneous = [s for s, c_true in ctrue.items() if c_true > 0]
    heterogeneous = [s for s, c_true in ctrue.items() if c_true == 0]

    for backend in BACKENDS:
        accuracy = run_accuracy(backend, homogeneous)
        with open(RESULTS_DIR / f"mae-{backend}.json", "w") as f:
            json.dump(accuracy, f, indent=2)

        run_recovery(backend, heterogeneous)
        free_gpu()
