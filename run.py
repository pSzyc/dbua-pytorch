"""Run the PyTorch and JAX DBUA implementations for two comparison experiments
and save the converged sound-speed maps plus final metrics for analyse.py.

Experiments
-----------
1. all-loss     : one homogeneous (constant-speed) phantom, every focusing loss,
                  both backends, ``ALL_LOSS_N_ITERS`` iterations. Feeds the
                  per-cost-function comparison tables.
2. all-phantoms : the phase-error loss on every phantom, both backends,
                  ``ALL_PHANTOMS_N_ITERS`` iterations. Feeds the PE difference /
                  accuracy plots.

Results layout (under ``results/``)
    {sample}-{loss}-{backend}.npy   converged sound-speed map
    losses-{backend}.json           all-loss metrics  -> {loss:   metrics}
    phantoms-{backend}.json         all-phantoms      -> {sample: metrics}
"""

import gc
import json
from pathlib import Path

import numpy as np

# Aviable cost functions
LOSSES = ["sb", "lc", "cf", "pe"]

# --- Experiment 1: all losses on a single homogeneous phantom ----------------
ALL_LOSS_SAMPLE = "1540" 
ALL_LOSS_N_ITERS = 100

# --- Experiment 2: phase error on every phantom ------------------------------
ALL_PHANTOMS_LOSS = "pe"
ALL_PHANTOMS_N_ITERS = 300

BACKENDS = ["torch", "jax"]

RESULTS_DIR = Path("results")


def _run_one_torch(sample: str, loss: str, n_iters: int) -> dict:
    from TorchDbua.conf import DBUAConfig
    from TorchDbua.dbua import main

    config = DBUAConfig(sample=sample, loss_name=loss, n_iters=n_iters, plot=False)
    c, metrics = main(config)
    np.save(RESULTS_DIR / f"{sample}-{loss}-torch.npy", c.cpu().numpy())
    return metrics


def _run_one_jax(sample: str, loss: str, n_iters: int) -> dict:
    import JaxDbua.dbua as jax_dbua

    # main() reads N_ITERS as a module global at call time; override it here.
    jax_dbua.N_ITERS = n_iters
    c, metrics = jax_dbua.main(sample, loss, plot=False)
    np.save(RESULTS_DIR / f"{sample}-{loss}-jax.npy", np.array(c))
    return metrics


_RUNNERS = {"torch": _run_one_torch, "jax": _run_one_jax}


def _safe(runner, sample: str, loss: str, n_iters: int) -> dict:
    """Run one optimization, capturing failures so the sweep keeps going."""
    try:
        return runner(sample, loss, n_iters)
    except Exception as e:
        return {"error": str(e)}


def run_backend(backend: str, samples: list[str]) -> tuple[dict, dict]:
    """Run both experiments for one backend. Returns (losses_res, phantoms_res)."""
    runner = _RUNNERS[backend]

    # Experiment 1: all losses on the single homogeneous phantom.
    losses_res = {
        loss: _safe(runner, ALL_LOSS_SAMPLE, loss, ALL_LOSS_N_ITERS)
        for loss in LOSSES
    }

    # Experiment 2: phase error on every phantom.
    phantoms_res = {
        sample: _safe(runner, sample, ALL_PHANTOMS_LOSS, ALL_PHANTOMS_N_ITERS)
        for sample in samples
    }

    return losses_res, phantoms_res


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

    samples = list(DBUAConfig().ctrue.keys())

    for backend in BACKENDS:
        losses_res, phantoms_res = run_backend(backend, samples)
        with open(RESULTS_DIR / f"losses-{backend}.json", "w") as f:
            json.dump(losses_res, f, indent=2)
        with open(RESULTS_DIR / f"phantoms-{backend}.json", "w") as f:
            json.dump(phantoms_res, f, indent=2)
        free_gpu()
