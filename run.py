"""Run the DBUA optimization with the proposed loss over every phantom.

A single experiment: the proposed loss (``LOSS``, phase error) driving
reconstruction on every phantom -- homogeneous (constant-speed) and
heterogeneous alike -- on both backends, ``N_ITERS`` iterations.

For each run we keep the full metrics dict ``main()`` returns: the four
focusing-loss scores evaluated on the converged map, plus -- for homogeneous
phantoms, whose ground-truth speed is a known constant -- the MAE (+/- standard
error) of the recovered sound speed. Heterogeneous phantoms have no single
ground-truth speed, so they simply carry no MAE.

Runs with ``plot=True`` so each backend saves its own SoS / B-mode figure
(``scratch/{sample}_{loss}_{backend}.png``) and loss-vs-speed survey.

Results layout (under ``results/``)
    {sample}-{loss}-{backend}.npy   converged sound-speed map
    phantoms-{backend}.json         {sample: {sb, lc, cf, pe[, c_true, mean_c, mae, mae_se]}}

``analyse.py`` turns ``phantoms-{backend}.json`` into the comparison table.
"""

import gc
import json
from pathlib import Path

import numpy as np

# Loss driving the optimization (the proposed method, phase error).
LOSS = "pe"

# Gradient-descent iterations per run.
N_ITERS = 300

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


def run_phantoms(backend: str, samples: list[str]) -> dict:
    """Run the proposed loss on every phantom for one backend.

    Returns ``{sample: metrics}`` -- the full metrics dict ``main()`` returns
    (focusing scores + MAE where a ground-truth constant exists).
    """
    runner = _RUNNERS[backend]
    return {
        sample: _safe(runner, sample, LOSS, N_ITERS)
        for sample in samples
    }


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
        phantoms = run_phantoms(backend, samples)
        with open(RESULTS_DIR / f"phantoms-{backend}.json", "w") as f:
            json.dump(phantoms, f, indent=2)
        free_gpu()
