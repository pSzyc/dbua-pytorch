import json
from functools import partial
from pathlib import Path

import numpy as np
import torch

from tqdm import tqdm

from TorchDbua.conf import DBUAConfig
from TorchDbua.plotting import plot_errors_vs_sound_speeds, createFigure, updateFigure
from TorchDbua.processing import load_dataset, makeImage, bmode_title
from TorchDbua.execution_manager import ExecutionManager


def get_optimal_sos(config: DBUAConfig, execution_manager: ExecutionManager):
    const_c = lambda val: torch.full(
        size=(config.sound_speed_nxc, config.sound_speed_nzc),
        fill_value=float(val),
        dtype=config.real_dtype,
        device=config.device
    )

    # find optimal global sound speed for initalization
    c0 = np.linspace(1340, 1740, 201)
    with torch.no_grad():
        dsb = np.array([float(execution_manager.sb_loss(const_c(cc))) for cc in c0])
        dlc = np.array([float(execution_manager.lc_loss(const_c(cc))) for cc in c0])
        dcf = np.array([float(execution_manager.cf_loss(const_c(cc))) for cc in c0])
        dpe = np.array([float(execution_manager.pe_loss(const_c(cc))) for cc in c0])

    # Plot global sound speed error
    if config.plot:
        plot_errors_vs_sound_speeds(c0, dsb, dlc, dcf, dpe, config.sample)

    return torch.nn.Parameter(const_c(c0[int(np.argmin(dpe))]))

def optimization_loop(optimizer, c, execution_manager: ExecutionManager):
    config = execution_manager.get_config()
    c_true = config.ctrue[config.sample]
    nxc, nzc = config.sound_speed_nxc, config.sound_speed_nzc

    xi, zi = execution_manager.get_xi_zi()
    xc, zc = execution_manager.get_xc_zc()
    nxi, nzi = execution_manager.get_nxi_nzi()

    if config.plot:
        # Bind the fixed acquisition data / geometry into plain c -> value callables
        # for the plotting layer, so it stays decoupled from the ExecutionManager.
        em = execution_manager
        make_image = partial(makeImage, em.iqdata, em.t0, em.fs, em.fd, em._tof_image)
        make_title = partial(bmode_title, em.sb_loss, em.cf_loss, em.pe_loss)

        # Create the figure once, outside the optimization loop
        handles = createFigure(make_image, make_title, c, 0, c_true, xi, zi, xc, zc, nxi, nzi, nxc, nzc)

    for i in tqdm(range(config.n_iters)):
        optimizer.zero_grad()
        objective = execution_manager.loss(c, config.loss_name)
        objective.backward()
        optimizer.step()
        if config.plot and i % 10 == 0:
            # Reuse the existing figure
            updateFigure(make_image, make_title, c, i + 1, c_true, config.sample, nxi, nzi, nxc, nzc, handles)


def compute_final_metrics(c, execution_manager: ExecutionManager, config: DBUAConfig):
    """Evaluate the converged sound-speed map ``c``.

    Returns the final cost-function errors (all four focusing losses on ``c``)
    and, for uniform phantoms (whose ground-truth speed is a known constant),
    the reconstruction error of ``c`` against that constant.
    """
    with torch.no_grad():
        metrics = {
            "sb": float(execution_manager.sb_loss(c)),
            "lc": float(execution_manager.lc_loss(c)),
            "cf": float(execution_manager.cf_loss(c)),
            "pe": float(execution_manager.pe_loss(c)),
        }

    c_true = config.ctrue[config.sample]
    if c_true > 0:  # uniform phantom: error w.r.t. the known constant real value
        c_np = c.detach().cpu().numpy()
        abs_err = np.abs(c_np - c_true).ravel()
        metrics["c_true"] = float(c_true)
        metrics["mean_c"] = float(np.mean(c_np))
        metrics["mae"] = float(np.mean(abs_err))
        # Standard error of the MAE: sample std of per-node abs errors / sqrt(N).
        metrics["mae_se"] = float(np.std(abs_err, ddof=1) / np.sqrt(abs_err.size))

    return metrics


def main(config: DBUAConfig):

    assert (
        config.sample in config.ctrue
    ), f'The data sample string was "{config.sample}".\
                            \nOptions are {", ".join(config.ctrue.keys()).lstrip(" ,")}.'

    # Get IQ data, time zeros, sampling and demodulation frequency, and element positions
    iqdata, t0, fs, fd, elpos, _, _ = load_dataset(config.data_dir / f"{config.sample}.mat")

    # Move acquisition data onto the compute device as torch tensors.
    fs, fd = float(fs), float(fd)
    iqdata = torch.as_tensor(np.ascontiguousarray(iqdata), device=config.device)
    iqdata = iqdata.to(torch.complex64 if iqdata.is_complex() else config.real_dtype)
    t0 = torch.as_tensor(np.asarray(t0), dtype=config.real_dtype, device=config.device)
    elpos = torch.as_tensor(np.asarray(elpos), dtype=config.real_dtype, device=config.device)

    execution_manager = ExecutionManager(
        iqdata=iqdata,
        t0=t0,
        fs=fs,
        fd=fd,
        elpos=elpos,
        config=config,
    )

    c = get_optimal_sos(config, execution_manager)

    # Create the optimizer (AMSGrad variant of Adam, matching optax.amsgrad)
    optimizer = torch.optim.Adam([c], lr=config.learning_rate, amsgrad=True)
    optimization_loop(optimizer, c, execution_manager)

    # Final sound-speed map plus its cost-function / reconstruction errors
    c = c.detach()
    metrics = compute_final_metrics(c, execution_manager, config)
    return c, metrics


if __name__ == "__main__":
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Run all examples, keeping going if any single sample fails
    results = {}
    for sample in DBUAConfig().ctrue.keys():
        try:
            config = DBUAConfig(sample=sample)
            c, metrics = main(config)
            np.save(results_dir / f"{sample}-torch.npy", c.cpu().numpy())
            results[sample] = metrics
        except Exception as e:
            results[sample] = {"error": str(e)}

    # Write the final metrics / errors as JSON
    with open(results_dir / "results-torch.json", "w") as f:
        json.dump(results, f, indent=2)
