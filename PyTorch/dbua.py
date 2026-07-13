import numpy as np
import torch

from tqdm import tqdm

from PyTorch.conf import DBUAConfig
from PyTorch.plotting import plot_errors_vs_sound_speeds, createFigure, updateFigure
from PyTorch.processing import load_dataset
from PyTorch.execution_manager import ExecutionManager


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
    plot_errors_vs_sound_speeds(c0, dsb, dlc, dcf, dpe, config.sample)

    return torch.nn.Parameter(const_c(c0[int(np.argmin(dpe))]))

def optimization_loop(optimizer, c, execution_manager: ExecutionManager):
    config = execution_manager.get_config()
    c_true = config.ctrue[config.sample]
    nxc, nzc = config.sound_speed_nxc, config.sound_speed_nzc

    xi, zi = execution_manager.get_xi_zi()
    xc, zc = execution_manager.get_xc_zc()
    nxi, nzi = execution_manager.get_nxi_nzi()

    # Create the figure once, outside the optimization loop
    handles = createFigure(c, 0, c_true, xi, zi, xc, zc, nxi, nzi, nxc, nzc)

    for i in tqdm(range(config.n_iters)):
        optimizer.zero_grad()
        objective = execution_manager.loss(c, config.loss_name)
        objective.backward()
        optimizer.step()
        if i % 10 == 0:
            # Reuse the existing figure
            updateFigure(c, i + 1, c_true, config.sample, nxi, nzi, nxc, nzc, handles)


def main(config: DBUAConfig):

    assert (
        config.sample in config.ctrue
    ), f'The data sample string was "{config.sample}".\
                            \nOptions are {", ".join(config.ctrue.keys()).lstrip(" ,")}.'

    # Get IQ data, time zeros, sampling and demodulation frequency, and element positions
    iqdata, t0, fs, fd, elpos, _, _ = load_dataset(config.sample)

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
    return c.detach()


if __name__ == "__main__":
    config = DBUAConfig()
    c = main(config)
    np.save(c, f"{config.sample}-torch.npy")

    # # Run all examples
    # for sample in config.ctrue.keys():
    #     print(sample)
    #     main(DBUAConfig(sample=sample))
