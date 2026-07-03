from pathlib import Path
import numpy as np
import torch
from PyTorch.das import das
from PyTorch.paths import time_of_flight
from hdf5storage import loadmat
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from PyTorch.losses import (
    lag_one_coherence,
    coherence_factor,
    phase_error,
    total_variation,
    speckle_brightness,
)
import time


N_ITERS = 301
LEARNING_RATE = 10
ASSUMED_C = 1540  # [m/s]

# B-mode limits in m
BMODE_X_MIN = -12e-3
BMODE_X_MAX = 12e-3
BMODE_Z_MIN = 0e-3
BMODE_Z_MAX = 40e-3

# Sound speed grid in m
SOUND_SPEED_X_MIN = -12e-3
SOUND_SPEED_X_MAX = 12e-3
SOUND_SPEED_Z_MIN = 0e-3
SOUND_SPEED_Z_MAX = 40e-3
SOUND_SPEED_NXC = 19
SOUND_SPEED_NZC = 31

# Phase estimate kernel size in samples
NXK, NZK = 5, 5

# Phase estimate patch grid size in samples
NXP, NZP = 17, 17
PHASE_ERROR_X_MIN = -20e-3
PHASE_ERROR_X_MAX = 20e-3
PHASE_ERROR_Z_MIN = 4e-3
PHASE_ERROR_Z_MAX = 44e-3

# Loss options
# -"pe" for phase error
# -"sb" for speckle brightness
# -"cf" for coherence factor
# -"lc" for lag one coherence

LOSS = "pe"

# Data options:
# (Constant Phantoms)
# - 1420
# - 1465
# - 1480
# - 1510
# - 1540
# - 1555
# - 1570
# (Heterogeneous Phantoms)
# - inclusion
# - inclusion_layer
# - four_layer
# - two_layer
# - checker2
# - checker8

SAMPLE = "checker2"

CTRUE = {
    "1420": 1420,
    "1465": 1465,
    "1480": 1480,
    "1510": 1510,
    "1540": 1540,
    "1555": 1555,
    "1570": 1570,
    "inclusion": 0,
    "inclusion_layer": 0,
    "four_layer": 0,
    "two_layer": 0,
    "checker2": 0,
    "checker8": 0
}


# Refocused plane wave datasets from base dataset directory
DATA_DIR = Path("./data")

# Compute device and real dtype. CUDA is the primary target; fall back to MPS
# (mac dev boxes) or CPU. Sound speed / geometry are float32; IQ data is complex64.
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
REAL_DTYPE = torch.float32


def _np(x):
    """Detach a (possibly tensor) value to a host numpy array for plotting."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def imagesc(xc, y, img, dr, **kwargs):
    """MATLAB style imagesc"""
    dx = xc[1] - xc[0]
    dy = y[1] - y[0]
    ext = [xc[0] - dx / 2, xc[-1] + dx / 2, y[-1] + dy / 2, y[0] - dy / 2]
    im = plt.imshow(img, vmin=dr[0], vmax=dr[1], extent=ext, **kwargs)
    plt.colorbar()
    return im

def load_dataset(sample):
    mdict = loadmat(f"{DATA_DIR}/{sample}.mat")
    iqdata = mdict["iqdata"]
    fs = mdict["fs"][0, 0]  # Sampling frequency
    fd = mdict["fd"][0, 0]  # Demodulation frequency
    dsf = mdict["dsf"][0, 0]  # Downsampling factor
    t = mdict["t"]  # time vector
    t0 = mdict["t0"]  # time zero of transmit
    elpos = mdict["elpos"]  # element position
    return iqdata, t0, fs, fd, elpos, dsf, t


def plot_errors_vs_sound_speeds(c0, dsb, dlc, dcf, dpe, sample):
    plt.clf()
    plt.plot(c0, dsb, label="Speckle Brightness")
    plt.plot(c0, dlc, label="Lag One Coherence")
    plt.plot(c0, dcf, label="Coherence Factor")
    # divided by 10 for visualization
    plt.plot(c0, dpe / 10, label="Phase Error")
    plt.grid()
    plt.xlabel("Global sound speed (m/s)")
    plt.ylabel("Loss function")
    plt.title(sample)
    plt.legend()
    plt.savefig(f"images/losses_{sample}.png")
    plt.savefig("scratch.png")
    plt.clf()


def main(sample, loss_name):

    assert (
        sample in CTRUE
    ), f'The data sample string was "{sample}".\
                            \nOptions are {", ".join(CTRUE.keys()).lstrip(" ,")}.'

    # Get IQ data, time zeros, sampling and demodulation frequency, and element positions
    iqdata, t0, fs, fd, elpos, _, _ = load_dataset(sample)
    # Move acquisition data onto the compute device as torch tensors.
    fs, fd = float(fs), float(fd)
    iqdata = torch.as_tensor(np.ascontiguousarray(iqdata), device=DEVICE)
    iqdata = iqdata.to(torch.complex64 if iqdata.is_complex() else REAL_DTYPE)
    t0 = torch.as_tensor(np.asarray(t0), dtype=REAL_DTYPE, device=DEVICE)
    elpos = torch.as_tensor(np.asarray(elpos), dtype=REAL_DTYPE, device=DEVICE)
    xe, _, ze = elpos
    wl0 = ASSUMED_C / fd  # wavelength (λ)

    # B-mode image dimensions
    xi = torch.arange(BMODE_X_MIN, BMODE_X_MAX, wl0 / 3, dtype=REAL_DTYPE, device=DEVICE)
    zi = torch.arange(BMODE_Z_MIN, BMODE_Z_MAX, wl0 / 3, dtype=REAL_DTYPE, device=DEVICE)
    nxi, nzi = xi.numel(), zi.numel()
    xi, zi = torch.meshgrid(xi, zi, indexing="ij")

    # Sound speed grid dimensions
    xc = torch.linspace(SOUND_SPEED_X_MIN, SOUND_SPEED_X_MAX, SOUND_SPEED_NXC,
                        dtype=REAL_DTYPE, device=DEVICE)
    zc = torch.linspace(SOUND_SPEED_Z_MIN, SOUND_SPEED_Z_MAX, SOUND_SPEED_NZC,
                        dtype=REAL_DTYPE, device=DEVICE)
    dxc, dzc = xc[1] - xc[0], zc[1] - zc[0]

    # Kernels to use for loss calculations (2λ x 2λ patches)
    xk, zk = torch.meshgrid(
        (torch.arange(NXK, dtype=REAL_DTYPE, device=DEVICE) - (NXK - 1) / 2) * wl0 / 2,
        (torch.arange(NZK, dtype=REAL_DTYPE, device=DEVICE) - (NZK - 1) / 2) * wl0 / 2,
        indexing="ij")

    # Kernel patch centers, distributed throughout the field of view
    xpc, zpc = torch.meshgrid(
        torch.linspace(PHASE_ERROR_X_MIN, PHASE_ERROR_X_MAX, NXP,
                       dtype=REAL_DTYPE, device=DEVICE),
        torch.linspace(PHASE_ERROR_Z_MIN, PHASE_ERROR_Z_MAX, NZP,
                       dtype=REAL_DTYPE, device=DEVICE),
        indexing="ij")

    # Explicit broadcasting. Dimensions will be [elements, pixels, patches]
    xe = torch.reshape(xe, (-1, 1, 1))
    ze = torch.reshape(ze, (-1, 1, 1))
    xp = torch.reshape(xpc, (1, -1, 1)) + torch.reshape(xk, (1, 1, -1))
    zp = torch.reshape(zpc, (1, -1, 1)) + torch.reshape(zk, (1, 1, -1))
    xp = xp + 0 * zp  # Manual broadcasting
    zp = zp + 0 * xp  # Manual broadcasting

    # Compute time-of-flight for each {image, patch} pixel to each element
    def tof_image(c): return time_of_flight(
        xe, ze, xi, zi, xc, zc, c, fnum=0.5, npts=64)

    def tof_patch(c): return time_of_flight(
        xe, ze, xp, zp, xc, zc, c, fnum=0.5, npts=64)

    def makeImage(c):
        t = tof_image(c)
        return torch.abs(das(iqdata, t - t0, t, fs, fd))

    def loss_wrapper(func, c):
        t = tof_patch(c)
        return (func)(iqdata, t - t0, t, fs, fd)

    # Define loss functions
    def sb_loss(c):
        return 1 - loss_wrapper(speckle_brightness, c)

    def lc_loss(c):
        return 1 - torch.mean(loss_wrapper(lag_one_coherence, c))

    def cf_loss(c):
        return 1 - torch.mean(loss_wrapper(coherence_factor, c))

    def pe_loss(c):
        t = tof_patch(c)
        dphi = phase_error(iqdata, t - t0, t, fs, fd)
        valid = dphi != 0
        nan = torch.full_like(dphi, float("nan"))
        dphi = torch.where(valid, torch.where(valid, dphi, nan), nan)
        return torch.nanmean(torch.log1p(torch.square(100 * dphi)))

    def tv(c):
        return total_variation(c) * dxc * dzc

    def loss(c):
        if loss_name == "sb":  # Speckle brightness
            return sb_loss(c) + tv(c) * 1e2
        elif loss_name == "lc":  # Lag one coherence
            return lc_loss(c) + tv(c) * 1e2
        elif loss_name == "cf":  # Coherence factor
            return cf_loss(c) + tv(c) * 1e2
        elif loss_name == "pe":  # Phase error
            return pe_loss(c) + tv(c) * 1e2
        else:
            raise NotImplementedError

    # A constant global-sound-speed map (no grad needed for the survey)
    def const_c(val):
        return torch.full((SOUND_SPEED_NXC, SOUND_SPEED_NZC), float(val),
                          dtype=REAL_DTYPE, device=DEVICE)

    # find optimal global sound speed for initalization
    c0 = np.linspace(1340, 1740, 201)
    with torch.no_grad():
        dsb = np.array([float(sb_loss(const_c(cc))) for cc in c0])
        dlc = np.array([float(lc_loss(const_c(cc))) for cc in c0])
        dcf = np.array([float(cf_loss(const_c(cc))) for cc in c0])
        dpe = np.array([float(pe_loss(const_c(cc))) for cc in c0])
    # Use the sound speed with the optimal phase error to initialize sound speed map
    c = torch.nn.Parameter(const_c(c0[int(np.argmin(dpe))]))

    # Plot global sound speed error
    plot_errors_vs_sound_speeds(c0, dsb, dlc, dcf, dpe, sample)

    # Create the optimizer (AMSGrad variant of Adam, matching optax.amsgrad)
    optimizer = torch.optim.Adam([c], lr=LEARNING_RATE, amsgrad=True)

    # Create the figure writer
    fig, _ = plt.subplots(1, 2, figsize=[9, 4])
    vobj = FFMpegWriter(fps=30)
    vobj.setup(fig, "videos/%s_opt%s.mp4" % (sample, loss_name), dpi=144)

    # Create the image axes for plotting
    ximm = _np(xi[:, 0] * 1e3)
    zimm = _np(zi[0, :] * 1e3)
    xcmm = _np(xc * 1e3)
    zcmm = _np(zc * 1e3)
    bdr = [-45, +5]
    cdr = np.array([-50, +50]) + \
        CTRUE[sample] if CTRUE[sample] > 0 else [1400, 1600]
    cmap = "seismic" if CTRUE[sample] > 0 else "jet"

    # Create a nice figure on first call, update on subsequent calls
    @torch.no_grad()
    def makeFigure(cimg, i, handles=None):
        b = _np(makeImage(cimg))
        if handles is None:
            bmax = np.max(b)
        else:
            hbi, hci, hbt, hct, bmax = handles
        bimg = b / bmax
        bimg = bimg + 1e-10 * (bimg == 0)  # Avoid nans
        bimg = 20 * np.log10(bimg)
        bimg = np.reshape(bimg, (nxi, nzi)).T
        cimg_np = np.reshape(_np(cimg), (SOUND_SPEED_NXC, SOUND_SPEED_NZC)).T

        if handles is None:
            # On the first call, report the fps of the torch beamformer
            nrep = 30
            tic = time.perf_counter_ns()
            for _ in range(nrep):
                _ = makeImage(cimg)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            toc = time.perf_counter_ns()
            print("torchbf runs at %.1f fps." % (nrep / ((toc - tic) * 1e-9)))

            # On the first time, create the figure
            fig.clf()
            plt.subplot(121)
            hbi = imagesc(ximm, zimm, bimg, bdr, cmap="bone",
                          interpolation="bicubic")
            hbt = plt.title(
                "SB: %.2f, CF: %.3f, PE: %.3f" % (
                    float(sb_loss(c)), float(cf_loss(c)), float(pe_loss(c)))
            )
            plt.xlim(ximm[0], ximm[-1])
            plt.ylim(zimm[-1], zimm[0])
            plt.subplot(122)
            hci = imagesc(xcmm, zcmm, cimg_np, cdr, cmap=cmap,
                          interpolation="bicubic")
            if CTRUE[sample] > 0:  # When ground truth is provided, show the error
                hct = plt.title(
                    "Iteration %d: MAE %.2f"
                    % (i, np.mean(np.abs(cimg_np - CTRUE[sample])))
                )
            else:
                hct = plt.title("Iteration %d: Mean value %.2f" %
                                (i, np.mean(cimg_np)))

            plt.xlim(ximm[0], ximm[-1])
            plt.ylim(zimm[-1], zimm[0])
            fig.tight_layout()
            return hbi, hci, hbt, hct, bmax
        else:
            hbi.set_data(bimg)
            hci.set_data(cimg_np)
            hbt.set_text(
                "SB: %.2f, CF: %.3f, PE: %.3f" % (
                    float(sb_loss(c)), float(cf_loss(c)), float(pe_loss(c)))
            )
            if CTRUE[sample] > 0:
                hct.set_text(
                    "Iteration %d: MAE %.2f"
                    % (i, np.mean(np.abs(cimg_np - CTRUE[sample])))
                )
            else:
                hct.set_text("Iteration %d: Mean value %.2f" %
                             (i, np.mean(cimg_np)))

        plt.savefig(f"scratch/{sample}.png")

    # Initialize figure
    handles = makeFigure(c, 0)

    # Optimization loop
    for i in tqdm(range(N_ITERS)):
        optimizer.zero_grad()
        objective = loss(c)
        objective.backward()
        optimizer.step()
        makeFigure(c, i + 1, handles)  # Update figure
        vobj.grab_frame()  # Add to video writer
    vobj.finish()  # Close video writer

    return c.detach()


if __name__ == "__main__":
    main(SAMPLE, LOSS)

    # # Run all examples
    # for sample in CTRUE.keys():
    #     print(sample)
    #     main(sample, LOSS)
