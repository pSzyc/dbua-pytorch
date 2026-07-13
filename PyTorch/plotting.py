import matplotlib.pyplot as plt
import torch
import time
import numpy as np

from PyTorch.processing import makeImage, loss

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

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
    plt.savefig(f"images/losses_torch_{sample}.png")
    plt.clf()

def imagesc(xc, y, img, dr, **kwargs):
    """MATLAB style imagesc"""
    dx = xc[1] - xc[0]
    dy = y[1] - y[0]
    ext = [xc[0] - dx / 2, xc[-1] + dx / 2, y[-1] + dy / 2, y[0] - dy / 2]
    im = plt.imshow(img, vmin=dr[0], vmax=dr[1], extent=ext, **kwargs)
    plt.colorbar()
    return im

def _np(x):
    """Detach a (possibly tensor) value to a host numpy array for plotting."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _display_images(c, bmax, nxi, nzi, nxc, nzc):
    """Log-compressed B-mode image and reshaped sound-speed image for plotting."""
    b = _np(makeImage(c))
    bimg = b / bmax
    bimg = bimg + 1e-10 * (bimg == 0)  # Avoid nans
    bimg = 20 * np.log10(bimg)
    bimg = np.reshape(bimg, (nxi, nzi)).T
    cimg = np.reshape(_np(c), (nxc, nzc)).T
    return bimg, cimg


def _bmode_title(c):
    return "SB: %.2f, CF: %.3f, PE: %.3f" % (
        float(loss(c, "sb")), float(loss(c, "cf")), float(loss(c, "pe")))


def _sos_title(i, cimg, c_true):
    if c_true > 0:  # When ground truth is provided, show the error
        return "Iteration %d: MAE %.2f" % (i, np.mean(np.abs(cimg - c_true)))
    return "Iteration %d: Mean value %.2f" % (i, np.mean(cimg))


@torch.no_grad()
def createFigure(c, i, c_true, xi, zi, xc, zc, nxi, nzi, nxc, nzc):
    """Create the two-panel (B-mode + sound speed) figure once and return the
    handles needed to update it in place on subsequent iterations."""

    # Create the image axes for plotting
    ximm = _np(xi[:, 0] * 1e3)
    zimm = _np(zi[0, :] * 1e3)
    xcmm = _np(xc * 1e3)
    zcmm = _np(zc * 1e3)
    bdr = [-45, +5]
    cdr = np.array([-50, +50]) + \
        c_true if c_true > 0 else [1400, 1600]
    cmap = "seismic" if c_true > 0 else "jet"

    # Report the fps of the torch beamformer
    nrep = 30
    tic = time.perf_counter_ns()
    for _ in range(nrep):
        _ = makeImage(c)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    toc = time.perf_counter_ns()
    print("torchbf runs at %.1f fps." % (nrep / ((toc - tic) * 1e-9)))

    bmax = np.max(_np(makeImage(c)))
    bimg, cimg = _display_images(c, bmax, nxi, nzi, nxc, nzc)

    fig, _ = plt.subplots(1, 2, figsize=[9, 4])
    plt.subplot(121)
    hbi = imagesc(ximm, zimm, bimg, bdr, cmap="bone", interpolation="bicubic")
    hbt = plt.title(_bmode_title(c))
    plt.xlim(ximm[0], ximm[-1])
    plt.ylim(zimm[-1], zimm[0])
    plt.subplot(122)
    hci = imagesc(xcmm, zcmm, cimg, cdr, cmap=cmap, interpolation="bicubic")
    hct = plt.title(_sos_title(i, cimg, c_true))
    plt.xlim(ximm[0], ximm[-1])
    plt.ylim(zimm[-1], zimm[0])
    fig.tight_layout()

    return fig, hbi, hci, hbt, hct, bmax


@torch.no_grad()
def updateFigure(c, i, c_true, sample, nxi, nzi, nxc, nzc, handles):
    """Update the figure created by createFigure in place and save it to disk."""
    fig, hbi, hci, hbt, hct, bmax = handles

    bimg, cimg = _display_images(c, bmax, nxi, nzi, nxc, nzc)

    hbi.set_data(bimg)
    hci.set_data(cimg)
    hbt.set_text(_bmode_title(c))
    hct.set_text(_sos_title(i, cimg, c_true))

    fig.savefig(f"scratch/{sample}.png")
