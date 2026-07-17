from hdf5storage import loadmat
import torch

from TorchDbua.das import das

def load_dataset(data_path):
    mdict = loadmat(data_path) # f"{DATA_DIR}/{sample}.mat")
    iqdata = mdict["iqdata"]
    fs = mdict["fs"][0, 0]  # Sampling frequency
    fd = mdict["fd"][0, 0]  # Demodulation frequency
    dsf = mdict["dsf"][0, 0]  # Downsampling factor
    t = mdict["t"]  # time vector
    t0 = mdict["t0"]  # time zero of transmit
    elpos = mdict["elpos"]  # element position
    return iqdata, t0, fs, fd, elpos, dsf, t

def bmode_title(sb_loss, cf_loss, pe_loss, c):
    """B-mode panel title for sound-speed map ``c``.

    Free function with explicit dependencies (mirrors ``makeImage``): the three
    data-term loss callables ``c -> scalar`` (no TV, matching the original
    title). Bind them with ``functools.partial`` so callers invoke
    ``bmode_title(c)``. Evaluate only inside a ``torch.no_grad()`` context --
    torch.compile guards on grad-mode, so a grad-enabled call would force a
    second recompile of the beamformer just to draw the title.
    """
    return "SB: %.2f, CF: %.3f, PE: %.3f" % (
        float(sb_loss(c)), float(cf_loss(c)), float(pe_loss(c)))


def makeImage(iqdata, t0, fs, fd, tof_image, c):
    """Log-magnitude B-mode image for sound-speed map ``c``.

    Free function with explicit dependencies (no manager coupling): ``tof_image``
    is a callable ``c -> [elements, *pixdims]`` giving the image-geometry
    time-of-flight, and the beamsum uses the acquisition IQ / timing
    (``iqdata``, ``t0``, ``fs``, ``fd``). Bind the fixed args with
    ``functools.partial`` so callers can invoke ``make_image(c)``.
    """
    t = tof_image(c)
    return torch.abs(das(iqdata, t - t0, t, fs, fd))
