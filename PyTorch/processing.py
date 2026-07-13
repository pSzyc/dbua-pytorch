from hdf5storage import loadmat
import torch

from PyTorch.das import das

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

def makeImage(self, iqdata, t0, fs, fd):
    t = self.tof_image(c)
    return torch.abs(das(iqdata, t - t0, t, fs, fd))
