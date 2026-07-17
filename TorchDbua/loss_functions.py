import torch
from functools import partial
from TorchDbua.das import das

def _nansum(x, dim):
    """nansum that also supports complex tensors (torch.nansum does not).

    Matches jnp.nansum: an element is dropped (treated as 0) if either its real
    or imaginary part is NaN.
    """
    if x.is_complex():
        return torch.sum(torch.where(torch.isnan(x), torch.zeros_like(x), x), dim=dim)
    return torch.nansum(x, dim=dim)

@partial(torch.compile)
def lag_one_coherence(iq, t_tx, t_rx, fs, fd):
    """
    Lag-one coherence of the receive aperture (DOI: 10.1109/TUFFC.2018.2855653).
    The LOC measures the quality of a signal relative to its noise, and can be
    used to select acoustic output.

    Args:
        iq:   [ntx, nrx, nsamps] complex  Raw baseband IQ channel data. Transposed
                                          internally to place the rx aperture first.
        t_tx: [ntx, *pixdims]             Transmit time-of-flight delays (seconds):
                                          travel time from each tx element/event to
                                          each pixel. Summed over in the beamsum.
        t_rx: [nrx, *pixdims]             Receive time-of-flight delays (seconds):
                                          travel time from each pixel to each rx
                                          element. Kept per-channel (A=eye) so
                                          coherence can be measured across rx.
        fs:   scalar                      Sampling frequency (Hz); converts delays
                                          to sample indices via fs * t.
        fd:   scalar                      Demodulation/center frequency (Hz) for the
                                          baseband phase rotation. Use 0 for RF data.

    Returns:
        ncc: [*pixdims]  Normalized lag-one correlation coefficient per pixel.
    """
    iq = torch.transpose(iq, 0, 1)  # Place rx aperture in 0-th index
    rxdata = das(iq, t_rx, t_tx, fs, fd, torch.eye(iq.shape[0]))  # Get rx channel data
    # Compute the correlation coefficient
    xy = torch.real(_nansum(rxdata[:-1] * torch.conj(rxdata[1:]), 0))
    xx = torch.nansum(torch.abs(rxdata[:-1]) ** 2, axis=0)
    yy = torch.nansum(torch.abs(rxdata[1:]) ** 2, axis=0)
    ncc = xy / torch.sqrt(xx * yy)
    return ncc


@partial(torch.compile)
def coherence_factor(iq, t_tx, t_rx, fs, fd):
    """
    The coherence factor of the receive aperture (DOI: 10.1121/1.410562).
    The CF is a focusing criterion used to measure the amount of aberration in
    an image.
    """
    iq = iq.permute(1, 0, 2)  # Place rx aperture in 0-th index
    rxdata = das(iq, t_rx, t_tx, fs, fd, torch.eye(iq.shape[0]))  # Get rx channel data
    num = torch.abs(_nansum(rxdata, 0))
    den = torch.nansum(torch.abs(rxdata), axis=0)
    return num / den


@partial(torch.compile)
def speckle_brightness(iq, t_tx, t_rx, fs, fd):
    """
    The speckle brightness criterion (DOI: 10.1121/1.397889)
    Speckle brightness can be used to measure the focusing quality.
    """
    return torch.nanmean(torch.abs(das(iq, t_tx, t_rx, fs, fd)))


@torch.compile
def total_variation(c):
    """
    Total variation of sound speed map in x and z.
    The sound speed map c should be specified as a 2D matrix of size [nx, nz]
    """
    tvx = torch.nanmean(torch.square(torch.diff(c, axis=0)))
    tvz = torch.nanmean(torch.square(torch.diff(c, axis=1)))
    return tvx + tvz


@partial(torch.compile)
def phase_error(iq, t_tx, t_rx, fs, fd, thresh=0.9):
    """
    The phase error between translating transmit and receive apertures.
    This error is closesly related to the "Translated Transmit Apertures" algorithm
    (DOI: 10.1109/58.585209), where translated transmit and receive apertures
    with common midpoint should have perfect speckle correlation by the van
    Cittert Zernike theorem (DOI: 10.1121/1.418235). High correlation will
    result in high-quality phase shift estimates (DOI: 10.1121/10.0000809).
    CUTE also takes a similar approach (DOI: 10.1016/j.ultras.2020.106168),
    but in the angular basis instead of the element basis.
    """
    # Compute the IQ data for given transmit and receive subapertures.
    # The IQ data matrix will look as follows:
    #               (Tx index, Rx index)
    #   A B C    A: (2, 0)   B: (2, 1)   C: (2, 2)
    #   D E F    D: (1, 0)   E: (1, 1)   F: (1, 2)
    #   G H I    G: (0, 0)   H: (0, 1)   I: (0, 2)
    # The diagonals correspond tx/rx pairs with common midpoints, e.g.:
    #   A, E, and I have a midpoint at 1.
    #   D and H have a midpoint at 0.5.
    #   G has a midpoint at 0.
    #   B and F have a midpoint at 1.5.
    #   C has a midpoint at 2.
    #
    # We create tx and rx subapertures of size 2*halfsa+1 elements, with
    # spacing determined by dx. These are made using das_subap.
    nrx, ntx, nsamps = iq.shape
    mask = torch.zeros((nrx, ntx))
    halfsa = 8  # Half of a subaperture
    dx = 1  # Subaperture increment
    for diag in range(-halfsa, halfsa + 1):
        mask = mask + torch.diag(torch.ones((ntx - abs(diag),)), diag)
    mask = mask[halfsa : mask.shape[0] - halfsa : dx]
    At = torch.flip(mask, dims=[0])
    Ar = mask
    iqfoc = das(iq, t_tx, t_rx, fs, fd, At, Ar)

    # Now compute the correlation between neighboring pulse-echo signals with
    # common midpoints. If <A,B> is the correlation between A and B, we want
    #   <A, E>, <E, I>, <B, F>, <D, H>. The corners are naturally cut off.
    xy = iqfoc[:-1, :-1] * torch.conj(iqfoc[+1:, +1:])
    xx = iqfoc[:-1, :-1] * torch.conj(iqfoc[:-1, :-1])
    yy = iqfoc[+1:, +1:] * torch.conj(iqfoc[+1:, +1:])
    # Use the "double where" trick to remove correlations with only one signal
    valid1 = (iqfoc[:-1, :-1] != 0) & (iqfoc[1:, 1:] != 0)
    xy = torch.where(valid1, torch.where(valid1, xy, 0), 0)
    xx = torch.where(valid1, torch.where(valid1, xx, 0), 0)
    yy = torch.where(valid1, torch.where(valid1, yy, 0), 0)
    # Determine where the correlation coefficient is high enough to use
    xy = torch.sum(xy, axis=-1)  # Sum over kernel
    xx = torch.sum(xx, axis=-1)  # Sum over kernel
    yy = torch.sum(yy, axis=-1)  # Sum over kernel
    ccsq = torch.square(torch.abs(xy)) / (torch.abs(xx) * torch.abs(yy))
    valid2 = ccsq > thresh * thresh
    xy = torch.where(valid2, torch.where(valid2, xy, 0), 0)
    # Convert
    xy = torch.flip(xy, dims=[0])  # Anti-diagonal --> diagonal
    xy = torch.reshape(xy, (*xy.shape[:2], -1))
    xy = xy.permute(2, 0, 1)  # Place subap dimensions inside
    xy = torch.triu(xy) + torch.conj(torch.tril(xy)).permute(0, 2, 1)
    dphi = torch.angle(xy)  # Compute the phase shift.
    return dphi
