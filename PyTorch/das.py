from functools import partial
import torch
from torch.utils.checkpoint import checkpoint


@partial(torch.compile)
def das(iqraw, tA, tB, fs, fd, A=None, B=None, apoA=1, apoB=1, interp="cubic"):
    """
    Delay-and-sum IQ data according to a given time delay profile.
    @param iqraw   [na, nb, nsamps]  Raw IQ data (baseband)
    @param tA      [na, *pixdims]    Time delays to apply to dimension 0 of iq
    @param tB      [nb, *pixdims]    Time delays to apply to dimension 1 of iq
    @param fs      scalar            Sampling frequency to convert from time to samples
    @param fd      scalar            Demodulation frequency (0 for RF modulated data)
    @param A       [*na_out, na]     Linear combination of dimension 0 of iqraw
    @param B       [*nb_out, nb]     Linear combination of dimension 1 of iqraw
    @param apoA    [na, *pixdims]    Broadcastable apodization on dimension 0 of iq
    @param apoB    [nb, *pixdims]    Broadcastable apodization on dimension 1 of iq
    @param interp  string            Interpolation method to use
    @return iqfoc  [*na_out, *nb_out, *pixel_dims]   Beamformed IQ data

    The tensors A and B specify how to combine the "elements" in dimensions 0 and 1 of
    iqraw via a tensor contraction. If A or B are None, they default to a vector of ones,
    i.e. a simple sum over all elements. If A or B are identity matrices, the result will
    be the delayed-but-not-summed output. A and B can be arbitrary tensors of arbitrary
    size, as long as the inner most dimension matches na or nb, respectively. Another
    alternative use case is for subaperture beamforming.

    Note that via acoustic reciprocity, it does not matter whether a or b correspond to
    the transmit or receive "elements".
    """
    na, nb = iqraw.shape[0], iqraw.shape[1]

    # The default linear combination is to sum all elements.
    if A is None:
        A = torch.ones((na,), dtype=iqraw.real.dtype, device=iqraw.device)
    if B is None:
        B = torch.ones((nb,), dtype=iqraw.real.dtype, device=iqraw.device)

    # Choose the interpolating function
    fints = {
        "nearest": interp_nearest,
        "linear": interp_linear,
        "cubic": interp_cubic,
        "lanczos3": lambda x, t: interp_lanczos(x, t, nlobe=3),
        "lanczos5": lambda x, t: interp_lanczos(x, t, nlobe=5),
    }
    fint = fints[interp]

    # Baseband interpolator
    def bbint(iq, t):
        iqfoc = fint(iq, fs * t)
        return iqfoc * torch.exp(2j * torch.pi * fd * t)

    # # Delay-and-sum beamforming (vmap inner, vmap outer)
    # # This method uses vmap to push both the inner and outer loops into XLA, which uses
    # # uses more memory, but can take advantage of XLA's parallelization.  However, it is
    # # slower when memory bandwidth is a bottleneck.
    # def das_b(iq_i, tA_i):
    #     return jnp.tensordot(B, vmap(bbint)(iq_i, tA_i + tB) * apoB, (-1, 0))
    # return jnp.tensordot(A, vmap(das_b)(iqraw, tA) * apoA, (-1, 0))

    # Delay-and-sum beamforming (vmap inner, loop outer).
    # Vectorize over the b aperture, but walk the a aperture one row at a time so the
    # full [na, nb, *pixdims] round-trip tensor is never materialized. Each row is
    # gradient-checkpointed (recomputed in backward instead of stored) to bound
    # activation memory -- the torch analog of JAX's @checkpoint + lax.map.
    #
    # NOTE: Python's builtin map() is NOT jax.lax.map: map(das_b, (iqraw, tA)) would
    # call das_b(iqraw) then das_b(tA). Iterate the leading (a) axis explicitly instead.
    def das_b(x):
        iq_i, tA_i = x
        return torch.tensordot(B, torch.vmap(bbint)(iq_i, tA_i + tB) * apoB, (-1, 0))

    rows = [
        checkpoint(das_b, (iqraw[i], tA[i]), use_reentrant=False)
        for i in range(na)
    ]
    stacked = torch.stack(rows, dim=0) * apoA
    return torch.tensordot(A, stacked, dims=([-1], [0]))


def safe_access(x: torch.Tensor, s: torch.Tensor):
    """Safe gather of x along its last axis at (possibly out-of-bounds) indices s.
    @param x: [..., nsamps] values to interpolate (batched over the leading dims)
    @param s: [..., P] sample indices to access; out-of-bounds indices return 0
    @return: [..., P] gathered values (0 where s is out of bounds)
    """
    nsamps = x.shape[-1]
    s = s.long()
    valid = (s >= 0) & (s < nsamps)
    vals = torch.gather(x, -1, s.clamp(0, nsamps - 1))
    return torch.where(valid, vals, torch.zeros_like(vals))


def interp_nearest(x: torch.Tensor, si: torch.Tensor):
    """1D nearest neighbor interpolation.
    @param x: [..., nsamps] values to interpolate
    @param si: [..., P] indices to interpolate at
    @return: Interpolated signal
    """
    idx = torch.round(si).clamp(0, x.shape[-1] - 1).long()
    return torch.gather(x, -1, idx)


def interp_linear(x: torch.Tensor, si: torch.Tensor):
    """1D linear interpolation.
    @param x: [..., nsamps] values to interpolate
    @param si: [..., P] indices to interpolate at
    @return: Interpolated signal
    """
    s = torch.trunc(si)  # Integer part (toward zero, matching jnp.modf)
    f = si - s  # Fractional part
    x0 = safe_access(x, s + 0)
    x1 = safe_access(x, s + 1)
    return (1 - f) * x0 + f * x1


def interp_cubic(x: torch.Tensor, si: torch.Tensor):
    """1D cubic Hermite interpolation.
    @param x: [..., nsamps] values to interpolate
    @param si: [..., P] indices to interpolate at
    @return: Interpolated signal
    """
    s = torch.trunc(si)  # Integer part (toward zero, matching jnp.modf)
    f = si - s  # Fractional part
    # Values
    x0 = safe_access(x, s - 1)
    x1 = safe_access(x, s + 0)
    x2 = safe_access(x, s + 1)
    x3 = safe_access(x, s + 2)
    # Coefficients
    a0 = 0 + f * (-1 + f * (+2 * f - 1))
    a1 = 2 + f * (+0 + f * (-5 * f + 3))
    a2 = 0 + f * (+1 + f * (+4 * f - 3))
    a3 = 0 + f * (+0 + f * (-1 * f + 1))
    return (a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3) / 2


def _lanczos_helper(x, nlobe=3):
    """Lanczos kernel"""
    a = (nlobe + 1) / 2
    return torch.where(torch.abs(x) < a, torch.sinc(x) * torch.sinc(x / a),
                       torch.zeros_like(x))


def interp_lanczos(x: torch.Tensor, si: torch.Tensor, nlobe=3):
    """Lanczos interpolation.
    @param x: [..., nsamps] values to interpolate
    @param si: [..., P] indices to interpolate at
    @param nlobe: Number of lobes of the sinc function (e.g., 3 or 5)
    @return: Interpolated signal
    """
    s = torch.trunc(si)  # Integer part (toward zero, matching jnp.modf)
    f = si - s  # Fractional part
    x0 = safe_access(x, s - 1)
    x1 = safe_access(x, s + 0)
    x2 = safe_access(x, s + 1)
    x3 = safe_access(x, s + 2)
    a0 = _lanczos_helper(f + 1, nlobe)
    a1 = _lanczos_helper(f + 0, nlobe)
    a2 = _lanczos_helper(f - 1, nlobe)
    a3 = _lanczos_helper(f - 2, nlobe)
    return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3
