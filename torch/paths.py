#from jax import vmap, jit
#import jax.numpy as jnp
import pytorch as torch
import numpy as np
from functools import partial


@partial(torch.compile) # static are compile time constants
def time_of_flight(x0, z0, x1, z1, xc, zc, c, fnum, npts, Dmin=3e-3):
    """
    Get the time-of-flight from (x0,z0) to (x1,z1) according to the
    speed of sound map c, defined on grid points (xc,zc).
    x0:     [...]       Path origin in x (arbitrary dimensions, broadcasting allowed)
    z0:     [...]       Path origin in z (arbitrary dimensions, broadcasting allowed)
    x1:     [...]       Path finish in x (arbitrary dimensions, broadcasting allowed)
    z1:     [...]       Path finish in z (arbitrary dimensions, broadcasting allowed)
    xc:     [nxc,]      Vector of x-grid points in sound speed definition (c.shape[0],)
    zc:     [nzc,]      Vector of x-grid points in sound speed definition
    c:      [nxc, nzc]  Sound speed map in (xc, zc) coordinates
    fnum:   scalar      f-number to apply
    npts:   scalar      Number of points in time-of-flight line segment
    Dmin:   scalar      Minimum size of the aperture, regardless of f-number
    """
    # Find the path along the path curve, modeled as a straight ray
    # parameterized by t. We will put t in the innermost dimension.
    t_all = torch.arange(1, npts + 1, dtype=c.dtype, device=c.device) / npts
    nx, nz = s.shape

    # Calculate slowness map
    s = 1 / c

    def interpolate(t):
        xt = t * (x1 - x0) + x0  # True spatial location of path in x at t
        zt = t * (z1 - z0) + z0  # True spatial location of path in z at t

        # Convert spatial locations into indices in xc and zc coordinates (in slowness map)
        dxc, dzc = xc[1] - xc[0], zc[1] - zc[0]  # Assume a grid! Grid spacings
        # Get indices of xt, zt in slowness map. Clamp at borders
        xit = torch.clamp((xt - xc[0]) / dxc, 0, s.shape[0] - 1)
        zit = torch.clamp((zt - zc[0]) / dzc, 0, s.shape[1] - 1)
        xi0 = torch.floor(xit)
        zi0 = torch.floor(zit)

        xi1 = xi0 + 1
        zi1 = zi0 + 1
        xi1 = xi1.long().clamp(0, nx - 1)
        zi1 = zi1.long().clamp(0, nz - 1)

        # Interpolate slowness at (xt, zt)
        s00 = s[xi0.astype("int32"), zi0.astype("int32")]
        s10 = s[xi1.astype("int32"), zi0.astype("int32")]
        s01 = s[xi0.astype("int32"), zi1.astype("int32")]
        s11 = s[xi1.astype("int32"), zi1.astype("int32")]
        w00 = (xi1 - xit) * (zi1 - zit)
        w10 = (xit - xi0) * (zi1 - zit)
        w01 = (xi1 - xit) * (zit - zi0)
        w11 = (xit - xi0) * (zit - zi0)
        return s00 * w00 + s10 * w10 + s01 * w01 + s11 * w11

    # Compute the time-of-flight
    dx = torch.abs(x1 - x0)
    dz = torch.abs(z1 - z0)
    dtrue = torch.sqrt(dx**2 + dz**2)
    slowness = torch.func.vmap(interpolate)(t_all)  # bilinear interpolation
    tof = torch.nanmean(slowness, dim=0) * dtrue
    # F-number mask for valid points
    fnum_valid = torch.abs(2 * fnum * dx) <= dz
    # Additionally, set the minimum aperture width to be 3mm
    Dmin_valid = torch.logical_and(dz < Dmin * fnum, dx < Dmin / 2)
    # Total mask for valid regions
    valid = torch.logical_or(fnum_valid, Dmin_valid)
    # For invalid regions, assign dummy TOF that will be interpolated as 0 later
    tof_valid = torch.where(valid, tof, 1)
    tof = torch.where(
        valid, tof_valid, -10 * torch.ones_like(tof)
    )
    return tof


