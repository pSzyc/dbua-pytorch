from typing import Callable

import torch

from PyTorch.conf import DBUAConfig
from PyTorch.loss_functions import (
    lag_one_coherence,
    coherence_factor,
    phase_error,
    total_variation,
    speckle_brightness,
)
from PyTorch.paths import time_of_flight


class ExecutionManager:
    """Builds the fixed geometry (image grid, sound-speed grid, phase-error
    patches) for one dataset and evaluates the sound-speed loss functions on it.

    All grids are constructed once in ``__init__``; the loss methods then take a
    candidate sound-speed map ``c`` of shape ``[nxc, nzc]`` and return a scalar.
    """

    iqdata: torch.Tensor
    t0: torch.Tensor
    fs: float
    fd: float

    def __init__(
        self,
        iqdata: torch.Tensor,
        t0: torch.Tensor,
        fs: float,
        fd: float,
        elpos: torch.Tensor,
        config: DBUAConfig,
    ) -> None:
        self.iqdata = iqdata
        self.t0 = t0
        self.fs = fs
        self.fd = fd
        self.config = config

        # Wavelength (λ) at the assumed sound speed; sets the grid/kernel spacing.
        self.wl0 = config.assumed_c / fd

        # B-mode image dimensions
        xi, zi = self._img_dim()
        self.nxi, self.nzi = xi.numel(), zi.numel()
        self.xi, self.zi = torch.meshgrid(xi, zi, indexing="ij")

        # Sound speed grid dimensions
        self.xc, self.zc = self._c_dim()
        self.dxc = self.xc[1] - self.xc[0]
        self.dzc = self.zc[1] - self.zc[0]

        # Explicit broadcasting. Dimensions will be [elements, pixels, patches]
        xe, _, ze = elpos
        self.xe = torch.reshape(xe, (-1, 1, 1))
        self.ze = torch.reshape(ze, (-1, 1, 1))

        # Kernels to use for loss calculations (2λ x 2λ patches)
        xk, zk = self._kernels()

        # Kernel patch centers, distributed throughout the field of view
        xpc, zpc = self._kernel_patch()

        xp = torch.reshape(xpc, (1, -1, 1)) + torch.reshape(xk, (1, 1, -1))
        zp = torch.reshape(zpc, (1, -1, 1)) + torch.reshape(zk, (1, 1, -1))
        self.xp = xp + 0 * zp  # Manual broadcasting
        self.zp = zp + 0 * xp  # Manual broadcasting

    # --- Time-of-flight ------------------------------------------------------

    def _tof_image(self, c: torch.Tensor) -> torch.Tensor:
        """Time-of-flight from every element to every B-mode image pixel."""
        return time_of_flight(self.xe, self.ze, self.xi, self.zi, self.xc, self.zc, c, fnum=0.5, npts=64)

    def _tof_patch(self, c: torch.Tensor) -> torch.Tensor:
        """Time-of-flight from every element to every phase-error patch pixel."""
        return time_of_flight(self.xe, self.ze, self.xp, self.zp, self.xc, self.zc, c, fnum=0.5, npts=64)

    # --- Loss terms ----------------------------------------------------------

    def sb_loss(self, c: torch.Tensor) -> torch.Tensor:
        """Speckle-brightness loss (lower is better focused)."""
        return 1 - self._loss_wrapper(speckle_brightness, c)

    def lc_loss(self, c: torch.Tensor) -> torch.Tensor:
        """Lag-one-coherence loss."""
        return 1 - torch.mean(self._loss_wrapper(lag_one_coherence, c))

    def cf_loss(self, c: torch.Tensor) -> torch.Tensor:
        """Coherence-factor loss."""
        return 1 - torch.mean(self._loss_wrapper(coherence_factor, c))

    def pe_loss(self, c: torch.Tensor) -> torch.Tensor:
        """Phase-error loss: mean log1p of squared inter-aperture phase error."""
        t = self._tof_patch(c)
        dphi = phase_error(self.iqdata, t - self.t0, t, self.fs, self.fd)
        # "double where" trick: mask invalid (zero) entries to NaN without
        # letting their gradients leak through either branch.
        valid = dphi != 0
        nan = torch.full_like(dphi, float("nan"))
        dphi = torch.where(valid, torch.where(valid, dphi, nan), nan)
        return torch.nanmean(torch.log1p(torch.square(100 * dphi)))

    def _loss_wrapper(self, func: Callable[..., torch.Tensor], c: torch.Tensor) -> torch.Tensor:
        """Evaluate a beamforming loss ``func`` on the patch geometry for ``c``."""
        t = self._tof_patch(c)
        return func(self.iqdata, t - self.t0, t, self.fs, self.fd)

    def loss(self, c: torch.Tensor, loss_name: str) -> torch.Tensor:
        """Total loss: the selected data term plus a total-variation penalty."""
        match loss_name:
            case "sb":  # Speckle brightness
                data_term = self.sb_loss(c)
            case "lc":  # Lag one coherence
                data_term = self.lc_loss(c)
            case "cf":  # Coherence factor
                data_term = self.cf_loss(c)
            case "pe":  # Phase error
                data_term = self.pe_loss(c)
            case _:
                raise NotImplementedError(f"Unknown loss: {loss_name!r}")
        return data_term + self._tv(c) * 1e2

    def _tv(self, c: torch.Tensor) -> torch.Tensor:
        """Total-variation regularizer, scaled by grid cell area (dxc·dzc) so
        its strength is independent of the sound-speed grid resolution."""
        return total_variation(c) * self.dxc * self.dzc

    # --- Grid construction ---------------------------------------------------

    def _img_dim(self) -> tuple[torch.Tensor, torch.Tensor]:
        """B-mode image axes (xi, zi) sampled at λ/3 spacing."""
        config = self.config
        step = self.wl0 / 3
        return (
            torch.arange(config.bmode_x_min, config.bmode_x_max, step, dtype=config.real_dtype, device=config.device),
            torch.arange(config.bmode_z_min, config.bmode_z_max, step, dtype=config.real_dtype, device=config.device),
        )

    def _c_dim(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sound-speed grid axes (xc, zc)."""
        config = self.config
        return (
            torch.linspace(config.sound_speed_x_min, config.sound_speed_x_max, config.sound_speed_nxc, dtype=config.real_dtype, device=config.device),
            torch.linspace(config.sound_speed_z_min, config.sound_speed_z_max, config.sound_speed_nzc, dtype=config.real_dtype, device=config.device),
        )

    def _kernel_patch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Meshgrid of phase-error patch centers across the field of view."""
        config = self.config
        return torch.meshgrid(
            torch.linspace(config.phase_error_x_min, config.phase_error_x_max, config.nxp, dtype=config.real_dtype, device=config.device),
            torch.linspace(config.phase_error_z_min, config.phase_error_z_max, config.nzp, dtype=config.real_dtype, device=config.device),
            indexing="ij",
        )

    def _kernels(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Meshgrid of kernel sample offsets (2λ × 2λ patch) about a patch center."""
        config = self.config
        return torch.meshgrid(
            (torch.arange(config.nxk, dtype=config.real_dtype, device=config.device) - (config.nxk - 1) / 2) * self.wl0 / 2,
            (torch.arange(config.nzk, dtype=config.real_dtype, device=config.device) - (config.nzk - 1) / 2) * self.wl0 / 2,
            indexing="ij",
        )

    # --- Accessors -----------------------------------------------------------

    def get_xi_zi(self) -> tuple[torch.Tensor, torch.Tensor]:
        """B-mode image coordinate meshgrids (xi, zi)."""
        return self.xi, self.zi

    def get_xc_zc(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sound-speed grid axes (xc, zc)."""
        return self.xc, self.zc

    def get_nxi_nzi(self) -> tuple[int, int]:
        """B-mode image pixel counts (nxi, nzi)."""
        return self.nxi, self.nzi

    def get_config(self) -> DBUAConfig:
        """The DBUAConfig this manager was built from."""
        return self.config
