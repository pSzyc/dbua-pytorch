from dataclasses import dataclass, field
from pathlib import Path

import torch


def _default_device() -> torch.device:
    """Select the compute device. CUDA is the primary target; fall back to MPS
    (mac dev boxes) or CPU."""
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


# Ground-truth global sound speed [m/s] per dataset. Constant phantoms carry
# their nominal speed; heterogeneous phantoms have no single value, recorded as 0.
_CTRUE_DEFAULT = {
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
    "checker8": 0,
}


@dataclass
class DBUAConfig:
    """Configuration for the DBUA sound-speed estimation pipeline.

    Collects every tunable constant in one place, each with a default value and
    a description. Instantiate with no arguments for the defaults, or override
    individual fields per run.
    """

    # --- Optimization --------------------------------------------------------
    n_iters: int = 301                  # Number of gradient-descent iterations
    learning_rate: float = 10           # Adam (AMSGrad) learning rate
    assumed_c: float = 1540             # Assumed global sound speed for init [m/s]

    # --- B-mode image extent [m] ---------------------------------------------
    bmode_x_min: float = -12e-3         # Lateral min
    bmode_x_max: float = 12e-3          # Lateral max
    bmode_z_min: float = 0e-3           # Axial (depth) min
    bmode_z_max: float = 40e-3          # Axial (depth) max

    # --- Sound speed grid extent [m] and node counts -------------------------
    sound_speed_x_min: float = -12e-3   # Lateral min
    sound_speed_x_max: float = 12e-3    # Lateral max
    sound_speed_z_min: float = 0e-3     # Axial (depth) min
    sound_speed_z_max: float = 40e-3    # Axial (depth) max
    sound_speed_nxc: int = 19           # Number of lateral sound speed nodes
    sound_speed_nzc: int = 31           # Number of axial sound speed nodes

    # --- Phase estimate kernel size [samples] --------------------------------
    nxk: int = 5                        # Lateral kernel size
    nzk: int = 5                        # Axial kernel size

    # --- Phase estimate patch grid -------------------------------------------
    nxp: int = 17                       # Lateral patch count
    nzp: int = 17                       # Axial patch count
    phase_error_x_min: float = -20e-3   # Patch-center lateral min [m]
    phase_error_x_max: float = 20e-3    # Patch-center lateral max [m]
    phase_error_z_min: float = 4e-3     # Patch-center axial min [m]
    phase_error_z_max: float = 44e-3    # Patch-center axial max [m]

    # --- Run selection -------------------------------------------------------
    # loss_name options: "pe" phase error, "sb" speckle brightness,
    #                    "cf" coherence factor, "lc" lag-one coherence
    loss_name: str = "pe"               # Loss used to drive the optimization
    # sample options: constant phantoms ("1420".."1570") or heterogeneous
    #                phantoms ("inclusion", "inclusion_layer", "four_layer",
    #                "two_layer", "checker2", "checker8")
    sample: str = "checker2"            # Dataset to reconstruct

    # --- Data & compute ------------------------------------------------------
    data_dir: Path = Path("./data")     # Base dir of refocused plane-wave datasets
    ctrue: dict = field(default_factory=lambda: dict(_CTRUE_DEFAULT))  # Ground-truth c per dataset
    device: torch.device = field(default_factory=_default_device)     # Compute device
    real_dtype: torch.dtype = torch.float32  # Real dtype (IQ data uses complex64)
