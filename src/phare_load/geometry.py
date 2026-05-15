"""Domain geometry and shell-volume sampling.

User convention: -x points toward the Sun.
GSE convention (used internally by the MP/BS models): +x points toward the Sun.
The mapping is simply x_gse = -x_user.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .constants import Re_km, SolarWind, NOMINAL_SW
from .models import shue_mp, jelinek_bs, THETA_MAX


# Domain extent in Earth radii, in the *user* convention (-x sunward).
DOMAIN_USER = {
    "x_min": -30.0,   # sunward boundary
    "x_max": 150.0,   # tail boundary
    "y_min": -30.0,
    "y_max": 30.0,
    "z_min": -30.0,
    "z_max": 30.0,
}


@dataclass(frozen=True)
class ShellSample:
    in_shell: np.ndarray   # bool array on sampling grid
    dV_Re3: float          # volume of one sampling cell, Re^3
    volume_Re3: float      # total shell volume in Re^3
    x_user: np.ndarray     # 3D x coordinate in user frame (Re)
    y: np.ndarray
    z: np.ndarray


def domain_extent_km():
    Lx = (DOMAIN_USER["x_max"] - DOMAIN_USER["x_min"]) * Re_km
    Ly = (DOMAIN_USER["y_max"] - DOMAIN_USER["y_min"]) * Re_km
    Lz = (DOMAIN_USER["z_max"] - DOMAIN_USER["z_min"]) * Re_km
    return Lx, Ly, Lz


def sample_shell(
    sw: SolarWind = NOMINAL_SW,
    pad_Re: float = 2.0,
    sample_dx_Re: float = 0.5,
) -> ShellSample:
    """Sample the magnetosheath-plus-buffer shell on a regular grid.

    The shell is { (x,y,z) : r_mp(theta) - pad <= r <= r_bs(theta) + pad }
    inside the user-domain box. theta is computed from the +x_GSE axis,
    i.e. from -x_user.

    Returns boolean mask and cell volume so the caller can integrate.
    """
    d = DOMAIN_USER
    xs = np.arange(d["x_min"], d["x_max"] + sample_dx_Re, sample_dx_Re)
    ys = np.arange(d["y_min"], d["y_max"] + sample_dx_Re, sample_dx_Re)
    zs = np.arange(d["z_min"], d["z_max"] + sample_dx_Re, sample_dx_Re)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    # Convert to GSE: x_gse = -x_user
    Xg = -X
    r = np.sqrt(Xg * Xg + Y * Y + Z * Z)
    # Avoid div-by-zero at origin; r=0 maps to theta=0 (subsolar).
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_t = np.where(r > 0, Xg / r, 1.0)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    theta = np.arccos(cos_t)

    r_mp = shue_mp(theta, sw)
    r_bs = jelinek_bs(theta, sw)

    # Restrict the kinetic shell to the angular region where the MP/BS
    # fits are reliable (dayside + flanks). The deep tail is excluded.
    in_shell = (
        (theta <= THETA_MAX)
        & (r >= (r_mp - pad_Re))
        & (r <= (r_bs + pad_Re))
    )

    dV = sample_dx_Re ** 3
    volume = float(in_shell.sum()) * dV

    return ShellSample(
        in_shell=in_shell,
        dV_Re3=dV,
        volume_Re3=volume,
        x_user=X,
        y=Y,
        z=Z,
    )
