"""Domain geometry and sampling of the PIC levels (L1, L2).

User convention: -x points toward the Sun.
GSE convention (used internally by the MP/BS models): +x points toward the Sun.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .constants import Re_km, SolarWind, NOMINAL_SW
from .models import shue_mp, jelinek_bs


# Domain extent in Earth radii, in the *user* convention (-x sunward).
DOMAIN_USER = {
    "x_min": -30.0,
    "x_max": 150.0,
    "y_min": -30.0,
    "y_max": 30.0,
    "z_min": -30.0,
    "z_max": 30.0,
}


@dataclass(frozen=True)
class ShellSample:
    in_L1: np.ndarray         # bool: sheath shell with pad buffers (PIC)
    in_L2: np.ndarray         # bool: thin bands around MP and BS (PIC)
    dV_Re3: float             # volume of one sampling cell (Re^3)
    volume_L1_Re3: float
    volume_L2_Re3: float
    x_user: np.ndarray
    y: np.ndarray
    z: np.ndarray


def domain_extent_km():
    Lx = (DOMAIN_USER["x_max"] - DOMAIN_USER["x_min"]) * Re_km
    Ly = (DOMAIN_USER["y_max"] - DOMAIN_USER["y_min"]) * Re_km
    Lz = (DOMAIN_USER["z_max"] - DOMAIN_USER["z_min"]) * Re_km
    return Lx, Ly, Lz


def domain_volume_Re3() -> float:
    d = DOMAIN_USER
    return (
        (d["x_max"] - d["x_min"])
        * (d["y_max"] - d["y_min"])
        * (d["z_max"] - d["z_min"])
    )


def sample_shell(
    sw: SolarWind = NOMINAL_SW,
    l1_pad_Re: float = 3.0,
    l2_band_Re: float = 1.5,
    sample_dx_Re: float = 0.5,
) -> ShellSample:
    """Sample the PIC levels on a regular probe grid.

    L1 is the sheath shell with a `l1_pad_Re`-Re buffer outside the BS and
    inside the MP:
        L1 = { r_mp(theta) - l1_pad <= r <= r_bs(theta) + l1_pad }

    L2 is the union of two thin bands of half-thickness `l2_band_Re`
    around the MP and BS surfaces, intersected with L1:
        L2 = L1  and  ( |r - r_mp| <= l2_band  OR  |r - r_bs| <= l2_band )

    Where the BS surface lies outside the user-domain box, the box itself
    truncates the levels naturally.
    """
    d = DOMAIN_USER
    xs = np.arange(d["x_min"], d["x_max"] + sample_dx_Re, sample_dx_Re)
    ys = np.arange(d["y_min"], d["y_max"] + sample_dx_Re, sample_dx_Re)
    zs = np.arange(d["z_min"], d["z_max"] + sample_dx_Re, sample_dx_Re)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    Xg = -X
    r = np.sqrt(Xg * Xg + Y * Y + Z * Z)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_t = np.where(r > 0, Xg / r, 1.0)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    theta = np.arccos(cos_t)

    r_mp = shue_mp(theta, sw)
    r_bs = jelinek_bs(theta, sw)

    in_L1 = (r >= (r_mp - l1_pad_Re)) & (r <= (r_bs + l1_pad_Re))

    near_mp = np.abs(r - r_mp) <= l2_band_Re
    near_bs = np.abs(r - r_bs) <= l2_band_Re
    in_L2 = in_L1 & (near_mp | near_bs)

    dV = sample_dx_Re ** 3
    V1 = float(in_L1.sum()) * dV
    V2 = float(in_L2.sum()) * dV

    return ShellSample(
        in_L1=in_L1,
        in_L2=in_L2,
        dV_Re3=dV,
        volume_L1_Re3=V1,
        volume_L2_Re3=V2,
        x_user=X,
        y=Y,
        z=Z,
    )
