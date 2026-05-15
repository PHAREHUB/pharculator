"""Domain geometry and shell-volume sampling.

User convention: -x points toward the Sun.
GSE convention (used internally by the MP/BS models): +x points toward the Sun.
The mapping is simply x_gse = -x_user.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .constants import Re_km, SolarWind, NOMINAL_SW
from .models import shue_mp, jelinek_bs

import numpy as _np
# Angular cap for the shell membership. Beyond this angle the Jelinek BS
# flares unphysically and the model is not a useful sheath description.
SHELL_THETA_MAX = _np.deg2rad(130.0)


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
    in_shell: np.ndarray   # bool: L0 region
    in_L1: np.ndarray      # bool: boundary band (around MP + BS), ~10% of L0 volume
    in_L2: np.ndarray      # bool: narrower band inside L1, ~50% of L1 volume
    dV_Re3: float          # volume of one sampling cell, Re^3
    volume_Re3: float      # total L0 shell volume in Re^3
    volume_L1_Re3: float
    volume_L2_Re3: float
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
    l1_fraction: float = 0.10,
    l2_fraction_of_l1: float = 0.50,
) -> ShellSample:
    """Sample the magnetosheath shell and its refined sub-regions.

    L0: shell { (x,y,z) : r_mp(theta) - pad <= r <= r_bs(theta) + pad }
        intersected with the user-domain box.

    L1: thin band inside L0 around the BS and MP. Defined as the subset of
        L0 cells whose distance to either boundary surface is the smallest,
        chosen so that V(L1) ~ l1_fraction * V(L0).

    L2: even thinner band inside L1, the L1 subset closest to either boundary,
        with V(L2) ~ l2_fraction_of_l1 * V(L1).

    Theta is measured from +x_GSE (i.e. from -x_user). Models are evaluated
    with their natural extension at high theta; deep-tail points fall outside
    the shell automatically because r_mp blows up faster than the actual r.
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

    in_shell = (
        (theta <= SHELL_THETA_MAX)
        & (r >= (r_mp - pad_Re))
        & (r <= (r_bs + pad_Re))
    )

    dV = sample_dx_Re ** 3
    V0 = float(in_shell.sum()) * dV

    # Distance to nearer boundary surface (BS or MP), only meaningful on L0.
    d_bs = np.abs(r - r_bs)
    d_mp = np.abs(r - r_mp)
    d_boundary = np.minimum(d_bs, d_mp)
    # Outside L0, push distance to +inf so those cells never enter L1/L2.
    d_boundary_masked = np.where(in_shell, d_boundary, np.inf)

    # Pick the L1 cells: those with smallest d_boundary in L0, cumulative
    # volume capped at l1_fraction * V0.
    flat = d_boundary_masked.ravel()
    order = np.argsort(flat, kind="stable")
    cum = (np.arange(1, flat.size + 1)) * dV
    target_L1 = l1_fraction * V0
    target_L2 = l2_fraction_of_l1 * target_L1
    n_L1 = int(np.searchsorted(cum, target_L1, side="right"))
    n_L2 = int(np.searchsorted(cum, target_L2, side="right"))
    # Don't include points beyond L0 (their distance is inf)
    finite_mask = np.isfinite(flat[order])
    valid_count = int(finite_mask.sum())
    n_L1 = min(n_L1, valid_count)
    n_L2 = min(n_L2, valid_count)

    L1_flat = np.zeros(flat.size, dtype=bool)
    L1_flat[order[:n_L1]] = True
    L2_flat = np.zeros(flat.size, dtype=bool)
    L2_flat[order[:n_L2]] = True

    in_L1 = L1_flat.reshape(in_shell.shape)
    in_L2 = L2_flat.reshape(in_shell.shape)
    V1 = float(in_L1.sum()) * dV
    V2 = float(in_L2.sum()) * dV

    return ShellSample(
        in_shell=in_shell,
        in_L1=in_L1,
        in_L2=in_L2,
        dV_Re3=dV,
        volume_Re3=V0,
        volume_L1_Re3=V1,
        volume_L2_Re3=V2,
        x_user=X,
        y=Y,
        z=Z,
    )
