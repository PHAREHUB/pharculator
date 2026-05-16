"""Magnetopause (Shue 1998) and bow shock (Jelinek 2012) models.

Both are evaluated in standard GSE: +x toward the Sun, theta is the angle
from the +x axis. r is the geocentric distance in Earth radii.
"""

from __future__ import annotations

import numpy as np

from .config import SolarWind

_NOMINAL_SW = SolarWind()

# Numerical safety: keep (1+cos theta) above this to avoid divergence at pi.
_COS_EPS = 1e-3


def pdyn_nPa(sw: SolarWind | None = None) -> float:
    return (sw or _NOMINAL_SW).Pdyn_nPa


def _flare(theta):
    return 2.0 / np.maximum(1.0 + np.cos(np.asarray(theta, dtype=float)), _COS_EPS)


def shue_mp(theta, sw: SolarWind | None = None,
            dipole_strength: float = 1.0):
    sw = sw or _NOMINAL_SW
    Pd = sw.Pdyn_nPa
    Bz = sw.bz_nt
    r0 = (10.22 + 1.29 * np.tanh(0.184 * (Bz + 8.14))) * Pd ** (-1.0 / 6.6)
    alpha = (0.58 - 0.007 * Bz) * (1.0 + 0.024 * np.log(Pd))
    return dipole_strength ** (1.0 / 3.0) * r0 * _flare(theta) ** alpha


def jelinek_bs(theta, sw: SolarWind | None = None,
               dipole_strength: float = 1.0):
    sw = sw or _NOMINAL_SW
    Pd = sw.Pdyn_nPa
    R = 15.02 * Pd ** (-1.0 / 6.55)
    lam = 1.17
    return dipole_strength ** (1.0 / 3.0) * R * _flare(theta) ** lam


def subsolar_mp(sw: SolarWind | None = None, dipole_strength: float = 1.0) -> float:
    return float(shue_mp(0.0, sw, dipole_strength=dipole_strength))


def subsolar_bs(sw: SolarWind | None = None, dipole_strength: float = 1.0) -> float:
    return float(jelinek_bs(0.0, sw, dipole_strength=dipole_strength))
