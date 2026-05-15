"""Magnetopause (Shue 1998) and bow shock (Jelinek 2012) models.

Both are evaluated in standard GSE: +x toward the Sun, theta is the angle
from the +x axis. r is the geocentric distance in Earth radii.
"""

from __future__ import annotations

import numpy as np

from .constants import SolarWind, NOMINAL_SW

# Cap theta for model validity. Beyond this angle, the Shue and Jelinek
# fits extrapolate badly into the deep tail; we evaluate models at the cap
# so plotted curves remain finite, but the shell-volume routine treats
# points with theta > THETA_MAX as outside the kinetic region.
THETA_MAX = np.deg2rad(130.0)


def pdyn_nPa(sw: SolarWind = NOMINAL_SW) -> float:
    return sw.Pdyn_nPa


def shue_mp(theta, sw: SolarWind = NOMINAL_SW):
    """Shue et al. 1998 magnetopause distance in Earth radii.

    theta : angle from +x_GSE (Sun-Earth line) in radians. Scalar or array.
    Values above THETA_MAX are evaluated at THETA_MAX.
    """
    theta = np.asarray(theta, dtype=float)
    theta = np.minimum(theta, THETA_MAX)
    Pd = sw.Pdyn_nPa
    Bz = sw.Bz_nT
    r0 = (10.22 + 1.29 * np.tanh(0.184 * (Bz + 8.14))) * Pd ** (-1.0 / 6.6)
    alpha = (0.58 - 0.007 * Bz) * (1.0 + 0.024 * np.log(Pd))
    return r0 * (2.0 / (1.0 + np.cos(theta))) ** alpha


def jelinek_bs(theta, sw: SolarWind = NOMINAL_SW):
    """Jelinek et al. 2012 bow shock distance in Earth radii (paraboloid form).

    theta : angle from +x_GSE in radians.
    Values above THETA_MAX are evaluated at THETA_MAX.
    """
    theta = np.asarray(theta, dtype=float)
    theta = np.minimum(theta, THETA_MAX)
    Pd = sw.Pdyn_nPa
    R = 15.02 * Pd ** (-1.0 / 6.55)
    lam = 1.17
    return R * (2.0 / (1.0 + np.cos(theta))) ** lam


def subsolar_mp(sw: SolarWind = NOMINAL_SW) -> float:
    return float(shue_mp(0.0, sw))


def subsolar_bs(sw: SolarWind = NOMINAL_SW) -> float:
    return float(jelinek_bs(0.0, sw))
