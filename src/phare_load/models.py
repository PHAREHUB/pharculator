"""Magnetopause (Shue 1998) and bow shock (Jelinek 2012) models.

Both are evaluated in standard GSE: +x toward the Sun, theta is the angle
from the +x axis. r is the geocentric distance in Earth radii.
"""

from __future__ import annotations

import numpy as np

from .constants import SolarWind, NOMINAL_SW

# THETA_MAX delimits where the Shue / Jelinek fits are still considered
# physically meaningful. The model functions evaluate freely (the formula
# is just a mathematical extension at higher theta); THETA_MAX is exported
# so that callers (e.g. the shell-volume routine) can decide where to stop.
THETA_MAX = np.deg2rad(160.0)

# Numerical safety: keep (1+cos theta) above this to avoid divergence at pi.
_COS_EPS = 1e-3


def pdyn_nPa(sw: SolarWind = NOMINAL_SW) -> float:
    return sw.Pdyn_nPa


def _flare(theta):
    return 2.0 / np.maximum(1.0 + np.cos(np.asarray(theta, dtype=float)), _COS_EPS)


def shue_mp(theta, sw: SolarWind = NOMINAL_SW):
    """Shue et al. 1998 magnetopause distance in Earth radii.

    theta : angle from +x_GSE (Sun-Earth line) in radians. Scalar or array.
    """
    Pd = sw.Pdyn_nPa
    Bz = sw.Bz_nT
    r0 = (10.22 + 1.29 * np.tanh(0.184 * (Bz + 8.14))) * Pd ** (-1.0 / 6.6)
    alpha = (0.58 - 0.007 * Bz) * (1.0 + 0.024 * np.log(Pd))
    return r0 * _flare(theta) ** alpha


def jelinek_bs(theta, sw: SolarWind = NOMINAL_SW):
    """Jelinek et al. 2012 bow shock distance in Earth radii (paraboloid form).

    theta : angle from +x_GSE in radians.
    """
    Pd = sw.Pdyn_nPa
    R = 15.02 * Pd ** (-1.0 / 6.55)
    lam = 1.17
    return R * _flare(theta) ** lam


def subsolar_mp(sw: SolarWind = NOMINAL_SW) -> float:
    return float(shue_mp(0.0, sw))


def subsolar_bs(sw: SolarWind = NOMINAL_SW) -> float:
    return float(jelinek_bs(0.0, sw))
