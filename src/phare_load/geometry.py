"""Domain geometry and per-level region sampling, driven by a Config.

User convention: -x toward the Sun. Internally we use GSE (+x toward Sun)
for the MP/BS models, with x_gse = -x_user.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import Config, LevelSpec
from .models import shue_mp, jelinek_bs


@dataclass(frozen=True)
class RegionSample:
    masks: dict[str, np.ndarray]   # one boolean mask per PIC level (by name)
    volumes_Re3: dict[str, float]
    dV_Re3: float
    x_user: np.ndarray
    y: np.ndarray
    z: np.ndarray


def sample_regions(cfg: Config) -> RegionSample:
    d = cfg.domain
    dx = cfg.sample_dx_re
    xs = np.arange(d.x_min, d.x_max + dx, dx)
    ys = np.arange(d.y_min, d.y_max + dx, dx)
    zs = np.arange(d.z_min, d.z_max + dx, dx)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    Xg = -X
    r = np.sqrt(Xg * Xg + Y * Y + Z * Z)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_t = np.where(r > 0, Xg / r, 1.0)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    theta = np.arccos(cos_t)
    r_mp = shue_mp(theta, cfg.solar_wind, dipole_strength=cfg.dipole_strength)
    r_bs = jelinek_bs(theta, cfg.solar_wind, dipole_strength=cfg.dipole_strength)

    dV = dx ** 3
    masks: dict[str, np.ndarray] = {}
    volumes: dict[str, float] = {}
    prev_pic_mask: np.ndarray | None = None

    for spec in cfg.levels:
        if spec.kind != "pic":
            continue
        if spec.region == "full":
            m = np.ones_like(theta, dtype=bool)
        elif spec.region == "shell":
            m = (r >= (r_mp - spec.pad_re)) & (r <= (r_bs + spec.pad_re))
        elif spec.region == "band":
            m = (np.abs(r - r_mp) <= spec.band_re) | (np.abs(r - r_bs) <= spec.band_re)
        else:  # pragma: no cover
            raise AssertionError(spec.region)
        if cfg.dayside_only:
            m = m & (Xg >= 0)
        if prev_pic_mask is not None:
            m = m & prev_pic_mask
        masks[spec.name] = m
        volumes[spec.name] = float(m.sum()) * dV
        prev_pic_mask = m

    return RegionSample(
        masks=masks, volumes_Re3=volumes,
        dV_Re3=dV, x_user=X, y=Y, z=Z,
    )
