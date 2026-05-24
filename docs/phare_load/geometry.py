"""Domain geometry and per-level region sampling, driven by a Config.

Coordinates are standard GSE: +x toward the Sun, -x toward the tail.
The Shue and Jelinek models use the same convention, so no remapping is
needed between user space and the models.
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
    x: np.ndarray                  # GSE x (positive toward Sun)
    y: np.ndarray
    z: np.ndarray


def sample_regions(cfg: Config) -> RegionSample:
    d = cfg.domain
    dx = cfg.sample_dx_re
    xs = np.arange(d.x_min, d.x_max + dx, dx)
    ys = np.arange(d.y_min, d.y_max + dx, dx)
    zs = np.arange(d.z_min, d.z_max + dx, dx)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    r = np.sqrt(X * X + Y * Y + Z * Z)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_t = np.where(r > 0, X / r, 1.0)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    theta = np.arccos(cos_t)
    r_mp = shue_mp(theta, cfg.solar_wind, dipole_strength=cfg.dipole_strength)
    r_bs = jelinek_bs(theta, cfg.solar_wind, dipole_strength=cfg.dipole_strength)
    # φ = azimuth around +X_GSE: 0° = +Z (north), 90° = +Y (dusk),
    # 180° = -Z (south), 270° = -Y (dawn). Wrapped to [0, 360).
    phi = np.mod(np.rad2deg(np.arctan2(Y, Z)), 360.0)
    theta_deg = np.rad2deg(theta)

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
            m = np.zeros_like(theta, dtype=bool)
            if "mp" in spec.boundaries:
                m = m | (np.abs(r - r_mp) <= spec.band_re)
            if "bs" in spec.boundaries:
                m = m | (np.abs(r - r_bs) <= spec.band_re)
            patches = spec.resolved_patches()
            if patches:
                patch_m = np.zeros_like(theta, dtype=bool)
                for p in patches:
                    pm = np.ones_like(theta, dtype=bool)
                    if p.theta_min_deg is not None:
                        pm = pm & (theta_deg >= p.theta_min_deg)
                    if p.theta_max_deg is not None:
                        pm = pm & (theta_deg <= p.theta_max_deg)
                    if p.phi_ranges_deg:
                        phi_m = np.zeros_like(theta, dtype=bool)
                        for a, b in p.phi_ranges_deg:
                            phi_m = phi_m | ((phi >= a) & (phi <= b))
                        pm = pm & phi_m
                    patch_m = patch_m | pm
                m = m & patch_m
        else:  # pragma: no cover
            raise AssertionError(spec.region)
        if cfg.dayside_only:
            m = m & (X >= 0)
        if prev_pic_mask is not None:
            m = m & prev_pic_mask
        masks[spec.name] = m
        volumes[spec.name] = float(m.sum()) * dV
        prev_pic_mask = m

    return RegionSample(
        masks=masks, volumes_Re3=volumes,
        dV_Re3=dV, x=X, y=Y, z=Z,
    )
