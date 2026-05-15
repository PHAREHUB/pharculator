"""TOML-driven configuration.

Everything that previously lived as a constant or CLI flag is now read
from a config file. See `config.toml` at the repository root for an
example with the current defaults.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore


RegionKind = Literal["full", "shell", "band"]
LevelKind = Literal["mhd", "pic"]


@dataclass
class LevelSpec:
    name: str
    kind: LevelKind            # "mhd" (no particles) or "pic"
    dx_km: float
    region: RegionKind = "full"
    pad_re: float = 0.0        # for region == "shell"
    band_re: float = 0.0       # for region == "band"


@dataclass
class Domain:
    x_min: float = -30.0
    x_max: float = 150.0
    y_min: float = -30.0
    y_max: float = 30.0
    z_min: float = -30.0
    z_max: float = 30.0

    def volume_Re3(self) -> float:
        return ((self.x_max - self.x_min)
                * (self.y_max - self.y_min)
                * (self.z_max - self.z_min))


@dataclass
class SolarWind:
    n_cm3: float = 5.0
    v_kms: float = 400.0
    bz_nt: float = 0.0

    @property
    def Pdyn_nPa(self) -> float:
        return 1.6726e-6 * self.n_cm3 * self.v_kms ** 2


@dataclass
class Config:
    # physical constants
    delta_i_km: float = 100.0
    re_km: float = 6371.2

    # particle model
    ppc: int = 100
    bytes_per_particle: int = 56
    sec_per_particle_per_step: float = 10e-9

    # time-stepping
    target_hours: float = 3.0
    dt_finest_s: float = 1e-3
    dt_ratio_per_level: float = 4.0

    # geometry / sampling
    dayside_only: bool = True
    sample_dx_re: float = 0.5

    # reference uniform run
    reference_dx_km: float = 20.0

    domain: Domain = field(default_factory=Domain)
    solar_wind: SolarWind = field(default_factory=SolarWind)
    levels: list[LevelSpec] = field(default_factory=list)

    @property
    def n_steps_target(self) -> int:
        return int(round(self.target_hours * 3600.0 / self.dt_finest_s))

    @property
    def pic_levels(self) -> list[LevelSpec]:
        return [L for L in self.levels if L.kind == "pic"]

    def steps_per_finest(self, level_name: str) -> float:
        names = [L.name for L in self.levels]
        idx = names.index(level_name)
        idx_from_finest = len(self.levels) - 1 - idx
        return (1.0 / self.dt_ratio_per_level) ** idx_from_finest


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    cfg = Config()
    for key in (
        "delta_i_km", "re_km", "target_hours", "dt_finest_s",
        "dt_ratio_per_level", "dayside_only", "sample_dx_re",
        "reference_dx_km",
    ):
        if key in raw:
            setattr(cfg, key, raw[key])

    if "particle" in raw:
        p = raw["particle"]
        cfg.ppc = p.get("ppc", cfg.ppc)
        cfg.bytes_per_particle = p.get("bytes_per_particle", cfg.bytes_per_particle)
        cfg.sec_per_particle_per_step = p.get(
            "sec_per_particle_per_step", cfg.sec_per_particle_per_step)

    if "domain" in raw:
        cfg.domain = Domain(**raw["domain"])

    if "solar_wind" in raw:
        cfg.solar_wind = SolarWind(**raw["solar_wind"])

    if "levels" not in raw or not raw["levels"]:
        raise ValueError(
            f"Config {path} has no [[levels]] sections — "
            "at least one level is required.")
    cfg.levels = [LevelSpec(**L) for L in raw["levels"]]

    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if not cfg.levels:
        raise ValueError("At least one level is required.")
    for L in cfg.levels:
        if L.kind not in ("mhd", "pic"):
            raise ValueError(f"Level {L.name}: kind must be 'mhd' or 'pic'.")
        if L.region not in ("full", "shell", "band"):
            raise ValueError(f"Level {L.name}: region must be full/shell/band.")
        if L.region == "shell" and L.pad_re <= 0:
            raise ValueError(f"Level {L.name}: region='shell' needs pad_re > 0.")
        if L.region == "band" and L.band_re <= 0:
            raise ValueError(f"Level {L.name}: region='band' needs band_re > 0.")
    # dx must be non-increasing as we descend the level list (coarsest first)
    for prev, nxt in zip(cfg.levels, cfg.levels[1:]):
        if nxt.dx_km > prev.dx_km:
            raise ValueError(
                f"Levels must be listed coarsest first: {nxt.name}"
                f" (dx={nxt.dx_km}) is finer than {prev.name} (dx={prev.dx_km}).")
