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
    # Specify exactly one of dx_km or dx_di (in units of delta_i).
    dx_km: float | None = None
    dx_di: float | None = None
    region: RegionKind = "full"
    pad_re: float = 0.0        # for region == "shell"
    band_re: float = 0.0       # for region == "band"
    # For region == "band": which surface(s) the band hugs. Each entry must
    # be "mp" or "bs". Defaults to both.
    boundaries: list[str] = field(default_factory=lambda: ["mp", "bs"])

    def resolve_dx_km(self, delta_i_km: float) -> float:
        if (self.dx_km is None) == (self.dx_di is None):
            raise ValueError(
                f"Level {self.name}: specify exactly one of dx_km or dx_di.")
        return self.dx_km if self.dx_km is not None else self.dx_di * delta_i_km


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
    target_hours: float = 1.0
    # dt is expressed physically: anchor at the first PIC level ("L1"), then
    # scale as dx**2 to the finest level via dt_ratio_per_level (= 4 for the
    # standard refinement ratio 2 in space).
    omega_ci_inverse_s: float = 1.0        # physical duration of 1/Ω_ci, in s
    dt_L1_omega_ci: float = 0.05           # L1 dt, in units of 1/Ω_ci
    dt_finest_s: float | None = None       # optional override (absolute)
    dt_ratio_per_level: float = 4.0

    # geometry / sampling
    dayside_only: bool = True
    sample_dx_re: float = 0.5

    # Knob to "cheat" the magnetosphere size: relative to Earth's actual dipole
    # moment. Both the Shue magnetopause and the Jelinek bow shock are scaled
    # by dipole_strength**(1/3), which is the correct scaling at the subsolar
    # point (pressure balance gives r_mp proportional to M_E**(1/3)).
    dipole_strength: float = 1.0

    # reference uniform run: specify exactly one of these.
    reference_dx_km: float | None = None
    reference_dx_di: float | None = None

    def resolve_reference_dx_km(self) -> float:
        if (self.reference_dx_km is None) == (self.reference_dx_di is None):
            raise ValueError(
                "Specify exactly one of reference_dx_km or reference_dx_di.")
        return (self.reference_dx_km if self.reference_dx_km is not None
                else self.reference_dx_di * self.delta_i_km)

    domain: Domain = field(default_factory=Domain)
    solar_wind: SolarWind = field(default_factory=SolarWind)
    levels: list[LevelSpec] = field(default_factory=list)

    def resolve_dt_finest_s(self) -> float:
        """Resolve the finest-level dt in seconds.

        If `dt_finest_s` is set, use it verbatim. Otherwise derive from the
        L1 dt anchor: dt_finest = dt_L1 · (dx_finest / dx_L1)^2.
        """
        if self.dt_finest_s is not None:
            return self.dt_finest_s
        pic = self.pic_levels
        if not pic:
            raise ValueError(
                "Cannot derive dt_finest from L1 anchor: no PIC level "
                "defined. Set dt_finest_s explicitly.")
        dx_L1 = pic[0].resolve_dx_km(self.delta_i_km)
        dx_finest = pic[-1].resolve_dx_km(self.delta_i_km)
        dt_L1_s = self.dt_L1_omega_ci * self.omega_ci_inverse_s
        return dt_L1_s * (dx_finest / dx_L1) ** 2

    @property
    def n_steps_target(self) -> int:
        return int(round(self.target_hours * 3600.0 / self.resolve_dt_finest_s()))

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
        "delta_i_km", "re_km", "target_hours",
        "omega_ci_inverse_s", "dt_L1_omega_ci", "dt_finest_s",
        "dt_ratio_per_level", "dayside_only", "sample_dx_re",
        "reference_dx_km", "reference_dx_di", "dipole_strength",
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
    cfg.resolve_reference_dx_km()  # raises if both/neither given
    for L in cfg.levels:
        if L.kind not in ("mhd", "pic"):
            raise ValueError(f"Level {L.name}: kind must be 'mhd' or 'pic'.")
        if L.region not in ("full", "shell", "band"):
            raise ValueError(f"Level {L.name}: region must be full/shell/band.")
        if L.region == "shell" and L.pad_re <= 0:
            raise ValueError(f"Level {L.name}: region='shell' needs pad_re > 0.")
        if L.region == "band":
            if L.band_re <= 0:
                raise ValueError(f"Level {L.name}: region='band' needs band_re > 0.")
            if not L.boundaries:
                raise ValueError(f"Level {L.name}: boundaries must not be empty.")
            for b in L.boundaries:
                if b not in ("mp", "bs"):
                    raise ValueError(
                        f"Level {L.name}: boundaries entries must be 'mp' or 'bs',"
                        f" got {b!r}.")
        L.resolve_dx_km(cfg.delta_i_km)  # raises if both/neither given
    # dx must be non-increasing as we descend the level list (coarsest first)
    for prev, nxt in zip(cfg.levels, cfg.levels[1:]):
        if nxt.resolve_dx_km(cfg.delta_i_km) > prev.resolve_dx_km(cfg.delta_i_km):
            raise ValueError(
                f"Levels must be listed coarsest first: {nxt.name} is finer "
                f"than {prev.name}.")
