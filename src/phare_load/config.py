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
class Patch:
    """One angular patch on a band surface (region == 'band').

    A patch restricts the radial band to a (θ, φ) rectangle:
    - θ = zenith angle from +X_GSE (0° = subsolar, 180° = anti-solar).
    - φ = azimuth around +X_GSE: 0° = +Z (north), 90° = +Y (dusk),
      180° = -Z (south), 270° = -Y (dawn).

    None means unbounded on that axis. phi_ranges_deg is a list of arcs;
    they do not wrap, so dawn+dusk needs two entries.
    """
    theta_min_deg: float | None = None
    theta_max_deg: float | None = None
    phi_ranges_deg: list[tuple[float, float]] | None = None


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
    # Optional angular patch bounds (region == "band" only).
    # Two equivalent forms — pick the simpler one for your use case:
    #
    # 1. Single-patch shorthand: set any of theta_min_deg, theta_max_deg,
    #    phi_ranges_deg directly on the level. Equivalent to one Patch.
    # 2. Multi-patch union: set `patches = [Patch(...), Patch(...), ...]`.
    #    The band mask is restricted to the *union* of these patches.
    #
    # If `patches` is set, the top-level theta_*/phi_* fields must be unset.
    # If neither is set, the band covers the full surface (current behavior).
    theta_min_deg: float | None = None
    theta_max_deg: float | None = None
    phi_ranges_deg: list[tuple[float, float]] | None = None
    patches: list[Patch] | None = None

    def resolved_patches(self) -> list[Patch]:
        """Normalize the two shorthand forms to a single list of Patches.

        Returns [] when no angular restriction is requested (full band)."""
        if self.patches is not None:
            if (self.theta_min_deg is not None or self.theta_max_deg is not None
                    or self.phi_ranges_deg is not None):
                raise ValueError(
                    f"Level {self.name}: use either top-level theta_*/phi_* "
                    "OR `patches`, not both.")
            return list(self.patches)
        if (self.theta_min_deg is None and self.theta_max_deg is None
                and not self.phi_ranges_deg):
            return []
        return [Patch(
            theta_min_deg=self.theta_min_deg,
            theta_max_deg=self.theta_max_deg,
            phi_ranges_deg=self.phi_ranges_deg,
        )]

    def resolve_dx_km(self, delta_i_km: float) -> float:
        if (self.dx_km is None) == (self.dx_di is None):
            raise ValueError(
                f"Level {self.name}: specify exactly one of dx_km or dx_di.")
        return self.dx_km if self.dx_km is not None else self.dx_di * delta_i_km


@dataclass
class Domain:
    """Domain box in standard GSE: +x toward the Sun, -x toward the tail."""
    x_min: float = -100.0
    x_max: float = 50.0
    y_min: float = -50.0
    y_max: float = 50.0
    z_min: float = -50.0
    z_max: float = 50.0

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
    bytes_per_particle: int = 76
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

    # reference uniform run(s): specify exactly one of these. Either a single
    # float (one reference) or a list of floats (multiple, e.g. a coarse 1 δᵢ
    # baseline plus a fine 0.1 δᵢ target). Listed coarsest-first by convention.
    reference_dx_km: float | list[float] | None = None
    reference_dx_di: float | list[float] | None = None

    def resolve_reference_dx_km(self) -> float:
        """Back-compat single-value accessor — returns the FINEST reference."""
        return self.resolve_reference_dx_km_list()[-1]

    def resolve_reference_dx_km_list(self) -> list[float]:
        if (self.reference_dx_km is None) == (self.reference_dx_di is None):
            raise ValueError(
                "Specify exactly one of reference_dx_km or reference_dx_di.")
        raw = (self.reference_dx_km if self.reference_dx_km is not None
               else self.reference_dx_di)
        scale = 1.0 if self.reference_dx_km is not None else self.delta_i_km
        if isinstance(raw, (int, float)):
            values = [float(raw) * scale]
        else:
            values = [float(v) * scale for v in raw]
        if not values:
            raise ValueError("reference_dx_* list must not be empty.")
        # Sort coarsest -> finest for stable ordering in reports.
        return sorted(values, reverse=True)

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


def _level_from_dict(raw: dict) -> LevelSpec:
    raw = dict(raw)
    phi = raw.get("phi_ranges_deg")
    if phi is not None:
        raw["phi_ranges_deg"] = [tuple(p) for p in phi]
    patches = raw.get("patches")
    if patches is not None:
        normalized = []
        for p in patches:
            p = dict(p)
            pphi = p.get("phi_ranges_deg")
            if pphi is not None:
                p["phi_ranges_deg"] = [tuple(r) for r in pphi]
            normalized.append(Patch(**p))
        raw["patches"] = normalized
    return LevelSpec(**raw)


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
    cfg.levels = [_level_from_dict(L) for L in raw["levels"]]

    _validate(cfg)
    return cfg


def _validate_patch(level_name: str, p: Patch) -> None:
    tmin = p.theta_min_deg if p.theta_min_deg is not None else 0.0
    tmax = p.theta_max_deg if p.theta_max_deg is not None else 180.0
    if not (0.0 <= tmin <= 180.0 and 0.0 <= tmax <= 180.0):
        raise ValueError(
            f"Level {level_name}: patch theta_*_deg must be in [0, 180].")
    if tmin > tmax:
        raise ValueError(
            f"Level {level_name}: patch theta_min_deg must be <= theta_max_deg.")
    if p.phi_ranges_deg is not None:
        for rng in p.phi_ranges_deg:
            if len(rng) != 2:
                raise ValueError(
                    f"Level {level_name}: phi_ranges_deg entries must be "
                    f"(min, max) pairs, got {rng!r}.")
            a, b = float(rng[0]), float(rng[1])
            if not (0.0 <= a <= 360.0 and 0.0 <= b <= 360.0):
                raise ValueError(
                    f"Level {level_name}: phi_ranges_deg values must be in "
                    f"[0, 360], got {rng!r}.")
            if a >= b:
                raise ValueError(
                    f"Level {level_name}: phi range ({a}, {b}) must have "
                    "min < max; wrap-around must be split in two.")


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
            if L.patches is not None and (
                L.theta_min_deg is not None or L.theta_max_deg is not None
                or L.phi_ranges_deg is not None
            ):
                raise ValueError(
                    f"Level {L.name}: use either top-level theta_*/phi_* OR "
                    "`patches`, not both.")
            if L.patches is not None and not L.patches:
                raise ValueError(
                    f"Level {L.name}: `patches` must be non-empty when set.")
            for p in L.resolved_patches():
                _validate_patch(L.name, p)
        L.resolve_dx_km(cfg.delta_i_km)  # raises if both/neither given
    # dx must be non-increasing as we descend the level list (coarsest first)
    for prev, nxt in zip(cfg.levels, cfg.levels[1:]):
        if nxt.resolve_dx_km(cfg.delta_i_km) > prev.resolve_dx_km(cfg.delta_i_km):
            raise ValueError(
                f"Levels must be listed coarsest first: {nxt.name} is finer "
                f"than {prev.name}.")
