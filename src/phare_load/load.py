"""Memory and compute load estimates, driven by a Config."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config, LevelSpec
from .geometry import sample_regions, RegionSample


@dataclass
class LevelLoad:
    name: str
    kind: str            # "mhd" or "pic"
    dx_km: float
    volume_Re3: float
    n_cells: float
    n_particles: float
    ram_bytes: float
    sec_per_step: float
    steps_per_finest: float

    @property
    def is_pic(self) -> bool:
        return self.kind == "pic"

    @property
    def sec_per_finest_step(self) -> float:
        return self.sec_per_step * self.steps_per_finest


@dataclass
class LoadReport:
    levels: list[LevelLoad]
    amr_total: LevelLoad
    uniform_reference: LevelLoad
    n_steps_target: int
    cfg: Config

    def by_name(self, name: str) -> LevelLoad:
        for L in self.levels:
            if L.name == name:
                return L
        raise KeyError(name)

    @property
    def pic_levels(self) -> list[LevelLoad]:
        return [L for L in self.levels if L.is_pic]


def _level_from_spec(spec: LevelSpec, volume_Re3: float, cfg: Config,
                     steps_per_finest: float) -> LevelLoad:
    dx_km = spec.resolve_dx_km(cfg.delta_i_km)
    volume_km3 = volume_Re3 * (cfg.re_km ** 3)
    n_cells = volume_km3 / (dx_km ** 3)
    if spec.kind == "pic":
        n_particles = cfg.ppc * n_cells
        ram = n_particles * cfg.bytes_per_particle
        t = n_particles * cfg.sec_per_particle_per_step
    else:
        n_particles = 0.0
        ram = 0.0
        t = 0.0
    return LevelLoad(
        name=spec.name, kind=spec.kind, dx_km=dx_km,
        volume_Re3=volume_Re3, n_cells=n_cells,
        n_particles=n_particles, ram_bytes=ram, sec_per_step=t,
        steps_per_finest=steps_per_finest,
    )


def build_report(cfg: Config) -> tuple[LoadReport, RegionSample]:
    sample = sample_regions(cfg)

    levels: list[LevelLoad] = []
    for spec in cfg.levels:
        if spec.kind == "pic":
            v = sample.volumes_Re3[spec.name]
        else:
            v = cfg.domain.volume_Re3()
        levels.append(_level_from_spec(
            spec, v, cfg, cfg.steps_per_finest(spec.name)))

    # Reference uniform PIC over the whole box, sharing dt with the finest level.
    ref_dx_km = cfg.resolve_reference_dx_km()
    ref_spec = LevelSpec(
        name=f"uniform_{ref_dx_km:.0f}km",
        kind="pic",
        dx_km=ref_dx_km,
        region="full",
    )
    ref = _level_from_spec(ref_spec, cfg.domain.volume_Re3(), cfg,
                           steps_per_finest=1.0)

    pic = [L for L in levels if L.is_pic]
    total = LevelLoad(
        name="AMR PIC total",
        kind="pic",
        dx_km=float("nan"),
        volume_Re3=sum(L.volume_Re3 for L in pic),
        n_cells=sum(L.n_cells for L in pic),
        n_particles=sum(L.n_particles for L in pic),
        ram_bytes=sum(L.ram_bytes for L in pic),
        sec_per_step=sum(L.sec_per_finest_step for L in pic),
        steps_per_finest=1.0,
    )

    return LoadReport(
        levels=levels,
        amr_total=total,
        uniform_reference=ref,
        n_steps_target=cfg.n_steps_target,
        cfg=cfg,
    ), sample
