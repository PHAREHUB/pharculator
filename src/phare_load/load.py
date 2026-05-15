"""Memory and compute load estimates.

Three-level hierarchy:
- L0 : full domain, MHD only (no particles)
- L1 : sheath shell with 3 Re buffers, PIC at 20 km
- L2 : 1.5 Re bands around the MP and BS surfaces, PIC at 10 km

Reference uniform run: 10 km PIC over the full domain.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import (
    Re_km,
    BYTES_PER_PARTICLE,
    PPC,
    SEC_PER_PARTICLE_PER_STEP,
    DT_SECONDS,
    TARGET_RUN_HOURS,
    L0_DX_KM,
    L1_DX_KM,
    L2_DX_KM,
    REFERENCE_UNIFORM_DX_KM,
    SolarWind,
    NOMINAL_SW,
)
from .geometry import domain_extent_km, domain_volume_Re3, sample_shell, ShellSample


@dataclass
class LevelLoad:
    name: str
    dx_km: float
    is_pic: bool
    volume_Re3: float
    n_cells: float
    n_particles: float
    ram_bytes: float
    sec_per_step: float

    def as_row(self):
        return {
            "level": self.name,
            "dx_km": self.dx_km,
            "pic": self.is_pic,
            "volume_Re3": self.volume_Re3,
            "n_cells": self.n_cells,
            "n_particles": self.n_particles,
            "ram_TB": self.ram_bytes / 1e12,
            "core_sec_per_step": self.sec_per_step,
        }


@dataclass
class LoadReport:
    L0: LevelLoad
    L1: LevelLoad
    L2: LevelLoad
    amr_total: LevelLoad   # PIC totals (L1 + L2)
    uniform_reference: LevelLoad
    n_steps_target: int


def _pic_level(name: str, dx_km: float, volume_Re3: float) -> LevelLoad:
    volume_km3 = volume_Re3 * (Re_km ** 3)
    n_cells = volume_km3 / (dx_km ** 3)
    n_particles = PPC * n_cells
    ram = n_particles * BYTES_PER_PARTICLE
    t = n_particles * SEC_PER_PARTICLE_PER_STEP
    return LevelLoad(name, dx_km, True, volume_Re3, n_cells, n_particles, ram, t)


def _mhd_level(name: str, dx_km: float, volume_Re3: float) -> LevelLoad:
    volume_km3 = volume_Re3 * (Re_km ** 3)
    n_cells = volume_km3 / (dx_km ** 3)
    return LevelLoad(name, dx_km, False, volume_Re3, n_cells, 0.0, 0.0, 0.0)


def uniform_reference(dx_km: float = REFERENCE_UNIFORM_DX_KM) -> LevelLoad:
    return _pic_level(f"uniform_{dx_km:.0f}km", dx_km, domain_volume_Re3())


def amr_load(
    sw: SolarWind = NOMINAL_SW,
    l1_pad_Re: float = 3.0,
    l2_band_Re: float = 1.5,
    sample_dx_Re: float = 0.5,
) -> tuple[LevelLoad, LevelLoad, LevelLoad, LevelLoad, ShellSample]:
    shell = sample_shell(
        sw=sw,
        l1_pad_Re=l1_pad_Re,
        l2_band_Re=l2_band_Re,
        sample_dx_Re=sample_dx_Re,
    )
    L0 = _mhd_level("L0 (MHD)", L0_DX_KM, domain_volume_Re3())
    L1 = _pic_level("L1 (PIC)", L1_DX_KM, shell.volume_L1_Re3)
    L2 = _pic_level("L2 (PIC)", L2_DX_KM, shell.volume_L2_Re3)
    total = LevelLoad(
        name="AMR PIC total",
        dx_km=float("nan"),
        is_pic=True,
        volume_Re3=shell.volume_L1_Re3,
        n_cells=L1.n_cells + L2.n_cells,
        n_particles=L1.n_particles + L2.n_particles,
        ram_bytes=L1.ram_bytes + L2.ram_bytes,
        sec_per_step=L1.sec_per_step + L2.sec_per_step,
    )
    return L0, L1, L2, total, shell


def build_report(
    sw: SolarWind = NOMINAL_SW,
    l1_pad_Re: float = 3.0,
    l2_band_Re: float = 1.5,
    sample_dx_Re: float = 0.5,
    reference_dx_km: float = REFERENCE_UNIFORM_DX_KM,
) -> tuple[LoadReport, ShellSample]:
    L0, L1, L2, total, shell = amr_load(
        sw=sw,
        l1_pad_Re=l1_pad_Re,
        l2_band_Re=l2_band_Re,
        sample_dx_Re=sample_dx_Re,
    )
    ref = uniform_reference(reference_dx_km)
    n_steps = int(round(TARGET_RUN_HOURS * 3600.0 / DT_SECONDS))
    return (
        LoadReport(
            L0=L0,
            L1=L1,
            L2=L2,
            amr_total=total,
            uniform_reference=ref,
            n_steps_target=n_steps,
        ),
        shell,
    )
