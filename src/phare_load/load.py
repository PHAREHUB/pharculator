"""Memory and compute load estimates: uniform baseline vs AMR scenario."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

from .constants import (
    Re_km,
    BYTES_PER_PARTICLE,
    PPC,
    SEC_PER_PARTICLE_PER_STEP,
    DT_SECONDS,
    TARGET_RUN_HOURS,
    SolarWind,
    NOMINAL_SW,
)
from .geometry import domain_extent_km, sample_shell, ShellSample


@dataclass
class LevelLoad:
    name: str
    dx_km: float
    volume_Re3: float
    n_cells: float
    n_particles: float
    ram_bytes: float
    sec_per_step: float       # aggregate single-thread seconds per timestep

    def as_row(self):
        return {
            "level": self.name,
            "dx_km": self.dx_km,
            "volume_Re3": self.volume_Re3,
            "n_cells": self.n_cells,
            "n_particles": self.n_particles,
            "ram_TB": self.ram_bytes / 1e12,
            "core_sec_per_step": self.sec_per_step,
        }


@dataclass
class LoadReport:
    uniform: LevelLoad
    amr_levels: list   # list[LevelLoad]
    amr_total: LevelLoad
    shell_volume_Re3: float
    n_steps_target: int

    def to_dict(self) -> Dict:
        return {
            "uniform": self.uniform.as_row(),
            "amr_levels": [l.as_row() for l in self.amr_levels],
            "amr_total": self.amr_total.as_row(),
            "shell_volume_Re3": self.shell_volume_Re3,
            "n_steps_target": self.n_steps_target,
        }


def _level(name: str, dx_km: float, volume_Re3: float) -> LevelLoad:
    volume_km3 = volume_Re3 * (Re_km ** 3)
    n_cells = volume_km3 / (dx_km ** 3)
    n_particles = PPC * n_cells
    ram = n_particles * BYTES_PER_PARTICLE
    t = n_particles * SEC_PER_PARTICLE_PER_STEP
    return LevelLoad(name, dx_km, volume_Re3, n_cells, n_particles, ram, t)


def uniform_load(dx_km: float = 100.0) -> LevelLoad:
    Lx, Ly, Lz = domain_extent_km()
    volume_km3 = Lx * Ly * Lz
    volume_Re3 = volume_km3 / (Re_km ** 3)
    return _level(f"uniform_{int(dx_km)}km", dx_km, volume_Re3)


def amr_load(
    sw: SolarWind = NOMINAL_SW,
    pad_Re: float = 2.0,
    sample_dx_Re: float = 0.5,
    l1_fraction: float = 0.10,
    l2_fraction_of_l1: float = 0.50,
) -> tuple[list[LevelLoad], LevelLoad, ShellSample]:
    shell = sample_shell(sw=sw, pad_Re=pad_Re, sample_dx_Re=sample_dx_Re)
    V0 = shell.volume_Re3
    V1 = l1_fraction * V0
    V2 = l2_fraction_of_l1 * V1

    L0 = _level("L0", 40.0, V0)
    L1 = _level("L1", 20.0, V1)
    L2 = _level("L2", 10.0, V2)

    total = LevelLoad(
        name="AMR total",
        dx_km=float("nan"),
        volume_Re3=V0,  # PIC footprint
        n_cells=L0.n_cells + L1.n_cells + L2.n_cells,
        n_particles=L0.n_particles + L1.n_particles + L2.n_particles,
        ram_bytes=L0.ram_bytes + L1.ram_bytes + L2.ram_bytes,
        sec_per_step=L0.sec_per_step + L1.sec_per_step + L2.sec_per_step,
    )
    return [L0, L1, L2], total, shell


def build_report(
    sw: SolarWind = NOMINAL_SW,
    uniform_dx_km: float = 100.0,
    pad_Re: float = 2.0,
    sample_dx_Re: float = 0.5,
) -> tuple[LoadReport, ShellSample]:
    uniform = uniform_load(uniform_dx_km)
    levels, total, shell = amr_load(sw=sw, pad_Re=pad_Re, sample_dx_Re=sample_dx_Re)
    n_steps = int(round(TARGET_RUN_HOURS * 3600.0 / DT_SECONDS))
    return (
        LoadReport(
            uniform=uniform,
            amr_levels=levels,
            amr_total=total,
            shell_volume_Re3=shell.volume_Re3,
            n_steps_target=n_steps,
        ),
        shell,
    )
