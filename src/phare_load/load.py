"""Memory and compute load estimates.

Four-level hierarchy:
- L0 : full domain, MHD only (no particles), 0.4 delta_i
- L1 : sheath shell with 3 Re buffers, PIC at 0.4 delta_i
- L2 : 1.5 Re bands around MP and BS, PIC at 0.2 delta_i
- L3 : 0.5 Re bands around MP and BS, PIC at 0.1 delta_i

Reference uniform run: 0.1 delta_i PIC over the full domain ("L0 only" run).

Subcycling: dt scales as dx^2, so each coarser level fires every 4 steps of
the next finer one. L3 carries the base dt (same as the uniform reference).
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
    L3_DX_KM,
    REFERENCE_UNIFORM_DX_KM,
    STEPS_PER_L3,
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
    steps_per_L3: float = 1.0

    @property
    def sec_per_L3_step(self) -> float:
        return self.sec_per_step * self.steps_per_L3


@dataclass
class LoadReport:
    L0: LevelLoad
    L1: LevelLoad
    L2: LevelLoad
    L3: LevelLoad
    amr_total: LevelLoad
    uniform_reference: LevelLoad
    n_steps_target: int


def _pic_level(name: str, dx_km: float, volume_Re3: float,
               steps_per_L3: float = 1.0) -> LevelLoad:
    volume_km3 = volume_Re3 * (Re_km ** 3)
    n_cells = volume_km3 / (dx_km ** 3)
    n_particles = PPC * n_cells
    ram = n_particles * BYTES_PER_PARTICLE
    t = n_particles * SEC_PER_PARTICLE_PER_STEP
    return LevelLoad(name, dx_km, True, volume_Re3, n_cells, n_particles, ram, t,
                     steps_per_L3=steps_per_L3)


def _mhd_level(name: str, dx_km: float, volume_Re3: float,
               steps_per_L3: float = 1.0) -> LevelLoad:
    volume_km3 = volume_Re3 * (Re_km ** 3)
    n_cells = volume_km3 / (dx_km ** 3)
    return LevelLoad(name, dx_km, False, volume_Re3, n_cells, 0.0, 0.0, 0.0,
                     steps_per_L3=steps_per_L3)


def uniform_reference(dx_km: float = REFERENCE_UNIFORM_DX_KM) -> LevelLoad:
    return _pic_level(f"uniform_{dx_km:.0f}km", dx_km, domain_volume_Re3(),
                      steps_per_L3=STEPS_PER_L3["uniform"])


def amr_load(
    sw: SolarWind = NOMINAL_SW,
    l1_pad_Re: float = 3.0,
    l2_band_Re: float = 1.5,
    l3_band_Re: float = 0.5,
    sample_dx_Re: float = 0.5,
    dayside_only: bool = True,
):
    shell = sample_shell(
        sw=sw,
        l1_pad_Re=l1_pad_Re,
        l2_band_Re=l2_band_Re,
        l3_band_Re=l3_band_Re,
        sample_dx_Re=sample_dx_Re,
        dayside_only=dayside_only,
    )
    L0 = _mhd_level("L0 (MHD)", L0_DX_KM, domain_volume_Re3(),
                    steps_per_L3=STEPS_PER_L3["L0"])
    L1 = _pic_level("L1 (PIC)", L1_DX_KM, shell.volume_L1_Re3,
                    steps_per_L3=STEPS_PER_L3["L1"])
    L2 = _pic_level("L2 (PIC)", L2_DX_KM, shell.volume_L2_Re3,
                    steps_per_L3=STEPS_PER_L3["L2"])
    L3 = _pic_level("L3 (PIC)", L3_DX_KM, shell.volume_L3_Re3,
                    steps_per_L3=STEPS_PER_L3["L3"])
    # AMR PIC work per L3 step = sum over PIC levels of n_particles * steps_per_L3
    total = LevelLoad(
        name="AMR PIC total",
        dx_km=float("nan"),
        is_pic=True,
        volume_Re3=shell.volume_L1_Re3,
        n_cells=L1.n_cells + L2.n_cells + L3.n_cells,
        n_particles=L1.n_particles + L2.n_particles + L3.n_particles,
        ram_bytes=L1.ram_bytes + L2.ram_bytes + L3.ram_bytes,
        sec_per_step=L1.sec_per_L3_step + L2.sec_per_L3_step + L3.sec_per_L3_step,
        steps_per_L3=1.0,
    )
    return L0, L1, L2, L3, total, shell


def build_report(
    sw: SolarWind = NOMINAL_SW,
    l1_pad_Re: float = 3.0,
    l2_band_Re: float = 1.5,
    l3_band_Re: float = 0.5,
    sample_dx_Re: float = 0.5,
    reference_dx_km: float = REFERENCE_UNIFORM_DX_KM,
    dayside_only: bool = True,
):
    L0, L1, L2, L3, total, shell = amr_load(
        sw=sw,
        l1_pad_Re=l1_pad_Re,
        l2_band_Re=l2_band_Re,
        l3_band_Re=l3_band_Re,
        sample_dx_Re=sample_dx_Re,
        dayside_only=dayside_only,
    )
    ref = uniform_reference(reference_dx_km)
    n_steps = int(round(TARGET_RUN_HOURS * 3600.0 / DT_SECONDS))
    return (
        LoadReport(
            L0=L0, L1=L1, L2=L2, L3=L3,
            amr_total=total,
            uniform_reference=ref,
            n_steps_target=n_steps,
        ),
        shell,
    )
