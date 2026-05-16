"""Command-line entry point: takes a TOML config, prints the comparison and
writes the figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config, Config
from .load import build_report
from .models import subsolar_mp, subsolar_bs
from .plotting import make_figure


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent.parent / "config.toml"


def _fmt_bytes(n: float) -> str:
    if n <= 0:
        return "       —"
    for unit in ("B", "KB", "MB", "GB", "TB", "PB", "EB"):
        if n < 1024.0:
            return f"{n:7.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} ZB"


def _fmt_time(seconds: float) -> str:
    if seconds <= 0:
        return "    —"
    if seconds < 60:
        return f"{seconds:.2f} s"
    if seconds < 3600:
        return f"{seconds/60:.2f} min"
    if seconds < 86400:
        return f"{seconds/3600:.2f} h"
    if seconds < 86400 * 365:
        return f"{seconds/86400:.2f} d"
    return f"{seconds/86400/365:.2f} yr"


def _summary(cfg: Config) -> None:
    sw = cfg.solar_wind
    print("=" * 80)
    print(f"Config: delta_i = {cfg.delta_i_km:.0f} km, "
          f"PPC = {cfg.ppc}, particle = {cfg.bytes_per_particle} B")
    print(f"Solar wind: n = {sw.n_cm3:.2f} cm^-3, V = {sw.v_kms:.1f} km/s, "
          f"Bz = {sw.bz_nt:+.1f} nT  →  Pd = {sw.Pdyn_nPa:.3f} nPa")
    print(f"Domain: x in [{cfg.domain.x_min},{cfg.domain.x_max}] Re, "
          f"y,z in [{cfg.domain.y_min},{cfg.domain.y_max}] Re")
    print(f"Dayside-only PIC: {cfg.dayside_only}")
    dt_finest_s = cfg.resolve_dt_finest_s()
    dt_finest_omega_ci = dt_finest_s / cfg.omega_ci_inverse_s
    print(f"Time stepping: dt_L1 = {cfg.dt_L1_omega_ci} /Ω_ci, "
          f"dt scales as dx² (ratio per level = {cfg.dt_ratio_per_level})")
    print(f"  Ω_ci⁻¹ = {cfg.omega_ci_inverse_s:.2f} s")
    print(f"  dt_finest = {dt_finest_omega_ci:.4g} /Ω_ci "
          f"= {dt_finest_s:.4g} s")
    print(f"Run target: {cfg.target_hours} h physical  →  "
          f"N_finest = {cfg.n_steps_target:,} steps")
    ref_dx_km = cfg.resolve_reference_dx_km()
    print(f"Reference uniform PIC dx = {ref_dx_km} km  "
          f"({ref_dx_km/cfg.delta_i_km:.2f} delta_i)")
    print()
    print("Levels (coarsest → finest):")
    for L in cfg.levels:
        dx_km = L.resolve_dx_km(cfg.delta_i_km)
        region = L.region
        if region == "shell":
            region = f"shell  pad={L.pad_re:.1f} Re"
        elif region == "band":
            region = f"band   ±{L.band_re:.1f} Re of MP and BS"
        else:
            region = "full domain"
        print(f"  {L.name:<5} {L.kind.upper():<3}  dx = {dx_km:>5.1f} km "
              f"({dx_km/cfg.delta_i_km:>4.2f} di)   region: {region}   "
              f"steps/finest = {cfg.steps_per_finest(L.name):.4g}")
    print("=" * 80)


def main(argv=None):
    p = argparse.ArgumentParser(description="PHARE AMR load estimator.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help="Path to a TOML config file (default: bundled config.toml).")
    p.add_argument("--out", default="outputs/load_estimate.png")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    print(f"Loaded config: {args.config}")
    _summary(cfg)

    report, sample = build_report(cfg)

    ds = cfg.dipole_strength
    print(f"Shue MP subsolar    : {subsolar_mp(cfg.solar_wind, ds):.2f} Re   "
          f"(dipole_strength = {ds:.3g})")
    print(f"Jelinek BS subsolar : {subsolar_bs(cfg.solar_wind, ds):.2f} Re")
    print()
    print("Per-level volumes (PIC only):")
    for L in report.pic_levels:
        v_l1 = report.pic_levels[0].volume_Re3 if report.pic_levels else 1.0
        frac = 100.0 * L.volume_Re3 / max(v_l1, 1e-30)
        print(f"  {L.name}: {L.volume_Re3:>10.1f} Re^3  ({frac:5.1f}% of {report.pic_levels[0].name})")
    print()

    N = report.n_steps_target
    pic = report.pic_levels
    ref = report.uniform_reference

    # ----- 1. Memory footprint per level -------------------------------------
    print()
    print("─── 1. Memory footprint ─────────────────────────────────────────────")
    fmt_mem = "  {:<10}  dx = {:>5.1f} km  {:<3}  N_cells = {:>10.3e}   N_part = {:>10}   RAM = {}"
    for L in report.levels:
        npart = f"{L.n_particles:.3e}" if L.is_pic else "       —"
        print(fmt_mem.format(
            L.name, L.dx_km, L.kind.upper(), L.n_cells, npart,
            _fmt_bytes(L.ram_bytes)))
    amr_ram = sum(L.ram_bytes for L in pic)
    print(f"  → Total AMR PIC footprint:                            "
          f"RAM = {_fmt_bytes(amr_ram)}")
    print(f"    (reference uniform {ref.dx_km:.0f} km PIC)            "
          f"RAM = {_fmt_bytes(ref.ram_bytes)}")

    # ----- 2. Number of timesteps per level over the run ---------------------
    print()
    print("─── 2. Number of timesteps over the run ─────────────────────────────")
    print(f"  Run duration: {cfg.target_hours:.3g} h "
          f"= {cfg.target_hours * 3600 / cfg.omega_ci_inverse_s:.4g} /Ω_ci")
    for L in report.levels:
        dt_L_s = report.uniform_reference.dx_km  # placeholder, replaced below
        dt_L_s = cfg.resolve_dt_finest_s() / L.steps_per_finest
        dt_L_omega = dt_L_s / cfg.omega_ci_inverse_s
        n_steps = N * L.steps_per_finest
        print(f"  {L.name:<6}  dt = {dt_L_omega:.4g}/Ω_ci = {dt_L_s:.4g} s   "
              f"steps = {n_steps:>14,.0f}")
    print(f"  {'Σ all levels':<6}                                "
          f"steps = {sum(N * L.steps_per_finest for L in report.levels):>14,.0f}")
    print(f"  uniform {ref.dx_km:.0f} km  dt = {cfg.resolve_dt_finest_s()/cfg.omega_ci_inverse_s:.4g}/Ω_ci   "
          f"steps = {N:>14,.0f}")

    # ----- 3. Particle pushes per level --------------------------------------
    print()
    print("─── 3. Particle pushes per level ────────────────────────────────────")
    pic_work_per_level = {}
    for L in report.levels:
        if L.is_pic:
            n_steps = N * L.steps_per_finest
            pushes = n_steps * L.n_particles
            pic_work_per_level[L.name] = pushes
            print(f"  {L.name:<6}  pushes = {pushes:.3e}")
        else:
            print(f"  {L.name:<6}  pushes = — (MHD, no particles)")
    pic_work = sum(pic_work_per_level.values())
    ref_work = N * ref.n_particles
    print()
    print(f"  Σ AMR hierarchy (PIC)            : {pic_work:.3e} particle pushes")
    print(f"  equivalent uniform {ref.dx_km:.0f} km PIC    : {ref_work:.3e} particle pushes")
    print(f"  ratio uniform / AMR              : {ref_work / pic_work:.2f} ×")

    # ----- 4. Per-finest-step work breakdown ---------------------------------
    if pic:
        print()
        print("─── 4. Work breakdown per finest-level step ─────────────────────────")
        per_step_total = pic_work / N
        for L in pic:
            w = L.steps_per_finest * L.n_particles
            frac = 100 * w / per_step_total
            print(f"  {L.name}: {w:.3e}  ({frac:5.2f}% of AMR per-step work)")

    # ----- 5. CPU·hours -------------------------------------------------------
    print()
    print("─── 5. CPU·hours (cost = 10 ns / particle / push) ───────────────────")
    amr_sec = pic_work * cfg.sec_per_particle_per_step
    ref_sec = ref_work * cfg.sec_per_particle_per_step
    print(f"  AMR hierarchy            : {amr_sec/3600:>14,.0f} CPU·h "
          f"({amr_sec/3600/1e6:.2f} M CPU·h)")
    print(f"  equivalent uniform {ref.dx_km:.0f} km : {ref_sec/3600:>14,.0f} CPU·h "
          f"({ref_sec/3600/1e6:.2f} M CPU·h)")
    print(f"  ratio uniform / AMR      : {ref_sec / amr_sec:.2f} ×  (AMR cheaper)")
    print()
    print("  Wall-time on N cores (ideal linear scaling):")
    for ncores in (1, 10_000, 100_000, 1_000_000):
        print(f"    N = {ncores:>9}  cores : "
              f"uniform {_fmt_time(ref_sec/ncores):>14}   |   "
              f"AMR {_fmt_time(amr_sec/ncores):>14}")
    print()

    out = make_figure(report, sample, out_path=args.out)
    print(f"Figure written to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
