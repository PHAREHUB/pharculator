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
    print(f"Run target: {cfg.target_hours} h physical, "
          f"dt_finest = {cfg.dt_finest_s} s  →  N = {cfg.n_steps_target:,} steps")
    print(f"dt ratio per level = {cfg.dt_ratio_per_level}")
    print(f"Reference uniform PIC dx = {cfg.reference_dx_km} km  "
          f"({cfg.reference_dx_km/cfg.delta_i_km:.2f} delta_i)")
    print()
    print("Levels (coarsest → finest):")
    for L in cfg.levels:
        region = L.region
        if region == "shell":
            region = f"shell  pad={L.pad_re:.1f} Re"
        elif region == "band":
            region = f"band   ±{L.band_re:.1f} Re of MP and BS"
        else:
            region = "full domain"
        print(f"  {L.name:<5} {L.kind.upper():<3}  dx = {L.dx_km:>5.1f} km "
              f"({L.dx_km/cfg.delta_i_km:>4.2f} di)   region: {region}   "
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

    fmt = "{:>20}  {:>9}  {:>10}  {:>16}  {:>16}  {:>12}  {:>14}"
    print(fmt.format("level", "dx [km]", "steps/fin", "N_cells", "N_particles",
                     "RAM", "t/own step"))
    print("-" * 116)
    rows = list(report.levels) + [report.amr_total, report.uniform_reference]
    for L in rows:
        dx = "—" if L.dx_km != L.dx_km else f"{L.dx_km:.1f}"
        print(fmt.format(
            L.name, dx,
            f"{L.steps_per_finest:.4g}",
            f"{L.n_cells:.3e}",
            f"{L.n_particles:.3e}" if L.is_pic else "      0",
            _fmt_bytes(L.ram_bytes),
            _fmt_time(L.sec_per_step),
        ))
    print("-" * 116)

    N = report.n_steps_target
    pic = report.pic_levels
    ref = report.uniform_reference

    pic_step_count = N * sum(L.steps_per_finest for L in pic)
    pic_work = N * sum(L.steps_per_finest * L.n_particles for L in pic)
    ref_work = N * ref.n_particles

    print()
    print(f"=== Over N = {N:,} finest-level steps ===")
    print()
    print("Per-level advances and particle·step work:")
    for L in report.levels:
        n_steps = N * L.steps_per_finest
        tag = "MHD, not counted" if not L.is_pic else (
            f"work = {n_steps * L.n_particles:.3e} part·steps")
        print(f"  {L.name:<6} : {n_steps:>14,.0f} advances    {tag}")
    print()
    print("Totals (PIC only):")
    print(f"  AMR PIC step count   : {pic_step_count:>16,.0f}")
    print(f"  Uniform step count   : {N:>16,.0f}")
    print(f"  AMR / uniform        : {pic_step_count / N:.3f} ×")
    print()
    print(f"  AMR particle·steps   : {pic_work:.3e}")
    print(f"  Uniform particle·steps: {ref_work:.3e}")
    print(f"  uniform / AMR        : {ref_work / pic_work:.2f} ×  (AMR cheaper)")
    print()
    if pic:
        print("AMR per-finest-step work breakdown:")
        per_step_total = pic_work / N
        for L in pic:
            w = L.steps_per_finest * L.n_particles
            frac = 100 * w / per_step_total
            print(f"  {L.name}: {w:.3e}  ({frac:5.2f}%)")
    print()
    amr_sec = pic_work * cfg.sec_per_particle_per_step
    ref_sec = ref_work * cfg.sec_per_particle_per_step
    print("CPU-hours (= core-hours, independent of core count):")
    print(f"  reference uniform : {ref_sec/3600:>14,.0f} CPU·h "
          f"({ref_sec/3600/1e6:.2f} M CPU·h)")
    print(f"  AMR (sum PIC)     : {amr_sec/3600:>14,.0f} CPU·h "
          f"({amr_sec/3600/1e6:.2f} M CPU·h)")
    print()
    print("Wall-time on N cores (ideal linear scaling, PIC work only):")
    for ncores in (1, 10_000, 100_000, 1_000_000):
        print(f"  N = {ncores:>9}  cores : "
              f"ref {_fmt_time(ref_sec/ncores):>14}   |   "
              f"AMR {_fmt_time(amr_sec/ncores):>14}")
    print()

    out = make_figure(report, sample, out_path=args.out)
    print(f"Figure written to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
