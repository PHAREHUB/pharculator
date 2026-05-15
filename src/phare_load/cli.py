"""Command-line entry point: prints the load comparison and writes the figure."""

from __future__ import annotations

import argparse

from .constants import (
    SolarWind, NOMINAL_SW, BYTES_PER_PARTICLE, PPC,
    SEC_PER_PARTICLE_PER_STEP, DT_SECONDS, TARGET_RUN_HOURS,
)
from .load import build_report
from .models import subsolar_mp, subsolar_bs, pdyn_nPa
from .plotting import make_figure


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB", "PB", "EB"):
        if n < 1024.0:
            return f"{n:7.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} ZB"


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} s"
    if seconds < 3600:
        return f"{seconds/60:.2f} min"
    if seconds < 86400:
        return f"{seconds/3600:.2f} h"
    return f"{seconds/86400:.2f} d"


def main(argv=None):
    p = argparse.ArgumentParser(description="PHARE AMR load estimator.")
    p.add_argument("--n", type=float, default=NOMINAL_SW.n_cm3, help="SW density [cm-3]")
    p.add_argument("--V", type=float, default=NOMINAL_SW.V_kms, help="SW speed [km/s]")
    p.add_argument("--Bz", type=float, default=NOMINAL_SW.Bz_nT, help="IMF Bz [nT]")
    p.add_argument("--uniform-dx-km", type=float, default=100.0)
    p.add_argument("--pad-Re", type=float, default=2.0,
                   help="Buffer outside BS and inside MP for L0 shell")
    p.add_argument("--sample-dx-Re", type=float, default=0.5,
                   help="Volume sampling resolution in Re (smaller = more accurate)")
    p.add_argument("--out", default="outputs/load_estimate.png")
    args = p.parse_args(argv)

    sw = SolarWind(n_cm3=args.n, V_kms=args.V, Bz_nT=args.Bz)
    report, shell = build_report(
        sw=sw,
        uniform_dx_km=args.uniform_dx_km,
        pad_Re=args.pad_Re,
        sample_dx_Re=args.sample_dx_Re,
    )

    print("=" * 72)
    print("Solar wind:")
    print(f"  n   = {sw.n_cm3:.2f} cm^-3")
    print(f"  V   = {sw.V_kms:.1f} km/s")
    print(f"  Bz  = {sw.Bz_nT:+.1f} nT")
    print(f"  Pd  = {pdyn_nPa(sw):.3f} nPa")
    print()
    print("Model standoff:")
    print(f"  Shue magnetopause subsolar : {subsolar_mp(sw):.2f} Re")
    print(f"  Jelinek bow shock subsolar : {subsolar_bs(sw):.2f} Re")
    print()
    print(f"Shell volume (L0 PIC footprint, 2Re buffers): "
          f"{shell.volume_Re3:.2f} Re^3   "
          f"(sampling dx = {args.sample_dx_Re} Re)")
    print()
    print(f"Run target: {TARGET_RUN_HOURS} h of physical time, "
          f"dt = {DT_SECONDS} s  →  {report.n_steps_target:,} steps")
    print(f"Particle = {BYTES_PER_PARTICLE} B (7 × float64), PPC = {PPC}")
    print(f"Cost model: {SEC_PER_PARTICLE_PER_STEP*1e9:.0f} ns / particle / step "
          "(aggregate single-thread)")
    print("=" * 72)

    fmt = "{:>14}  {:>10}  {:>16}  {:>16}  {:>12}  {:>16}"
    print(fmt.format("level", "dx [km]", "N_cells", "N_particles", "RAM", "t/step"))
    print("-" * 96)
    from .load import uniform_load
    uni10 = uniform_load(10.0)
    rows = [report.uniform, uni10] + report.amr_levels + [report.amr_total]
    for L in rows:
        dx = "—" if L.dx_km != L.dx_km else f"{L.dx_km:.1f}"   # nan check
        ram_str = _fmt_bytes(L.ram_bytes)
        t_str = _fmt_time(L.sec_per_step)
        print(fmt.format(L.name, dx,
                         f"{L.n_cells:.3e}",
                         f"{L.n_particles:.3e}",
                         ram_str, t_str))
    print("-" * 96)

    uni = report.uniform
    amr = report.amr_total
    print()
    print("Reference ratios (particles):")
    print(f"  AMR / uniform_100km            : {amr.n_particles / uni.n_particles:8.2f} ×")
    print(f"  uniform_10km / AMR             : {uni10.n_particles / amr.n_particles:8.1f} ×")
    print(f"  uniform_10km / uniform_100km   : {uni10.n_particles / uni.n_particles:8.1f} ×")
    print()
    print(f"Wall-time for {report.n_steps_target:,} steps on N cores:")
    for ncores in (1, 1_000, 10_000, 100_000):
        t_uni = uni.sec_per_step * report.n_steps_target / ncores
        t_amr = amr.sec_per_step * report.n_steps_target / ncores
        print(f"  N = {ncores:>7}  cores : "
              f"uniform {_fmt_time(t_uni):>10}   |   AMR {_fmt_time(t_amr):>10}")
    print()

    out = make_figure(report, shell, sw=sw, out_path=args.out)
    print(f"Figure written to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
