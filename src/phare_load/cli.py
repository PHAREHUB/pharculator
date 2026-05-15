"""Command-line entry point: prints the load comparison and writes the figure."""

from __future__ import annotations

import argparse

from .constants import (
    SolarWind, NOMINAL_SW, BYTES_PER_PARTICLE, PPC, DELTA_I_KM,
    L0_DX_KM, L1_DX_KM, L2_DX_KM, REFERENCE_UNIFORM_DX_KM,
    SEC_PER_PARTICLE_PER_STEP, DT_SECONDS, TARGET_RUN_HOURS,
)
from .load import build_report
from .models import subsolar_mp, subsolar_bs, pdyn_nPa
from .plotting import make_figure


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
    return f"{seconds/86400:.2f} d"


def main(argv=None):
    p = argparse.ArgumentParser(description="PHARE AMR load estimator.")
    p.add_argument("--n", type=float, default=NOMINAL_SW.n_cm3, help="SW density [cm-3]")
    p.add_argument("--V", type=float, default=NOMINAL_SW.V_kms, help="SW speed [km/s]")
    p.add_argument("--Bz", type=float, default=NOMINAL_SW.Bz_nT, help="IMF Bz [nT]")
    p.add_argument("--l1-pad-Re", type=float, default=3.0,
                   help="L1 buffer outside the BS and inside the MP")
    p.add_argument("--l2-band-Re", type=float, default=1.5,
                   help="Half-thickness of the L2 band around each boundary")
    p.add_argument("--sample-dx-Re", type=float, default=0.5,
                   help="Probe-grid resolution for volume sampling")
    p.add_argument("--reference-dx-km", type=float, default=REFERENCE_UNIFORM_DX_KM,
                   help="Reference uniform-PIC resolution for comparison")
    p.add_argument("--out", default="outputs/load_estimate.png")
    args = p.parse_args(argv)

    sw = SolarWind(n_cm3=args.n, V_kms=args.V, Bz_nT=args.Bz)
    report, shell = build_report(
        sw=sw,
        l1_pad_Re=args.l1_pad_Re,
        l2_band_Re=args.l2_band_Re,
        sample_dx_Re=args.sample_dx_Re,
        reference_dx_km=args.reference_dx_km,
    )

    print("=" * 76)
    print("Solar wind:")
    print(f"  n   = {sw.n_cm3:.2f} cm^-3")
    print(f"  V   = {sw.V_kms:.1f} km/s")
    print(f"  Bz  = {sw.Bz_nT:+.1f} nT")
    print(f"  Pd  = {pdyn_nPa(sw):.3f} nPa")
    print()
    print(f"Ion inertial length delta_i = {DELTA_I_KM:.0f} km")
    print(f"  L0 = {L0_DX_KM:.0f} km = 0.4 delta_i  (MHD, full domain)")
    print(f"  L1 = {L1_DX_KM:.0f} km = 0.2 delta_i  (PIC, sheath shell ±{args.l1_pad_Re} Re)")
    print(f"  L2 = {L2_DX_KM:.0f} km = 0.1 delta_i  (PIC, {args.l2_band_Re} Re around MP and BS)")
    print(f"  reference uniform PIC = {args.reference_dx_km:.0f} km = "
          f"{args.reference_dx_km/DELTA_I_KM:.2f} delta_i (full domain)")
    print()
    print("Model standoff:")
    print(f"  Shue magnetopause subsolar : {subsolar_mp(sw):.2f} Re")
    print(f"  Jelinek bow shock subsolar : {subsolar_bs(sw):.2f} Re")
    print()
    print(f"L1 volume : {report.L1.volume_Re3:.1f} Re^3")
    print(f"L2 volume : {report.L2.volume_Re3:.1f} Re^3   "
          f"({100 * report.L2.volume_Re3 / report.L1.volume_Re3:.1f}% of L1)")
    print()
    print(f"Run target: {TARGET_RUN_HOURS} h of physical time, "
          f"dt = {DT_SECONDS} s  →  {report.n_steps_target:,} steps")
    print(f"Particle = {BYTES_PER_PARTICLE} B (7 × float64), PPC = {PPC}")
    print(f"Cost model: {SEC_PER_PARTICLE_PER_STEP*1e9:.0f} ns / particle / step "
          "(aggregate single-thread)")
    print("=" * 76)

    fmt = "{:>17}  {:>9}  {:>16}  {:>16}  {:>12}  {:>14}"
    print(fmt.format("level", "dx [km]", "N_cells", "N_particles", "RAM", "t/step"))
    print("-" * 100)
    rows = [report.L0, report.L1, report.L2, report.amr_total, report.uniform_reference]
    for L in rows:
        dx = "—" if L.dx_km != L.dx_km else f"{L.dx_km:.1f}"
        ram_str = _fmt_bytes(L.ram_bytes)
        t_str = _fmt_time(L.sec_per_step)
        print(fmt.format(
            L.name, dx,
            f"{L.n_cells:.3e}",
            f"{L.n_particles:.3e}" if L.is_pic else "      0",
            ram_str, t_str,
        ))
    print("-" * 100)

    amr = report.amr_total
    ref = report.uniform_reference
    print()
    print(f"Reference uniform {ref.dx_km:.0f} km PIC / AMR (L1+L2) ratios:")
    print(f"  particles : {ref.n_particles / amr.n_particles:8.1f} ×")
    print(f"  RAM       : {ref.ram_bytes   / amr.ram_bytes  :8.1f} ×")
    print(f"  time/step : {ref.sec_per_step/ amr.sec_per_step:8.1f} ×")
    print()
    print(f"Wall-time for {report.n_steps_target:,} steps on N cores:")
    for ncores in (1, 10_000, 100_000, 1_000_000):
        t_ref = ref.sec_per_step * report.n_steps_target / ncores
        t_amr = amr.sec_per_step * report.n_steps_target / ncores
        print(f"  N = {ncores:>9}  cores : "
              f"ref {_fmt_time(t_ref):>14}   |   AMR {_fmt_time(t_amr):>14}")
    print()

    out = make_figure(report, shell, sw=sw, out_path=args.out)
    print(f"Figure written to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
