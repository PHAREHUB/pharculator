"""Command-line entry point: prints the load comparison and writes the figure."""

from __future__ import annotations

import argparse

from .constants import (
    SolarWind, NOMINAL_SW, BYTES_PER_PARTICLE, PPC, DELTA_I_KM,
    L0_DX_KM, L1_DX_KM, L2_DX_KM, L3_DX_KM, REFERENCE_UNIFORM_DX_KM,
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
    if seconds < 86400 * 365:
        return f"{seconds/86400:.2f} d"
    return f"{seconds/86400/365:.2f} yr"


def main(argv=None):
    p = argparse.ArgumentParser(description="PHARE AMR load estimator.")
    p.add_argument("--n", type=float, default=NOMINAL_SW.n_cm3)
    p.add_argument("--V", type=float, default=NOMINAL_SW.V_kms)
    p.add_argument("--Bz", type=float, default=NOMINAL_SW.Bz_nT)
    p.add_argument("--l1-pad-Re", type=float, default=3.0)
    p.add_argument("--l2-band-Re", type=float, default=1.5)
    p.add_argument("--l3-band-Re", type=float, default=0.5)
    p.add_argument("--sample-dx-Re", type=float, default=0.5)
    p.add_argument("--reference-dx-km", type=float, default=REFERENCE_UNIFORM_DX_KM)
    p.add_argument("--out", default="outputs/load_estimate.png")
    args = p.parse_args(argv)

    sw = SolarWind(n_cm3=args.n, V_kms=args.V, Bz_nT=args.Bz)
    report, shell = build_report(
        sw=sw,
        l1_pad_Re=args.l1_pad_Re,
        l2_band_Re=args.l2_band_Re,
        l3_band_Re=args.l3_band_Re,
        sample_dx_Re=args.sample_dx_Re,
        reference_dx_km=args.reference_dx_km,
    )

    print("=" * 80)
    print("Solar wind:")
    print(f"  n   = {sw.n_cm3:.2f} cm^-3")
    print(f"  V   = {sw.V_kms:.1f} km/s")
    print(f"  Bz  = {sw.Bz_nT:+.1f} nT")
    print(f"  Pd  = {pdyn_nPa(sw):.3f} nPa")
    print()
    print(f"Ion inertial length delta_i = {DELTA_I_KM:.0f} km")
    print(f"  L0 = {L0_DX_KM:>4.0f} km = 0.4 di  (MHD, full domain;        1 step / 64 L3)")
    print(f"  L1 = {L1_DX_KM:>4.0f} km = 0.4 di  (PIC, sheath ±{args.l1_pad_Re:.1f} Re;     1 step / 16 L3)")
    print(f"  L2 = {L2_DX_KM:>4.0f} km = 0.2 di  (PIC, ±{args.l2_band_Re:.1f} Re of MP+BS;  1 step /  4 L3)")
    print(f"  L3 = {L3_DX_KM:>4.0f} km = 0.1 di  (PIC, ±{args.l3_band_Re:.1f} Re of MP+BS;  base dt)")
    print(f"  reference = {args.reference_dx_km:.0f} km PIC uniform "
          f"(0.{int(round(args.reference_dx_km/DELTA_I_KM*10))} di, full domain, same dt as L3)")
    print()
    print("Model standoff:")
    print(f"  Shue magnetopause subsolar : {subsolar_mp(sw):.2f} Re")
    print(f"  Jelinek bow shock subsolar : {subsolar_bs(sw):.2f} Re")
    print()
    print(f"PIC volumes  (Re^3): L1 = {report.L1.volume_Re3:>10.1f}")
    print(f"                     L2 = {report.L2.volume_Re3:>10.1f}   "
          f"({100 * report.L2.volume_Re3 / report.L1.volume_Re3:.1f}% of L1)")
    print(f"                     L3 = {report.L3.volume_Re3:>10.1f}   "
          f"({100 * report.L3.volume_Re3 / report.L1.volume_Re3:.1f}% of L1)")
    print()
    print(f"Run target: {TARGET_RUN_HOURS} h of physical time, "
          f"dt_L3 = {DT_SECONDS} s  →  N = {report.n_steps_target:,} L3 steps")
    print(f"Particle = {BYTES_PER_PARTICLE} B, PPC = {PPC}, "
          f"cost = {SEC_PER_PARTICLE_PER_STEP*1e9:.0f} ns/particle/step")
    print("=" * 80)

    fmt = "{:>17}  {:>9}  {:>10}  {:>16}  {:>16}  {:>12}  {:>14}"
    print(fmt.format("level", "dx [km]", "steps/L3", "N_cells", "N_particles",
                     "RAM", "t/own step"))
    print("-" * 116)
    rows = [report.L0, report.L1, report.L2, report.L3,
            report.amr_total, report.uniform_reference]
    for L in rows:
        dx = "—" if L.dx_km != L.dx_km else f"{L.dx_km:.1f}"
        ram_str = _fmt_bytes(L.ram_bytes)
        t_str = _fmt_time(L.sec_per_step)
        print(fmt.format(
            L.name, dx,
            f"{L.steps_per_L3:.4g}",
            f"{L.n_cells:.3e}",
            f"{L.n_particles:.3e}" if L.is_pic else "      0",
            ram_str, t_str,
        ))
    print("-" * 116)

    L1, L2, L3 = report.L1, report.L2, report.L3
    ref = report.uniform_reference
    N = report.n_steps_target

    print()
    print(f"=== Over N = {N:,} L3 steps (= {N:,} uniform steps, same dt) ===")
    print()
    print("Per-level advances and particle-step work:")
    for L in (report.L0, L1, L2, L3):
        n_steps = N * L.steps_per_L3
        work = n_steps * L.n_particles
        tag = "MHD, not counted" if not L.is_pic else f"work = {work:.3e} part·steps"
        print(f"  {L.name:<10} : {n_steps:>14,.0f} advances    {tag}")

    pic_step_count = N * (L1.steps_per_L3 + L2.steps_per_L3 + L3.steps_per_L3)
    pic_work = N * (L1.steps_per_L3 * L1.n_particles
                    + L2.steps_per_L3 * L2.n_particles
                    + L3.steps_per_L3 * L3.n_particles)
    ref_steps = N
    ref_work = N * ref.n_particles

    print()
    print("Totals (PIC only — L0 has no particles):")
    print(f"  AMR PIC step count   : {pic_step_count:>16,.0f}")
    print(f"  Uniform step count   : {ref_steps:>16,.0f}")
    print(f"  AMR / uniform        : {pic_step_count / ref_steps:.3f} ×")
    print()
    print(f"  AMR particle·steps   : {pic_work:.3e}")
    print(f"  Uniform particle·steps: {ref_work:.3e}")
    print(f"  uniform / AMR        : {ref_work / pic_work:.2f} ×  (AMR cheaper)")
    print()
    print("AMR per-L3-step work breakdown (which level dominates the cost):")
    for L in (L1, L2, L3):
        w = L.steps_per_L3 * L.n_particles
        frac = 100 * w / (pic_work / N)
        print(f"  {L.name}: {w:.3e}  ({frac:5.2f}%)")
    print()
    print(f"Wall-time for the {N:,}-L3-step run on N cores "
          "(particle work only, MHD not counted):")
    amr_sec = pic_work * SEC_PER_PARTICLE_PER_STEP
    ref_sec = ref_work * SEC_PER_PARTICLE_PER_STEP
    for ncores in (1, 10_000, 100_000, 1_000_000):
        print(f"  N = {ncores:>9}  cores : "
              f"ref {_fmt_time(ref_sec/ncores):>14}   |   "
              f"AMR {_fmt_time(amr_sec/ncores):>14}")
    print()

    out = make_figure(report, shell, sw=sw, out_path=args.out)
    print(f"Figure written to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
