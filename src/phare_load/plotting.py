"""Meridional + equatorial plots of the AMR levels.

Coordinates are standard GSE: +x toward the Sun, -x toward the tail.
Plots invert the x-axis so the Sun ends up on the LEFT (magnetospheric
convention), with the tail extending to the right.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from .config import Config
from .geometry import RegionSample
from .models import shue_mp, jelinek_bs
from .load import LoadReport


_THETA_CURVE_MAX = np.deg2rad(178.0)


def _model_curve(model_fn, sw, xlim, ylim, n=2000, dipole_strength=1.0):
    """Return (x_gse, y) for both lobes of the model surface, clipped to the plot box.

    The model curve is rotationally symmetric around the GSE x-axis, so the same
    curve serves both meridional (X-Z) and equatorial (X-Y) slices.
    """
    theta = np.linspace(0.0, _THETA_CURVE_MAX, n)
    r = model_fn(theta, sw, dipole_strength=dipole_strength)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    x_lo, x_hi = sorted(xlim)
    inside = (x >= x_lo) & (x <= x_hi) & (y <= ylim)
    if inside.any():
        kmax = int(np.where(inside)[0].max()) + 1
        x = x[:kmax]
        y = y[:kmax]
    nan = np.array([np.nan])
    x_full = np.concatenate([x, nan, x])
    y_full = np.concatenate([y, nan, -y])
    return x_full, y_full


def _slice_index(sample: RegionSample, axis: str):
    if axis == "xz":
        return int(np.argmin(np.abs(sample.y[0, :, 0])))
    elif axis == "xy":
        return int(np.argmin(np.abs(sample.z[0, 0, :])))
    raise ValueError(axis)


def _slice(arr3d, axis, idx):
    if axis == "xz":
        return arr3d[:, idx, :]
    return arr3d[:, :, idx]


def _level_colors(n: int):
    cmap = plt.get_cmap("Blues")
    return [cmap(0.30 + 0.65 * i / max(n - 1, 1)) for i in range(n)]


def _shade_levels(ax, sample: RegionSample, report: LoadReport, axis: str):
    d = report.cfg.domain
    ax.add_patch(Rectangle(
        (d.x_min, d.z_min if axis == "xz" else d.y_min),
        d.x_max - d.x_min,
        (d.z_max - d.z_min) if axis == "xz" else (d.y_max - d.y_min),
        facecolor="#f0f4f8", edgecolor="none", zorder=0,
    ))

    pic_specs = [L for L in report.cfg.levels if L.kind == "pic"]
    if not pic_specs:
        return
    idx = _slice_index(sample, axis)
    X = _slice(sample.x, axis, idx)
    V = _slice(sample.z if axis == "xz" else sample.y, axis, idx)
    lvl = np.zeros_like(X, dtype=int)
    for i, spec in enumerate(pic_specs, start=1):
        m = _slice(sample.masks[spec.name], axis, idx)
        lvl[m] = i
    colors = _level_colors(len(pic_specs))
    ax.contourf(
        X, V, lvl.astype(float),
        levels=[i - 0.5 for i in range(1, len(pic_specs) + 2)],
        colors=colors, alpha=0.85, zorder=1,
    )


def _draw_panel(ax, sample: RegionSample, report: LoadReport,
                xlim=None, ylim=None, title="Meridional slice (Y=0)",
                annotate=True):
    cfg = report.cfg
    d = cfg.domain
    if xlim is None:
        xlim = (d.x_min, d.x_max)
    if ylim is None:
        ylim = (d.z_min, d.z_max)
    # Sun on the LEFT: invert the x-axis so larger (sunward) x sits at the left.
    ax.set_xlim(max(xlim), min(xlim))
    ax.set_ylim(*ylim)
    ax.set_xlabel("X_GSE [Re]    (Sun on the left)")
    ax.set_ylabel("Z [Re]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.4)
    axis = "xz"

    _shade_levels(ax, sample, report, axis)

    xl = ax.get_xlim()
    yl = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
    xmp, ymp = _model_curve(shue_mp, cfg.solar_wind, xl, yl,
                            dipole_strength=cfg.dipole_strength)
    xbs, ybs = _model_curve(jelinek_bs, cfg.solar_wind, xl, yl,
                            dipole_strength=cfg.dipole_strength)
    ax.plot(xmp, ymp, color="#b30000", lw=1.6, label="Magnetopause (Shue 1998)")
    ax.plot(xbs, ybs, color="#08306b", lw=1.6, label="Bow shock (Jelínek 2012)")

    ax.add_patch(Circle((0, 0), 1.0, color="k", zorder=5))
    ax.plot(0, 0, "wo", ms=2.5, zorder=6)
    ax.annotate("Earth", (0, 0), textcoords="offset points", xytext=(6, 6),
                fontsize=8, color="k")

    if annotate:
        lines = []
        for L in report.levels:
            if L.is_pic:
                lines.append(f"{L.name} ({L.dx_km:>4.0f} km, PIC): "
                             f"N = {L.n_particles:.2e}")
            else:
                lines.append(f"{L.name} ({L.dx_km:>4.0f} km, MHD): no particles")
        lines.append(f"Σ AMR PIC:           N = {report.amr_total.n_particles:.2e}")
        ref = report.uniform_reference
        lines.append(f"uniform {ref.dx_km:.0f} km PIC:   N = {ref.n_particles:.2e}")
        # Tail side (low x, large -x) is the RIGHT half of the panel after
        # inverting the axis — put the annotation there so it doesn't cover
        # the dayside boundaries (which sit on the left).
        ax.text(0.98, 0.98, "\n".join(lines), transform=ax.transAxes,
                va="top", ha="right", fontsize=8,
                family="monospace",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.6"))

    ax.legend(loc="lower left", fontsize=8)


def _dayside_zoom(report: LoadReport):
    cfg = report.cfg
    from .models import subsolar_mp, subsolar_bs
    r_mp = subsolar_mp(cfg.solar_wind, dipole_strength=cfg.dipole_strength)
    r_bs = subsolar_bs(cfg.solar_wind, dipole_strength=cfg.dipole_strength)
    # Dayside is +x. Show from a few Re past Earth into the dayside out to
    # ~4 Re beyond the BS standoff.
    x_min = -max(8.0, 0.7 * r_mp)
    x_max = 1.6 * r_bs
    half = 1.5 * r_bs
    return (x_min, x_max), (-half, half)


def make_figure(report: LoadReport, sample: RegionSample,
                out_path: str = "outputs/load_estimate.png"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    _draw_panel(axes[0], sample, report,
                title="Meridional slice (Y=0) — full domain")
    xlim, ylim = _dayside_zoom(report)
    _draw_panel(axes[1], sample, report,
                xlim=xlim, ylim=ylim,
                title="Dayside zoom (Y=0)",
                annotate=False)
    names = ", ".join(L.name for L in report.cfg.levels)
    fig.suptitle(f"PHARE global magnetosphere — levels: {names}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
