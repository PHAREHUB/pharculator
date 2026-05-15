"""Meridional + equatorial plots of the AMR levels."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib import colors as mcolors

from .config import Config
from .geometry import RegionSample
from .models import shue_mp, jelinek_bs
from .load import LoadReport


_THETA_CURVE_MAX = np.deg2rad(178.0)


def _model_curve_user(model_fn, sw, xlim, ylim, n=2000, dipole_strength=1.0):
    theta = np.linspace(0.0, _THETA_CURVE_MAX, n)
    r = model_fn(theta, sw, dipole_strength=dipole_strength)
    x_gse = r * np.cos(theta)
    y = r * np.sin(theta)
    x_user = -x_gse
    inside = (x_user >= xlim[0]) & (x_user <= xlim[1]) & (y <= ylim)
    if inside.any():
        kmax = int(np.where(inside)[0].max()) + 1
        x_user = x_user[:kmax]
        y = y[:kmax]
    nan = np.array([np.nan])
    x_full = np.concatenate([x_user, nan, x_user])
    y_full = np.concatenate([y, nan, -y])
    return x_full, y_full


def _slice_index(arr, axis):
    if axis == "xz":
        return np.argmin(np.abs(arr.y[0, :, 0])), "iy"
    elif axis == "xy":
        return np.argmin(np.abs(arr.z[0, 0, :])), "iz"
    raise ValueError(axis)


def _slice(arr3d, axis, idx):
    if axis == "xz":
        return arr3d[:, idx, :]
    return arr3d[:, :, idx]


def _level_colors(n: int):
    cmap = plt.get_cmap("Blues")
    # Pick progressively darker shades, skipping the very light end.
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
    idx, _ = _slice_index(sample, axis)
    X = _slice(sample.x_user, axis, idx)
    V = _slice(sample.z if axis == "xz" else sample.y, axis, idx)
    # Stack masks: cell value = highest level index that contains it.
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
    return colors


def _draw_panel(ax, axis: str, sample: RegionSample, report: LoadReport):
    cfg = report.cfg
    d = cfg.domain
    ax.set_xlim(d.x_min, d.x_max)
    if axis == "xz":
        ax.set_ylim(d.z_min, d.z_max)
        ax.set_ylabel("Z [Re]")
        ax.set_title("Meridional slice (Y=0)")
    else:
        ax.set_ylim(d.y_min, d.y_max)
        ax.set_ylabel("Y [Re]")
        ax.set_title("Equatorial slice (Z=0)")
    ax.set_xlabel("X [Re]    (Sun ←)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.4)

    _shade_levels(ax, sample, report, axis)

    xlim = ax.get_xlim()
    ylim = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
    xmp, ymp = _model_curve_user(shue_mp, cfg.solar_wind, xlim, ylim,
                                 dipole_strength=cfg.dipole_strength)
    xbs, ybs = _model_curve_user(jelinek_bs, cfg.solar_wind, xlim, ylim,
                                 dipole_strength=cfg.dipole_strength)
    ax.plot(xmp, ymp, color="#b30000", lw=1.6, label="Magnetopause (Shue 1998)")
    ax.plot(xbs, ybs, color="#08306b", lw=1.6, label="Bow shock (Jelínek 2012)")

    ax.add_patch(Circle((0, 0), 1.0, color="k", zorder=5))
    ax.plot(0, 0, "wo", ms=2.5, zorder=6)
    ax.annotate("Earth", (0, 0), textcoords="offset points", xytext=(6, 6),
                fontsize=8, color="k")

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
    ax.text(0.98, 0.98, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="right", fontsize=8,
            family="monospace",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.6"))

    ax.legend(loc="lower left", fontsize=8)


def make_figure(report: LoadReport, sample: RegionSample,
                out_path: str = "outputs/load_estimate.png"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    _draw_panel(axes[0], "xz", sample, report)
    _draw_panel(axes[1], "xy", sample, report)
    names = ", ".join(L.name for L in report.cfg.levels)
    fig.suptitle(f"PHARE global magnetosphere — levels: {names}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
