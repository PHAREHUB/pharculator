"""Meridional + equatorial plots of the AMR domain.

User axis convention: -x toward the Sun. Plots use the user frame.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from .constants import SolarWind, NOMINAL_SW
from .geometry import DOMAIN_USER, ShellSample
from .models import shue_mp, jelinek_bs
from .load import LoadReport


# Maximum theta used when drawing the model curves. Just below pi so the
# curves keep flaring until they exit the plot, but the formula stays finite.
_THETA_CURVE_MAX = np.deg2rad(178.0)


def _model_curve_user(model_fn, sw, xlim, ylim, n=2000):
    """Return (x_user, y_in_plane) for the model curve in the user frame.

    Rotationally symmetric around the GSE x-axis, so the same curve serves
    both meridional (X-Z) and equatorial (X-Y) slices. The curve is
    truncated as soon as it would leave the plot box (xlim is (xmin, xmax),
    ylim is the +y half-extent).
    """
    theta = np.linspace(0.0, _THETA_CURVE_MAX, n)
    r = model_fn(theta, sw)
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


def _slice_arrays(shell: ShellSample, axis: str):
    if axis == "xz":
        iy = np.argmin(np.abs(shell.y[0, :, 0]))
        X = shell.x_user[:, iy, :]
        V = shell.z[:, iy, :]
        m0 = shell.in_shell[:, iy, :]
        m1 = shell.in_L1[:, iy, :]
        m2 = shell.in_L2[:, iy, :]
    elif axis == "xy":
        iz = np.argmin(np.abs(shell.z[0, 0, :]))
        X = shell.x_user[:, :, iz]
        V = shell.y[:, :, iz]
        m0 = shell.in_shell[:, :, iz]
        m1 = shell.in_L1[:, :, iz]
        m2 = shell.in_L2[:, :, iz]
    else:
        raise ValueError(axis)
    return X, V, m0, m1, m2


def _shaded_levels(ax, shell: ShellSample, axis: str):
    """Shade L0, L1, L2 with progressively darker fill."""
    X, V, m0, m1, m2 = _slice_arrays(shell, axis)
    # Encode level index: 0 outside, 1 = L0\L1, 2 = L1\L2, 3 = L2
    lvl = np.zeros_like(m0, dtype=int)
    lvl[m0] = 1
    lvl[m1] = 2
    lvl[m2] = 3
    cmap_colors = ["#c6dbef", "#6baed6", "#08519c"]  # L0, L1, L2
    ax.contourf(
        X, V, lvl.astype(float),
        levels=[0.5, 1.5, 2.5, 3.5],
        colors=cmap_colors,
        alpha=0.75,
    )


def _draw_panel(ax, axis: str, shell: ShellSample, report: LoadReport, sw: SolarWind):
    d = DOMAIN_USER
    # Domain box
    ax.set_xlim(d["x_min"], d["x_max"])
    if axis == "xz":
        ax.set_ylim(d["z_min"], d["z_max"])
        ax.set_ylabel("Z [Re]")
        ax.set_title("Meridional slice (Y=0)")
    else:
        ax.set_ylim(d["y_min"], d["y_max"])
        ax.set_ylabel("Y [Re]")
        ax.set_title("Equatorial slice (Z=0)")
    ax.set_xlabel("X [Re]    (Sun ←)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.4)

    # Shaded L0/L1/L2 from the sampled masks
    _shaded_levels(ax, shell, axis)

    # Magnetopause and bow shock curves
    xlim = ax.get_xlim()
    ylim = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
    xmp, ymp = _model_curve_user(shue_mp, sw, xlim, ylim)
    xbs, ybs = _model_curve_user(jelinek_bs, sw, xlim, ylim)
    ax.plot(xmp, ymp, color="#b30000", lw=1.6, label="Magnetopause (Shue 1998)")
    ax.plot(xbs, ybs, color="#08306b", lw=1.6, label="Bow shock (Jelínek 2012)")

    # Earth: filled circle of radius 1 Re at origin
    ax.add_patch(Circle((0, 0), 1.0, color="k", zorder=5))
    ax.plot(0, 0, "wo", ms=2.5, zorder=6)
    ax.annotate("Earth", (0, 0), textcoords="offset points", xytext=(6, 6),
                fontsize=8, color="k")

    # Particle-count annotations
    L0, L1, L2 = report.amr_levels
    txt = (
        f"L0 (40 km, sheath shell): N = {L0.n_particles:.2e}\n"
        f"L1 (20 km, MP+BS bands ~10%V): N = {L1.n_particles:.2e}\n"
        f"L2 (10 km, ~50% of L1):   N = {L2.n_particles:.2e}\n"
        f"Σ AMR:                    N = {report.amr_total.n_particles:.2e}\n"
        f"Uniform {report.uniform.dx_km:.0f} km full:    N = {report.uniform.n_particles:.2e}"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes,
            va="top", ha="left", fontsize=8,
            family="monospace",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.6"))

    ax.legend(loc="lower right", fontsize=8)


def make_figure(report: LoadReport, shell: ShellSample, sw: SolarWind = NOMINAL_SW,
                out_path: str = "outputs/load_estimate.png"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    _draw_panel(axes[0], "xz", shell, report, sw)
    _draw_panel(axes[1], "xy", shell, report, sw)
    fig.suptitle("PHARE global magnetosphere — uniform vs AMR particle load",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
