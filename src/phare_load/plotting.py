"""Meridional + equatorial plots of the AMR domain.

User axis convention: -x toward the Sun. Plots use the user frame.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from .constants import SolarWind, NOMINAL_SW
from .geometry import DOMAIN_USER, ShellSample
from .models import shue_mp, jelinek_bs, THETA_MAX
from .load import LoadReport


def _model_curve_user(model_fn, sw, ylim, n=400):
    """Return (x_user, y_in_plane) for the model curve in the user frame.

    The model is rotationally symmetric around the GSE x-axis, so the same
    curve serves both meridional (X-Z) and equatorial (X-Y) slices.
    The curve is truncated where |y| exceeds the plot half-extent (ylim).
    """
    theta = np.linspace(0.0, THETA_MAX, n)
    r = model_fn(theta, sw)
    x_gse = r * np.cos(theta)
    y = r * np.sin(theta)
    x_user = -x_gse
    # Truncate at the first index where |y| exceeds the plot extent.
    keep = np.where(y <= ylim)[0]
    if keep.size:
        kmax = keep[-1] + 1
        x_user = x_user[:kmax]
        y = y[:kmax]
    # Insert NaN between the +y and -y halves so the mirror join is not drawn.
    nan = np.array([np.nan])
    x_full = np.concatenate([x_user, nan, x_user])
    y_full = np.concatenate([y, nan, -y])
    return x_full, y_full


def _shaded_shell(ax, shell: ShellSample, axis: str):
    """Shade the shell on a 2D slice using the sampled mask near the plane."""
    if axis == "xz":
        # slice at y ~ 0
        iy = np.argmin(np.abs(shell.y[0, :, 0]))
        mask = shell.in_shell[:, iy, :]
        X = shell.x_user[:, iy, :]
        V = shell.z[:, iy, :]
    elif axis == "xy":
        iz = np.argmin(np.abs(shell.z[0, 0, :]))
        mask = shell.in_shell[:, :, iz]
        X = shell.x_user[:, :, iz]
        V = shell.y[:, :, iz]
    else:
        raise ValueError(axis)

    ax.contourf(
        X, V, mask.astype(float),
        levels=[0.5, 1.5],
        colors=["#74a9cf"],
        alpha=0.55,
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

    # Shaded L0 region from the sampled shell
    _shaded_shell(ax, shell, axis)

    # Magnetopause and bow shock curves (clip to in-panel extent)
    ylim = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
    xmp, ymp = _model_curve_user(shue_mp, sw, ylim)
    xbs, ybs = _model_curve_user(jelinek_bs, sw, ylim)
    ax.plot(xmp, ymp, color="#b30000", lw=1.6, label="Magnetopause (Shue 1998)")
    ax.plot(xbs, ybs, color="#08519c", lw=1.6, label="Bow shock (Jelínek 2012)")

    # Earth: filled circle of radius 1 Re at origin (user frame, Earth at 0,0,0)
    ax.add_patch(Circle((0, 0), 1.0, color="k", zorder=5))
    ax.plot(0, 0, "wo", ms=2.5, zorder=6)
    ax.annotate("Earth", (0, 0), textcoords="offset points", xytext=(6, 6),
                fontsize=8, color="k")

    # Illustrative L1 / L2 patches near the subsolar magnetosheath.
    # The actual particle counts come from the volume fractions; this is
    # only a visual hint of where high-refinement regions might live.
    subsolar_mp = float(shue_mp(0.0, sw))
    subsolar_bs = float(jelinek_bs(0.0, sw))
    mid = -0.5 * (subsolar_mp + subsolar_bs)   # in user frame (sun is at -x)
    half = 0.5 * (subsolar_bs - subsolar_mp)
    # L1 patch: wider band
    from matplotlib.patches import Rectangle
    l1_w = 2 * half
    l1_h = 6.0
    ax.add_patch(Rectangle((mid - l1_w / 2, -l1_h / 2), l1_w, l1_h,
                           facecolor="#2b8cbe", alpha=0.55, edgecolor="none",
                           zorder=3))
    # L2 patch: narrower, inside L1
    l2_w = l1_w * 0.6
    l2_h = 3.0
    ax.add_patch(Rectangle((mid - l2_w / 2, -l2_h / 2), l2_w, l2_h,
                           facecolor="#08306b", alpha=0.7, edgecolor="none",
                           zorder=4))

    # Particle-count annotations (top-left of each panel)
    L0, L1, L2 = report.amr_levels
    txt = (
        f"L0 (40 km, shell):   N = {L0.n_particles:.2e}\n"
        f"L1 (20 km, 10% V):   N = {L1.n_particles:.2e}\n"
        f"L2 (10 km, 5% V):    N = {L2.n_particles:.2e}\n"
        f"Σ AMR:               N = {report.amr_total.n_particles:.2e}\n"
        f"Uniform 100 km full: N = {report.uniform.n_particles:.2e}"
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
