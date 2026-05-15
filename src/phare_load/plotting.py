"""Meridional + equatorial plots of the AMR levels.

User axis convention: -x toward the Sun. Plots use the user frame.
L0 fills the whole panel as a very light tint (MHD base level). L1 (sheath
shell) and L2 (boundary bands) are progressively darker.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from .constants import SolarWind, NOMINAL_SW
from .geometry import DOMAIN_USER, ShellSample
from .models import shue_mp, jelinek_bs
from .load import LoadReport


_THETA_CURVE_MAX = np.deg2rad(178.0)


def _model_curve_user(model_fn, sw, xlim, ylim, n=2000):
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


def _slice_masks(shell: ShellSample, axis: str):
    if axis == "xz":
        iy = np.argmin(np.abs(shell.y[0, :, 0]))
        X = shell.x_user[:, iy, :]
        V = shell.z[:, iy, :]
        m1 = shell.in_L1[:, iy, :]
        m2 = shell.in_L2[:, iy, :]
        m3 = shell.in_L3[:, iy, :]
    elif axis == "xy":
        iz = np.argmin(np.abs(shell.z[0, 0, :]))
        X = shell.x_user[:, :, iz]
        V = shell.y[:, :, iz]
        m1 = shell.in_L1[:, :, iz]
        m2 = shell.in_L2[:, :, iz]
        m3 = shell.in_L3[:, :, iz]
    else:
        raise ValueError(axis)
    return X, V, m1, m2, m3


def _shade_levels(ax, shell: ShellSample, axis: str):
    d = DOMAIN_USER
    ax.add_patch(Rectangle(
        (d["x_min"], -30 if axis == "xz" else d["y_min"]),
        d["x_max"] - d["x_min"],
        60.0,
        facecolor="#f0f4f8", edgecolor="none", zorder=0,
    ))

    X, V, m1, m2, m3 = _slice_masks(shell, axis)
    lvl = np.zeros_like(m1, dtype=int)
    lvl[m1] = 1
    lvl[m2] = 2
    lvl[m3] = 3
    ax.contourf(
        X, V, lvl.astype(float),
        levels=[0.5, 1.5, 2.5, 3.5],
        colors=["#c6dbef", "#4292c6", "#08306b"],   # L1, L2, L3
        alpha=0.8,
        zorder=1,
    )


def _draw_panel(ax, axis: str, shell: ShellSample, report: LoadReport, sw: SolarWind):
    d = DOMAIN_USER
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

    _shade_levels(ax, shell, axis)

    xlim = ax.get_xlim()
    ylim = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
    xmp, ymp = _model_curve_user(shue_mp, sw, xlim, ylim)
    xbs, ybs = _model_curve_user(jelinek_bs, sw, xlim, ylim)
    ax.plot(xmp, ymp, color="#b30000", lw=1.6, label="Magnetopause (Shue 1998)")
    ax.plot(xbs, ybs, color="#08306b", lw=1.6, label="Bow shock (Jelínek 2012)")

    ax.add_patch(Circle((0, 0), 1.0, color="k", zorder=5))
    ax.plot(0, 0, "wo", ms=2.5, zorder=6)
    ax.annotate("Earth", (0, 0), textcoords="offset points", xytext=(6, 6),
                fontsize=8, color="k")

    L0, L1, L2, L3 = report.L0, report.L1, report.L2, report.L3
    ref = report.uniform_reference
    txt = (
        f"L0 ({L0.dx_km:.0f} km, MHD, full box):    no particles\n"
        f"L1 ({L1.dx_km:.0f} km, sheath ±3 Re):     N = {L1.n_particles:.2e}\n"
        f"L2 ({L2.dx_km:.0f} km, 1.5 Re of MP+BS):  N = {L2.n_particles:.2e}\n"
        f"L3 ({L3.dx_km:.0f} km, 0.5 Re of MP+BS):  N = {L3.n_particles:.2e}\n"
        f"Σ AMR PIC:                          N = {report.amr_total.n_particles:.2e}\n"
        f"Uniform {ref.dx_km:.0f} km PIC (full box):    N = {ref.n_particles:.2e}"
    )
    # Annotation lives on the nightside (right half of the user frame) so
    # it does not cover the MP / BS curves on the dayside.
    ax.text(0.98, 0.98, txt, transform=ax.transAxes,
            va="top", ha="right", fontsize=8,
            family="monospace",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.6"))

    ax.legend(loc="lower left", fontsize=8)


def make_figure(report: LoadReport, shell: ShellSample, sw: SolarWind = NOMINAL_SW,
                out_path: str = "outputs/load_estimate.png"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    _draw_panel(axes[0], "xz", shell, report, sw)
    _draw_panel(axes[1], "xy", shell, report, sw)
    fig.suptitle("PHARE global magnetosphere — MHD (L0) + PIC AMR (L1, L2, L3)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
