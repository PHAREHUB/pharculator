"""3D rendering of the AMR PIC level regions, magnetopause, and bow shock.

Produces a standalone interactive Plotly HTML and optional static PNGs from
named camera presets. The default view is a Y<0 cutaway so the nested
L1 -> L4 shells are visible as half-onions hugging the magnetopause.

Coordinates are standard GSE (+x toward Sun). This module renders in raw GSE
without flipping the axis (unlike plotting.py, which flips x for the 2D
slices) because rotating an interactive 3D scene is the natural way to
explore orientation.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np

from .config import Config
from .geometry import RegionSample
from .load import LoadReport
from .models import shue_mp, jelinek_bs


_BLUES = [
    "#9ecae1",  # L1 (coarsest PIC) - light blue
    "#6baed6",
    "#3182bd",
    "#08519c",  # finest - dark
    "#08306b",
]


def _level_color(i: int, n: int) -> str:
    if n <= 1:
        return _BLUES[-2]
    idx = int(round(i * (len(_BLUES) - 1) / max(n - 1, 1)))
    return _BLUES[min(idx, len(_BLUES) - 1)]


def _mesh_for_mask(mask: np.ndarray, sample: RegionSample,
                   cutaway_y: bool = True):
    """Extract a triangulated surface for a boolean 3D mask.

    Returns (vertices_Nx3, faces_Mx3) in GSE coordinates [Re], or None if the
    mask is empty after the cutaway.
    """
    from skimage import measure

    work = mask.copy()
    if cutaway_y:
        # Hide cells with Y >= 0 so the dayside half-onion is exposed.
        work = work & (sample.y < 0)
    if not work.any():
        return None

    dx = sample.dV_Re3 ** (1.0 / 3.0)
    # Pad with False so marching cubes closes the surface at the mask edge.
    padded = np.pad(work, 1, mode="constant", constant_values=False)
    verts, faces, _, _ = measure.marching_cubes(
        padded.astype(np.float32), level=0.5, spacing=(dx, dx, dx),
    )
    # marching_cubes coords are in (i, j, k) order matching the padded array.
    # Origin of the un-padded array, in GSE Re:
    origin = np.array([sample.x[0, 0, 0],
                       sample.y[0, 0, 0],
                       sample.z[0, 0, 0]])
    # Undo the 1-cell pad shift, then translate to GSE origin.
    verts = verts - dx
    verts = verts + origin
    return verts, faces


def _model_surface(model_fn, sw, dipole_strength: float,
                   n_theta: int = 60, n_phi: int = 120,
                   theta_max_deg: float = 130.0,
                   cutaway_y: bool = True):
    """Build a parametric surface (theta, phi) for an axisymmetric boundary.

    Returns (X, Y, Z) 2D arrays for go.Surface.
    """
    theta = np.linspace(0.0, np.deg2rad(theta_max_deg), n_theta)
    # Phi runs over the full revolution; we set Y>=0 hemisphere to NaN so
    # Plotly draws only the visible half (cutaway). Phi=0 -> +Y, phi=pi -> -Y.
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    r = model_fn(TH, sw, dipole_strength=dipole_strength)
    X = r * np.cos(TH)
    Y = r * np.sin(TH) * np.sin(PH)
    Z = r * np.sin(TH) * np.cos(PH)
    if cutaway_y:
        mask = Y >= 0
        X = np.where(mask, np.nan, X)
        Y = np.where(mask, np.nan, Y)
        Z = np.where(mask, np.nan, Z)
    return X, Y, Z


def _earth_sphere(radius: float = 1.0, n: int = 24):
    u = np.linspace(0.0, 2.0 * np.pi, n)
    v = np.linspace(0.0, np.pi, n // 2)
    U, V = np.meshgrid(u, v, indexing="ij")
    X = radius * np.cos(U) * np.sin(V)
    Y = radius * np.sin(U) * np.sin(V)
    Z = radius * np.cos(V)
    return X, Y, Z


def _domain_box(cfg: Config):
    d = cfg.domain
    xs = [d.x_min, d.x_max]
    ys = [d.y_min, 0.0]  # only the visible (Y<=0) half of the box
    zs = [d.z_min, d.z_max]
    corners = np.array([(x, y, z) for x in xs for y in ys for z in zs])
    edges = []
    for i, a in enumerate(corners):
        for b in corners[i + 1:]:
            diff = np.abs(a - b) > 1e-9
            if diff.sum() == 1:
                edges.append((a, b))
    xs_line, ys_line, zs_line = [], [], []
    for a, b in edges:
        xs_line += [a[0], b[0], None]
        ys_line += [a[1], b[1], None]
        zs_line += [a[2], b[2], None]
    return xs_line, ys_line, zs_line


_CAMERA_PRESETS = {
    "front":   dict(eye=dict(x=1.8,  y=1.2, z=0.6)),
    "oblique": dict(eye=dict(x=1.4,  y=1.6, z=1.0)),
    "tail":    dict(eye=dict(x=-1.6, y=1.4, z=0.8)),
    "top":     dict(eye=dict(x=0.0,  y=0.05, z=2.2)),
}


def build_figure_3d(report: LoadReport, sample: RegionSample,
                    cutaway_y: bool = True,
                    camera_preset: str = "oblique"):
    """Build the 3D Plotly Figure (PIC shells + MP + BS + Earth) and return it.

    The caller is responsible for writing HTML / PNG. Used by the CLI and by
    the Streamlit web app.
    """
    import plotly.graph_objects as go

    cfg = report.cfg
    fig = go.Figure()

    pic_specs = [L for L in cfg.levels if L.kind == "pic"]
    n = len(pic_specs)
    for i, spec in enumerate(pic_specs):
        mesh = _mesh_for_mask(sample.masks[spec.name], sample,
                              cutaway_y=cutaway_y)
        if mesh is None:
            continue
        verts, faces = mesh
        color = _level_color(i, n)
        # Opacity grows toward finest so inner shells stay readable when
        # outer shells are also drawn.
        opacity = 0.30 + 0.55 * (i / max(n - 1, 1))
        fig.add_trace(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color, opacity=opacity, name=spec.name,
            showlegend=True, flatshading=True, hoverinfo="name",
        ))

    # MP and BS surfaces.
    for fn, label, color, op in (
        (shue_mp,    "Magnetopause (Shue 1998)",  "#b30000", 0.20),
        (jelinek_bs, "Bow shock (Jelinek 2012)",  "#08306b", 0.15),
    ):
        X, Y, Z = _model_surface(fn, cfg.solar_wind, cfg.dipole_strength,
                                 cutaway_y=cutaway_y)
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            showscale=False, opacity=op,
            colorscale=[[0, color], [1, color]],
            name=label, showlegend=True, hoverinfo="name",
            contours=dict(x=dict(highlight=False),
                          y=dict(highlight=False),
                          z=dict(highlight=False)),
        ))

    # Earth.
    EX, EY, EZ = _earth_sphere()
    fig.add_trace(go.Surface(
        x=EX, y=EY, z=EZ,
        showscale=False, opacity=1.0,
        colorscale=[[0, "#202020"], [1, "#202020"]],
        name="Earth", showlegend=True, hoverinfo="name",
    ))

    # Domain bounding box (visible half).
    bx, by, bz = _domain_box(cfg)
    fig.add_trace(go.Scatter3d(
        x=bx, y=by, z=bz, mode="lines",
        line=dict(color="rgba(120,120,120,0.5)", width=2),
        name="Domain", showlegend=True, hoverinfo="name",
    ))

    names = ", ".join(L.name for L in cfg.levels)
    fig.update_layout(
        title=f"PHARE global magnetosphere — levels: {names}"
              f"  ({'Y<0 cutaway, ' if cutaway_y else ''}Sun at +X)",
        scene=dict(
            xaxis_title="X_GSE [Re]  (Sun = +X)",
            yaxis_title="Y_GSE [Re]",
            zaxis_title="Z_GSE [Re]",
            aspectmode="data",
            camera=_CAMERA_PRESETS.get(camera_preset,
                                       _CAMERA_PRESETS["oblique"]),
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def make_figure_3d(report: LoadReport, sample: RegionSample,
                   html_path: str = "outputs/load_estimate_3d.html",
                   png_paths: Iterable[tuple[str, str]] | None = None,
                   cutaway_y: bool = True) -> str:
    """CLI helper: build the figure, write HTML (and optional PNGs), return html_path."""
    fig = build_figure_3d(report, sample, cutaway_y=cutaway_y)

    os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)

    if png_paths:
        try:
            for preset, path in png_paths:
                cam = _CAMERA_PRESETS.get(preset)
                if cam is None:
                    print(f"  [plot3d] unknown camera preset '{preset}', "
                          f"using oblique")
                    cam = _CAMERA_PRESETS["oblique"]
                fig.update_layout(scene_camera=cam)
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                fig.write_image(path, width=1400, height=1000, scale=1)
        except Exception as e:
            print(f"  [plot3d] PNG export failed ({e}); HTML still written. "
                  f"Install kaleido or screenshot from the HTML.")

    return html_path
