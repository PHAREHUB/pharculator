"""Interactive 2D figure (Plotly) of the AMR levels, magnetopause, bow shock.

Same content as plotting.py's matplotlib figure but interactive (zoom/pan/hover)
and fast to update — used by the web app. The matplotlib version stays as the
CLI default for static PNG export.
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .geometry import RegionSample
from .load import LoadReport
from .models import shue_mp, jelinek_bs, subsolar_mp, subsolar_bs


_THETA_CURVE_MAX = np.deg2rad(178.0)

# Background + Blues palette: index 0 = "outside any PIC level",
# index 1..n = L1..Ln (coarsest to finest PIC).
_LEVEL_COLORS = [
    "#f0f4f8",  # background
    "#c6dbef",
    "#9ecae1",
    "#6baed6",
    "#3182bd",
    "#08519c",
    "#08306b",
]


def _model_curve(model_fn, sw, dipole_strength: float):
    """Return (x, y) for both lobes of an axisymmetric boundary surface."""
    theta = np.linspace(0.0, _THETA_CURVE_MAX, 2000)
    r = model_fn(theta, sw, dipole_strength=dipole_strength)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    nan = np.array([np.nan])
    return (np.concatenate([x, nan, x]),
            np.concatenate([y, nan, -y]))


def _slice_index_y0(sample: RegionSample) -> int:
    return int(np.argmin(np.abs(sample.y[0, :, 0])))


def _compute_y0_masks(cfg: Config, xs: np.ndarray, zs: np.ndarray):
    """Compute per-PIC-level boolean masks on the Y=0 plane at the given grid.

    Independent of cfg.sample_dx_re — used by the 2D plot to get smooth
    region boundaries without forcing the full 3D geometry sample to be
    expensively fine.
    """
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    r = np.sqrt(X * X + Z * Z)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_t = np.where(r > 0, X / r, 1.0)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    theta = np.arccos(cos_t)
    r_mp = shue_mp(theta, cfg.solar_wind, dipole_strength=cfg.dipole_strength)
    r_bs = jelinek_bs(theta, cfg.solar_wind,
                      dipole_strength=cfg.dipole_strength)

    masks: dict[str, np.ndarray] = {}
    prev_mask: np.ndarray | None = None
    for spec in cfg.levels:
        if spec.kind != "pic":
            continue
        if spec.region == "full":
            m = np.ones_like(theta, dtype=bool)
        elif spec.region == "shell":
            m = (r >= (r_mp - spec.pad_re)) & (r <= (r_bs + spec.pad_re))
        elif spec.region == "band":
            m = np.zeros_like(theta, dtype=bool)
            if "mp" in spec.boundaries:
                m = m | (np.abs(r - r_mp) <= spec.band_re)
            if "bs" in spec.boundaries:
                m = m | (np.abs(r - r_bs) <= spec.band_re)
        else:  # pragma: no cover
            raise AssertionError(spec.region)
        if cfg.dayside_only:
            m = m & (X >= 0)
        if prev_mask is not None:
            m = m & prev_mask
        masks[spec.name] = m
        prev_mask = m
    return masks


def _dayside_zoom(cfg: Config):
    r_mp = subsolar_mp(cfg.solar_wind, dipole_strength=cfg.dipole_strength)
    r_bs = subsolar_bs(cfg.solar_wind, dipole_strength=cfg.dipole_strength)
    x_min = -max(8.0, 0.7 * r_mp)
    x_max = 1.6 * r_bs
    half = 1.5 * r_bs
    return (x_min, x_max), (-half, half)


def _mask_to_svg_path(mask_xz: np.ndarray,
                       x_coords: np.ndarray,
                       z_coords: np.ndarray) -> str | None:
    """Return an SVG path tracing the boundary of `mask_xz`, in data coords.

    Uses scikit-image's marching-squares to extract sub-cell-precision
    contours, then concatenates them into a single path. Combined with
    Plotly's fillrule='evenodd', this fills the True region of the mask
    correctly even when there are holes (e.g. sheath shell, MP band).

    Returns None if the mask is empty.
    """
    from skimage import measure

    if not mask_xz.any():
        return None
    # find_contours returns (row, col) sub-pixel coordinates of the iso-0.5
    # boundary on a float array. Pad with False to ensure boundaries don't run
    # off the grid (otherwise they'd open and break the fill).
    padded = np.pad(mask_xz.astype(float), 1, mode="constant",
                    constant_values=0.0)
    contours = measure.find_contours(padded, 0.5)
    if not contours:
        return None

    nx = len(x_coords)
    nz = len(z_coords)
    segments: list[str] = []
    for c in contours:
        # Undo the 1-cell pad and convert index → data coordinate.
        ix = c[:, 0] - 1.0
        iz = c[:, 1] - 1.0
        x_real = np.interp(ix, np.arange(nx), x_coords)
        z_real = np.interp(iz, np.arange(nz), z_coords)
        head = f"M {x_real[0]:.3f},{z_real[0]:.3f}"
        rest = " ".join(f"L {x:.3f},{z:.3f}"
                        for x, z in zip(x_real[1:], z_real[1:]))
        segments.append(f"{head} {rest} Z")
    return " ".join(segments)


def _add_panel(fig, report: LoadReport,
               col: int, xlim, ylim, show_legend: bool,
               panel_dx_re: float):
    import plotly.graph_objects as go

    cfg = report.cfg

    # Build a fresh, fine 2D grid covering this panel's view box. This is
    # independent of cfg.sample_dx_re (which controls the 3D volume integration)
    # — cheap because the slice is only 2D.
    xs = np.arange(min(xlim), max(xlim) + panel_dx_re, panel_dx_re)
    zs = np.arange(min(ylim), max(ylim) + panel_dx_re, panel_dx_re)
    masks = _compute_y0_masks(cfg, xs, zs)

    pic_specs = [L for L in cfg.levels if L.kind == "pic"]
    n_pic = len(pic_specs)
    colors = _LEVEL_COLORS[: n_pic + 1]

    # Background fill for the domain box.
    fig.add_shape(type="rect",
                  x0=min(xlim), x1=max(xlim),
                  y0=min(ylim), y1=max(ylim),
                  fillcolor=colors[0], line=dict(width=0),
                  layer="below",
                  row=1, col=col)

    # Per-PIC-level filled polygons via marching squares + evenodd fill.
    # Drawing from coarsest to finest stacks them visually as nested shells.
    for i, spec in enumerate(pic_specs, start=1):
        path = _mask_to_svg_path(masks[spec.name], xs, zs)
        if path is None:
            continue
        fig.add_shape(
            type="path", path=path,
            fillrule="evenodd",
            fillcolor=colors[i] if i < len(colors) else colors[-1],
            line=dict(color=colors[i] if i < len(colors) else colors[-1],
                      width=0.5),
            opacity=0.85,
            layer="below",
            row=1, col=col,
        )

    # Legend chips for each PIC level (left panel only).
    if show_legend:
        for i, spec in enumerate(pic_specs, start=1):
            c = colors[i] if i < len(colors) else colors[-1]
            dx_km = spec.resolve_dx_km(cfg.delta_i_km)
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=12, color=c, symbol="square"),
                name=f"{spec.name} (PIC, dx = {dx_km:.1f} km)",
                hoverinfo="skip",
            ), row=1, col=col)

    # Magnetopause and bow shock.
    x_mp, y_mp = _model_curve(shue_mp, cfg.solar_wind, cfg.dipole_strength)
    x_bs, y_bs = _model_curve(jelinek_bs, cfg.solar_wind, cfg.dipole_strength)
    fig.add_trace(go.Scatter(
        x=x_mp, y=y_mp, mode="lines",
        line=dict(color="#b30000", width=2),
        name="Magnetopause (Shue 1998)",
        showlegend=show_legend,
        hoverinfo="skip",
    ), row=1, col=col)
    fig.add_trace(go.Scatter(
        x=x_bs, y=y_bs, mode="lines",
        line=dict(color="#08306b", width=2),
        name="Bow shock (Jelinek 2012)",
        showlegend=show_legend,
        hoverinfo="skip",
    ), row=1, col=col)

    # Earth as a filled black disk in data coordinates.
    fig.add_shape(
        type="circle",
        x0=-1, x1=1, y0=-1, y1=1,
        fillcolor="black",
        line=dict(color="black"),
        row=1, col=col,
    )

    # Reverse the x-axis so the Sun (+x) sits on the LEFT (magnetospheric convention).
    fig.update_xaxes(range=[max(xlim), min(xlim)],
                     title_text="X<sub>GSE</sub> [Re]  (Sun ←)",
                     row=1, col=col)
    fig.update_yaxes(range=list(ylim),
                     scaleanchor=f"x{col}" if col > 1 else "x",
                     scaleratio=1.0,
                     title_text="Z [Re]",
                     row=1, col=col)


def build_figure_2d(report: LoadReport, sample: RegionSample | None = None,
                    full_dx_re: float = 0.3,
                    zoom_dx_re: float = 0.05):
    """Build the interactive 2D Plotly figure and return it.

    `sample` is accepted for API compatibility but ignored — the 2D plot
    samples its own fine Y=0 slice via `_compute_y0_masks`. `full_dx_re`
    controls the resolution of the full-domain panel; `zoom_dx_re` the
    resolution of the dayside zoom panel (typically several × finer).
    """
    from plotly.subplots import make_subplots
    _ = sample  # kept in signature for back-compat with older callers

    cfg = report.cfg
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Meridional slice (Y≈0) — full domain",
                        "Dayside zoom (Y≈0)"),
        horizontal_spacing=0.10,
    )

    full_xlim = (cfg.domain.x_min, cfg.domain.x_max)
    full_ylim = (cfg.domain.z_min, cfg.domain.z_max)
    _add_panel(fig, report, col=1,
               xlim=full_xlim, ylim=full_ylim, show_legend=True,
               panel_dx_re=full_dx_re)

    day_xlim, day_ylim = _dayside_zoom(cfg)
    _add_panel(fig, report, col=2,
               xlim=day_xlim, ylim=day_ylim, show_legend=False,
               panel_dx_re=zoom_dx_re)

    names = ", ".join(L.name for L in cfg.levels)
    fig.update_layout(
        title=f"PHARE global magnetosphere — levels: {names}",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                    xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=60, b=10),
        plot_bgcolor="white",
    )
    return fig
