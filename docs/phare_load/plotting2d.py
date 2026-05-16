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


def _dayside_zoom(cfg: Config):
    r_mp = subsolar_mp(cfg.solar_wind, dipole_strength=cfg.dipole_strength)
    r_bs = subsolar_bs(cfg.solar_wind, dipole_strength=cfg.dipole_strength)
    x_min = -max(8.0, 0.7 * r_mp)
    x_max = 1.6 * r_bs
    half = 1.5 * r_bs
    return (x_min, x_max), (-half, half)


def _discrete_colorscale(colors: list[str]):
    """Build a step-function Plotly colorscale from a list of colors."""
    n = len(colors)
    cs = []
    for i, c in enumerate(colors):
        cs.append([i / n, c])
        cs.append([(i + 1) / n, c])
    # Make sure the scale is strictly monotonic in [0, 1].
    cs[-1][0] = 1.0
    return cs


def _add_panel(fig, sample: RegionSample, report: LoadReport,
               col: int, xlim, ylim, show_legend: bool):
    import plotly.graph_objects as go

    cfg = report.cfg
    idx_y0 = _slice_index_y0(sample)

    # 1D coordinate arrays at the Y≈0 slice.
    xs = sample.x[:, idx_y0, 0]
    zs = sample.z[0, idx_y0, :]

    pic_specs = [L for L in cfg.levels if L.kind == "pic"]
    n_pic = len(pic_specs)

    # Build a per-cell level index in the slice (0 = outside, 1..n = PIC level).
    lvl = np.zeros((len(xs), len(zs)), dtype=int)
    for i, spec in enumerate(pic_specs, start=1):
        m = sample.masks[spec.name][:, idx_y0, :]
        lvl[m] = i

    colors = _LEVEL_COLORS[: n_pic + 1]
    fig.add_trace(go.Heatmap(
        x=xs, y=zs,
        z=lvl.T,  # heatmap expects (ny, nx)
        zmin=0, zmax=max(n_pic, 1),
        colorscale=_discrete_colorscale(colors),
        showscale=False,
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

    # Legend entries (custom level chips) — only on the left panel.
    if show_legend:
        for i, spec in enumerate(pic_specs, start=1):
            c = colors[i] if i < len(colors) else colors[-1]
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=12, color=c, symbol="square"),
                name=f"{spec.name} ({spec.kind})",
                hoverinfo="skip",
            ), row=1, col=col)


def build_figure_2d(report: LoadReport, sample: RegionSample):
    """Build the interactive 2D Plotly figure and return it."""
    from plotly.subplots import make_subplots

    cfg = report.cfg
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Meridional slice (Y≈0) — full domain",
                        "Dayside zoom (Y≈0)"),
        horizontal_spacing=0.10,
    )

    full_xlim = (cfg.domain.x_min, cfg.domain.x_max)
    full_ylim = (cfg.domain.z_min, cfg.domain.z_max)
    _add_panel(fig, sample, report, col=1,
               xlim=full_xlim, ylim=full_ylim, show_legend=True)

    day_xlim, day_ylim = _dayside_zoom(cfg)
    _add_panel(fig, sample, report, col=2,
               xlim=day_xlim, ylim=day_ylim, show_legend=False)

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
