"""Streamlit / stlite front-end for the PHARE load estimator.

Runs unchanged on:
- a regular Streamlit server (`streamlit run docs/streamlit_app.py`)
- stlite (in the browser via Pyodide), served from GitHub Pages.

The phare_load package source is mounted alongside this file (see index.html).
"""

from __future__ import annotations

import io
import sys
import traceback
from dataclasses import asdict

import streamlit as st

from phare_load.config import (
    Config, LevelSpec, Domain, SolarWind,
)
from phare_load.load import build_report
from phare_load.models import subsolar_mp, subsolar_bs
from phare_load.plotting2d import build_figure_2d
from phare_load.plotting3d import build_figure_3d


st.set_page_config(
    page_title="PHARE load estimator",
    page_icon=None,
    layout="wide",
)

st.title("PHARE load estimator — interactive")
st.caption(
    "Estimate memory / CPU·h of a global magnetosphere PIC run with PHARE "
    "AMR vs uniform-PIC baselines. Source: "
    "[github.com/PHAREHUB/pharculator](https://github.com/PHAREHUB/pharculator). "
    "Full model: [MODEL.md](https://github.com/PHAREHUB/pharculator/blob/main/MODEL.md)."
)


# ---------------------------------------------------------------------------
# Default level templates
# ---------------------------------------------------------------------------

DEFAULT_LEVELS = [
    {"name": "L0", "kind": "mhd", "dx_di": 1.6,
     "region": "full",  "pad_re": 0.0, "band_re": 0.0,
     "boundaries": "mp,bs"},
    {"name": "L1", "kind": "pic", "dx_di": 0.8,
     "region": "shell", "pad_re": 3.0, "band_re": 0.0,
     "boundaries": "mp,bs"},
    {"name": "L2", "kind": "pic", "dx_di": 0.4,
     "region": "band",  "pad_re": 0.0, "band_re": 1.5,
     "boundaries": "mp"},
    {"name": "L3", "kind": "pic", "dx_di": 0.2,
     "region": "band",  "pad_re": 0.0, "band_re": 0.5,
     "boundaries": "mp"},
    {"name": "L4", "kind": "pic", "dx_di": 0.1,
     "region": "band",  "pad_re": 0.0, "band_re": 0.25,
     "boundaries": "mp"},
]


# ---------------------------------------------------------------------------
# Sidebar — global parameters
# ---------------------------------------------------------------------------

# Proton inertial length <-> density relation (cold plasma, protons only):
#   δᵢ [km] = c / ω_pi,  ω_pi = sqrt(n e² / (ε₀ m_p))
#   →  δᵢ [km] = 227.7 / sqrt(n [cm⁻³])
#   →  n [cm⁻³] = (227.7 / δᵢ [km])²
DI_N_CONST_KM = 227.7  # δᵢ × √n  for protons (km × cm^{-3/2})


def _di_to_n(di_km: float) -> float:
    return (DI_N_CONST_KM / di_km) ** 2


def _n_to_di(n_cm3: float) -> float:
    return DI_N_CONST_KM / (n_cm3 ** 0.5)


# Initialise the linked widgets in session_state so the bidirectional
# coupling has well-defined starting values.
if "delta_i_km" not in st.session_state:
    st.session_state.delta_i_km = 100.0
if "sw_n" not in st.session_state:
    # Keep the historical default (n = 5 cm⁻³) on first load even though it
    # corresponds to δᵢ ≈ 102 km, not exactly 100. The coupling only fires on
    # user-initiated changes, not on the initial render.
    st.session_state.sw_n = 5.0


def _on_delta_i_change():
    st.session_state.sw_n = _di_to_n(st.session_state.delta_i_km)


def _on_n_change():
    st.session_state.delta_i_km = _n_to_di(st.session_state.sw_n)


with st.sidebar:
    st.header("Physical scales")
    delta_i_km = st.number_input(
        "δᵢ (ion inertial length) [km]",
        min_value=1.0, max_value=10_000.0, step=10.0,
        key="delta_i_km", on_change=_on_delta_i_change,
        help="Coupled to solar-wind n (protons only): "
             "δᵢ = 227.7 / √n. Editing either side updates the other.")
    omega_ci_inverse_s = st.number_input("1/Ω_ci [s]",
                                         min_value=0.01, max_value=10.0,
                                         value=1.0, step=0.1)
    target_hours = st.number_input("Run duration [h]",
                                   min_value=0.01, max_value=24.0,
                                   value=1.0, step=0.1)
    dipole_strength = st.slider("dipole_strength (size knob)",
                                min_value=0.001, max_value=2.0, value=1.0,
                                step=0.05,
                                help="Scales the magnetosphere by "
                                     "dipole_strength^(1/3). 1.0 = Earth today.")

    st.header("Solar wind")
    sw_n = st.number_input(
        "n [cm⁻³]", min_value=0.001, max_value=1000.0, step=0.5,
        key="sw_n", on_change=_on_n_change,
        help="Coupled to δᵢ (see above).")
    sw_v = st.number_input("V [km/s]", min_value=200.0, max_value=1500.0,
                           value=400.0, step=10.0)
    sw_bz = st.number_input("Bz [nT]", min_value=-30.0, max_value=30.0,
                            value=0.0, step=0.5)

    st.header("Domain (Re, GSE)")
    col_a, col_b = st.columns(2)
    with col_a:
        x_min = st.number_input("x_min", value=-100.0, step=10.0)
        y_min = st.number_input("y_min", value=-50.0, step=5.0)
        z_min = st.number_input("z_min", value=-50.0, step=5.0)
    with col_b:
        x_max = st.number_input("x_max", value=50.0, step=10.0)
        y_max = st.number_input("y_max", value=50.0, step=5.0)
        z_max = st.number_input("z_max", value=50.0, step=5.0)

    st.header("AMR / time-stepping")
    dt_L1_omega_ci = st.number_input("dt_L1 [/Ω_ci]",
                                     min_value=0.001, max_value=1.0,
                                     value=0.05, step=0.01)
    dt_ratio_per_level = st.number_input("dt ratio per level",
                                         min_value=2.0, max_value=8.0,
                                         value=4.0, step=1.0)
    dayside_only = st.checkbox("dayside-only PIC", value=True)
    sample_dx_re = st.slider("probe lattice dx [Re]",
                             min_value=0.25, max_value=2.0, value=1.0,
                             step=0.25,
                             help="Coarser = faster but less accurate "
                                  "volumes. 0.5 = CLI default, 1.0 "
                                  "recommended in-browser.")

    st.header("Uniform references")
    refs_str = st.text_input(
        "reference_dx_di (comma-separated)",
        value="1.0, 0.1",
        help="One uniform baseline per dx. e.g. '1.0, 0.1' = coarse + fine.")

    st.header("Particle model")
    ppc = st.number_input("particles per cell", min_value=1, max_value=1000,
                          value=100, step=10)
    bytes_per_particle = st.number_input("bytes per particle",
                                         min_value=8, max_value=512,
                                         value=76, step=4)
    sec_per_push_ns = st.number_input("cost per push [ns]",
                                      min_value=1.0, max_value=1000.0,
                                      value=10.0, step=1.0)

    st.header("3D view")
    cutaway = st.checkbox("Y<0 cutaway", value=True,
                          help="Hide the Y≥0 half so nested shells are visible.")
    preset = st.selectbox("camera preset",
                          options=["oblique", "front", "tail", "top"],
                          index=0)
    show_2d = st.checkbox("Render 2D meridional figure", value=True,
                         help="Disable to skip matplotlib (faster live updates).")


# ---------------------------------------------------------------------------
# Main — level editor
# ---------------------------------------------------------------------------

st.subheader("AMR levels")
st.caption("Coarsest first. The last entry is the finest and anchors dt. "
           "PIC levels are nested by construction: each PIC mask is "
           "intersected with the previous PIC level's mask.")

if "levels_df" not in st.session_state:
    st.session_state.levels_df = list(DEFAULT_LEVELS)

# Streamlit's data_editor handles list-of-dicts directly.
edited_levels = st.data_editor(
    st.session_state.levels_df,
    num_rows="dynamic",
    column_config={
        "name": st.column_config.TextColumn("name", required=True),
        "kind": st.column_config.SelectboxColumn(
            "kind", options=["mhd", "pic"], required=True),
        "dx_di": st.column_config.NumberColumn(
            "dx_di [δᵢ]", min_value=0.01, max_value=10.0,
            step=0.05, format="%.3f"),
        "region": st.column_config.SelectboxColumn(
            "region", options=["full", "shell", "band"], required=True),
        "pad_re": st.column_config.NumberColumn(
            "pad_re [Re]", min_value=0.0, max_value=20.0, step=0.5),
        "band_re": st.column_config.NumberColumn(
            "band_re [Re]", min_value=0.0, max_value=20.0, step=0.25),
        "boundaries": st.column_config.TextColumn(
            "boundaries", help="comma-sep of 'mp','bs' (used by region='band')"),
    },
    key="levels_editor",
)
st.session_state.levels_df = edited_levels


# ---------------------------------------------------------------------------
# Build a Config from the UI state
# ---------------------------------------------------------------------------

def _parse_refs(s: str) -> list[float]:
    out = []
    for tok in s.replace(";", ",").split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    if not out:
        raise ValueError("At least one reference dx is required.")
    return out


def _row_to_level(row: dict) -> LevelSpec:
    bounds = [b.strip() for b in str(row.get("boundaries", "mp,bs")).split(",")
              if b.strip()]
    return LevelSpec(
        name=str(row["name"]),
        kind=row["kind"],
        dx_di=float(row["dx_di"]),
        region=row["region"],
        pad_re=float(row.get("pad_re") or 0.0),
        band_re=float(row.get("band_re") or 0.0),
        boundaries=bounds or ["mp", "bs"],
    )


def build_config() -> Config:
    cfg = Config(
        delta_i_km=delta_i_km,
        omega_ci_inverse_s=omega_ci_inverse_s,
        target_hours=target_hours,
        dipole_strength=dipole_strength,
        dt_L1_omega_ci=dt_L1_omega_ci,
        dt_ratio_per_level=dt_ratio_per_level,
        dayside_only=dayside_only,
        sample_dx_re=sample_dx_re,
        ppc=int(ppc),
        bytes_per_particle=int(bytes_per_particle),
        sec_per_particle_per_step=float(sec_per_push_ns) * 1e-9,
        reference_dx_di=_parse_refs(refs_str),
    )
    cfg.domain = Domain(x_min=x_min, x_max=x_max,
                        y_min=y_min, y_max=y_max,
                        z_min=z_min, z_max=z_max)
    cfg.solar_wind = SolarWind(n_cm3=sw_n, v_kms=sw_v, bz_nt=sw_bz)
    cfg.levels = [_row_to_level(r) for r in edited_levels
                  if r and r.get("name") and r.get("dx_di")]
    return cfg


# ---------------------------------------------------------------------------
# Live estimator (no Run button — the script reruns on every widget change)
# ---------------------------------------------------------------------------


@st.cache_data(max_entries=8, show_spinner=False)
def _cached_report(cfg_key: bytes):
    """Cached wrapper around build_report.

    `cfg_key` is a pickle of the Config; passing the pickle bytes makes
    Streamlit's cache key cheap and stable. Returns the (report, sample)
    pair so subsequent renders that only change display options (camera
    preset, cutaway flag) reuse the geometry computation.
    """
    import pickle
    cfg = pickle.loads(cfg_key)
    return build_report(cfg)


try:
    cfg = build_config()
except Exception as exc:
    st.error(f"Configuration error: {exc}")
    st.code(traceback.format_exc())
    st.stop()

with st.spinner("Computing…"):
    import pickle
    try:
        report, sample = _cached_report(pickle.dumps(cfg))
    except Exception as exc:
        st.error(f"Run failed: {exc}")
        st.code(traceback.format_exc())
        st.stop()

# ----- numeric results ------------------------------------------------------
st.subheader("Cost summary")
pic_work = sum(L.steps_per_finest * L.n_particles * report.n_steps_target
               for L in report.pic_levels)
amr_sec = pic_work * cfg.sec_per_particle_per_step
amr_ram = sum(L.ram_bytes for L in report.pic_levels)

cols = st.columns(1 + len(report.uniform_references))
cols[0].metric("AMR — CPU·h", f"{amr_sec/3600/1e6:,.2f} M",
               help=f"{amr_sec/3600:,.0f} CPU·h total")
cols[0].metric("AMR — RAM (PIC)", f"{amr_ram/2**50:,.2f} PB")
for i, ref in enumerate(report.uniform_references, start=1):
    ref_work = (report.n_steps_target * ref.steps_per_finest
                * ref.n_particles)
    ref_sec = ref_work * cfg.sec_per_particle_per_step
    cols[i].metric(f"Uniform {ref.dx_km:.1f} km — CPU·h",
                   f"{ref_sec/3600/1e6:,.2f} M",
                   delta=f"{ref_sec/amr_sec:.2f} × AMR")
    cols[i].metric(f"Uniform {ref.dx_km:.1f} km — RAM",
                   f"{ref.ram_bytes/2**50:,.2f} PB",
                   delta=f"{ref.ram_bytes/amr_ram:.2f} × AMR")

# ----- per-level table ------------------------------------------------------
st.subheader("Per-level breakdown")
N = report.n_steps_target
rows = []
for L in report.levels:
    n_steps = N * L.steps_per_finest
    pushes = n_steps * L.n_particles if L.is_pic else 0.0
    rows.append({
        "level": L.name,
        "kind": L.kind.upper(),
        "dx [km]": L.dx_km,
        "N_cells": L.n_cells,
        "N_part": L.n_particles,
        "RAM [TB]": L.ram_bytes / 2**40,
        "steps": n_steps,
        "pushes": pushes,
        "% of AMR per-step work": (
            100.0 * L.steps_per_finest * L.n_particles
            / (pic_work / N) if pic_work > 0 else 0.0),
    })

# Summary row: AMR totals across all PIC levels (MHD has zero particles).
pic_rows = [L for L in report.levels if L.is_pic]
if pic_rows:
    rows.append({
        "level": "AMR Σ",
        "kind": "TOTAL",
        "dx [km]": float("nan"),  # not aggregable
        "N_cells": sum(L.n_cells for L in pic_rows),
        "N_part":  sum(L.n_particles for L in pic_rows),
        "RAM [TB]": sum(L.ram_bytes for L in pic_rows) / 2**40,
        # Sum of per-level step counts across the hierarchy (total subcycled
        # advances, including coarse levels firing less often).
        "steps":   sum(N * L.steps_per_finest for L in report.levels),
        "pushes":  pic_work,
        "% of AMR per-step work": 100.0,
    })

st.dataframe(
    rows,
    hide_index=True,
    column_config={
        "dx [km]":  st.column_config.NumberColumn(format="%.1f"),
        "N_cells":  st.column_config.NumberColumn(format="%.2e"),
        "N_part":   st.column_config.NumberColumn(format="%.2e"),
        "RAM [TB]": st.column_config.NumberColumn(format="%.2f"),
        "steps":    st.column_config.NumberColumn(format="%.2e"),
        "pushes":   st.column_config.NumberColumn(format="%.2e"),
        "% of AMR per-step work": st.column_config.NumberColumn(format="%.2f"),
    },
)

# ----- subsolar reminders ---------------------------------------------------
ds = cfg.dipole_strength
st.caption(
    f"Subsolar MP = {subsolar_mp(cfg.solar_wind, ds):.2f} Re   |   "
    f"Subsolar BS = {subsolar_bs(cfg.solar_wind, ds):.2f} Re   |   "
    f"P_dyn = {cfg.solar_wind.Pdyn_nPa:.3f} nPa   |   "
    f"dipole_strength = {ds:.3g}")

# ----- 2D figure ------------------------------------------------------------
if show_2d:
    st.subheader("Meridional + dayside view")
    with st.spinner("Rendering 2D figure…"):
        fig2d = build_figure_2d(report, sample)
    st.plotly_chart(fig2d, use_container_width=True)

# ----- 3D figure ------------------------------------------------------------
st.subheader(f"3D view{' (Y<0 cutaway)' if cutaway else ''}")
with st.spinner("Rendering 3D figure…"):
    try:
        fig3d = build_figure_3d(report, sample,
                                cutaway_y=cutaway,
                                camera_preset=preset)
        st.plotly_chart(fig3d, use_container_width=True)
    except Exception as exc:
        st.warning(f"3D rendering failed: {exc}")
        st.code(traceback.format_exc())
