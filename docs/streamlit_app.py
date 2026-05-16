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

    with st.expander("HPC sizing (Alice Recoque)", expanded=False):
        st.caption("Memory-bound dispatch on France's upcoming exascale "
                   "machine. All numbers editable.")
        st.markdown("**GPU partition** (Venice + 4× MI430X)")
        hbm_per_gpu_gb = st.number_input("HBM per GPU [GB]",
                                         min_value=16.0, max_value=2000.0,
                                         value=432.0, step=16.0)
        gpus_per_node = st.number_input("GPUs per node",
                                        min_value=1, max_value=16,
                                        value=4, step=1)
        gpu_throughput_g = st.number_input(
            "Push throughput per GPU [G pushes/s]",
            min_value=0.1, max_value=500.0, value=30.0, step=1.0,
            help="Bandwidth-bound estimate. MI430X HBM4 ≈ 5 TB/s; "
                 "~80 B/push including halo → ~50-70 G/s peak, 30 effective.")
        gpu_power_kw = st.number_input("Power per GPU node [kW]",
                                       min_value=0.1, max_value=20.0,
                                       value=3.4, step=0.1)
        total_gpu_nodes = st.number_input(
            "Total GPU nodes in machine",
            min_value=1, max_value=100_000,
            value=3500, step=100,
            help="NOT officially published. Default extrapolated from the "
                 "announced ~1 EFlop FP64 target: at ~4 × 120 TF (4× MI430X) "
                 "≈ 480 TF/node and ~50–70 % efficiency to reach sustained "
                 "1 EFlop, the GPU partition lands in the 2 000–4 000 nodes "
                 "range. Edit this value if GENCI / EuroHPC publishes the "
                 "actual count.")

        st.markdown("**CPU partition** (SiPEARL Rhea2)")
        cores_per_cpu_node = st.number_input("Cores per CPU node",
                                             min_value=1, max_value=2048,
                                             value=128, step=8)
        cpu_ram_gb = st.number_input("RAM per CPU node [GB]",
                                     min_value=8.0, max_value=8192.0,
                                     value=1024.0, step=64.0)
        cpu_power_kw = st.number_input("Power per CPU node [kW]",
                                       min_value=0.05, max_value=5.0,
                                       value=0.6, step=0.05)
        total_cpu_nodes = st.number_input(
            "Total CPU nodes in machine",
            min_value=1, max_value=200_000,
            value=5000, step=100,
            help="NOT officially published. Even less constrained than the "
                 "GPU partition: announcements only say it is the smaller "
                 "partition (~5–15 % of total system FLOPs). 5 000 is an "
                 "order-of-magnitude default — adjust as soon as the real "
                 "spec is known.")

        st.markdown("**Energy**")
        pue = st.number_input("PUE",
                              min_value=1.0, max_value=2.5,
                              value=1.3, step=0.05)
        carbon_g_per_kwh = st.number_input(
            "Carbon intensity [gCO₂/kWh]",
            min_value=0.0, max_value=1000.0, value=50.0, step=10.0,
            help="France grid ≈ 50, EU average ≈ 250, coal ≈ 800.")

    with st.expander("HPC sizing (Adastra — CINES)", expanded=False):
        st.caption("Existing French/EuroHPC system at CINES, in production "
                   "since 2022. Three partitions; specs from "
                   "[dci-gitlab.cines.fr](https://dci.dci-gitlab.cines.fr/"
                   "webextranet/architecture/index.html).")
        st.markdown("**MI250X partition** (356 nodes)")
        ad_mi250_hbm_gb = st.number_input(
            "HBM per GCD [GB]", min_value=1.0, max_value=2000.0,
            value=64.0, step=1.0,
            help="MI250X has 2 GCDs of 64 GB HBM2e each; ROCm exposes them "
                 "as 2 logical GPUs.")
        ad_mi250_gcds_per_node = st.number_input(
            "GCDs per node", min_value=1, max_value=16,
            value=8, step=1, help="4× MI250X cards × 2 GCDs = 8 per node.")
        ad_mi250_throughput_g = st.number_input(
            "Push throughput per GCD [G pushes/s]",
            min_value=0.1, max_value=500.0, value=20.0, step=1.0,
            help="HBM2e ≈ 1.6 TB/s per GCD; bandwidth-bound ≈ 20 G/s "
                 "effective for PIC pushes.")
        ad_mi250_power_kw = st.number_input(
            "Power per MI250X node [kW]",
            min_value=0.1, max_value=10.0, value=2.67, step=0.1,
            help="From CINES docs.")
        ad_mi250_nodes = st.number_input(
            "Total MI250X nodes", min_value=1, max_value=10_000,
            value=356, step=1)

        st.markdown("**MI300A partition** (28 nodes)")
        ad_mi300_hbm_gb = st.number_input(
            "HBM per APU [GB]", min_value=1.0, max_value=2000.0,
            value=128.0, step=1.0, help="512 GB shared between 4 APUs.")
        ad_mi300_apus_per_node = st.number_input(
            "APUs per node", min_value=1, max_value=16,
            value=4, step=1)
        ad_mi300_throughput_g = st.number_input(
            "Push throughput per APU [G pushes/s]",
            min_value=0.1, max_value=500.0, value=60.0, step=1.0,
            help="HBM3 ≈ 5 TB/s; ~60 G/s effective.")
        ad_mi300_power_kw = st.number_input(
            "Power per MI300A node [kW]",
            min_value=0.1, max_value=10.0, value=3.5, step=0.1,
            help="Estimate — not in public docs.")
        ad_mi300_nodes = st.number_input(
            "Total MI300A nodes", min_value=1, max_value=10_000,
            value=28, step=1)

        st.markdown("**Genoa CPU partition** (544 nodes)")
        ad_genoa_cores = st.number_input(
            "Cores per Genoa node", min_value=1, max_value=2048,
            value=192, step=8, help="2× EPYC 9654 = 192 c.")
        ad_genoa_ram_gb = st.number_input(
            "RAM per Genoa node [GB]", min_value=8.0, max_value=8192.0,
            value=768.0, step=64.0)
        ad_genoa_power_kw = st.number_input(
            "Power per Genoa node [kW]",
            min_value=0.05, max_value=5.0, value=0.945, step=0.05)
        ad_genoa_nodes = st.number_input(
            "Total Genoa nodes", min_value=1, max_value=10_000,
            value=544, step=1)


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


def _sci(x: float) -> str:
    """Scientific notation as a string (avoids JS integer overflow in the
    frontend for values beyond 2**53)."""
    return f"{x:.2e}" if x else "—"


rows = []
for L in report.levels:
    n_steps = N * L.steps_per_finest
    pushes = n_steps * L.n_particles if L.is_pic else 0.0
    rows.append({
        "level": L.name,
        "kind": L.kind.upper(),
        "dx [km]": L.dx_km,
        "N_cells": _sci(L.n_cells),
        "N_part":  _sci(L.n_particles),
        "RAM [TB]": L.ram_bytes / 2**40,
        "steps":   _sci(n_steps),
        "pushes":  _sci(pushes),
        "% of AMR per-step work": (
            100.0 * L.steps_per_finest * L.n_particles
            / (pic_work / N) if pic_work > 0 else 0.0),
    })

st.dataframe(
    rows,
    hide_index=True,
    column_config={
        "dx [km]":  st.column_config.NumberColumn(format="%.1f"),
        "RAM [TB]": st.column_config.NumberColumn(format="%.2f"),
        "% of AMR per-step work": st.column_config.NumberColumn(format="%.2f"),
        # N_cells / N_part / steps / pushes are pre-formatted strings, rendered
        # as TextColumn — avoids the JS Number.MAX_SAFE_INTEGER overflow that
        # NumberColumn hits beyond ~9e15.
    },
)

# ----- Comparison with uniform models --------------------------------------
st.subheader("Comparison with uniform models")
st.caption(
    "Aggregates the AMR hierarchy vs each uniform-PIC reference run over "
    "the same physical duration. Each uniform reference uses its own "
    "CFL-appropriate dt (∝ dx²), so the comparison is physically honest.")

pic_levels = [L for L in report.levels if L.is_pic]
amr_n_cells = sum(L.n_cells for L in pic_levels)
amr_n_part  = sum(L.n_particles for L in pic_levels)
amr_steps_total = sum(N * L.steps_per_finest for L in report.levels)
amr_pushes = pic_work
amr_cpuh   = amr_sec / 3600

comp_rows = [{
    "model":    "AMR Σ",
    "dx [km]":  float("nan"),  # multi-resolution
    "N_cells":  _sci(amr_n_cells),
    "N_part":   _sci(amr_n_part),
    "RAM [TB]": amr_ram / 2**40,
    "steps":    _sci(amr_steps_total),
    "pushes":   _sci(amr_pushes),
    "CPU·h":    _sci(amr_cpuh),
    "vs AMR":   1.0,
}]

for ref in report.uniform_references:
    ref_steps  = N * ref.steps_per_finest
    ref_pushes = ref_steps * ref.n_particles
    ref_cpuh   = ref_pushes * cfg.sec_per_particle_per_step / 3600
    comp_rows.append({
        "model":    f"uniform {ref.dx_km:.1f} km",
        "dx [km]":  ref.dx_km,
        "N_cells":  _sci(ref.n_cells),
        "N_part":   _sci(ref.n_particles),
        "RAM [TB]": ref.ram_bytes / 2**40,
        "steps":    _sci(ref_steps),
        "pushes":   _sci(ref_pushes),
        "CPU·h":    _sci(ref_cpuh),
        "vs AMR":   ref_cpuh / amr_cpuh if amr_cpuh > 0 else float("nan"),
    })

st.dataframe(
    comp_rows,
    hide_index=True,
    column_config={
        "dx [km]":  st.column_config.NumberColumn(format="%.1f"),
        "RAM [TB]": st.column_config.NumberColumn(format="%.2f"),
        "vs AMR":   st.column_config.NumberColumn(
            format="%.2f ×",
            help="CPU·h relative to AMR. >1 = uniform is more expensive than AMR."),
    },
)


# ----- HPC dispatch on Alice Recoque ---------------------------------------
st.subheader("HPC dispatch on Alice Recoque")
st.caption(
    "Memory-bound sizing for each cost model on both partitions. "
    "Each GPU node has 4× MI430X (HBM) + 1× Venice CPU; each CPU node is "
    "a SiPEARL Rhea2. The “× Alice Recoque” column expresses the required "
    "node count as a fraction of the machine — if the run doesn't fit, "
    "it tells you **how many Alice Recoques** you would actually need.")


def _fmt_walltime(seconds: float) -> str:
    if seconds <= 0 or not (seconds == seconds):  # 0 or NaN
        return "—"
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{seconds/60:.1f} min"
    if seconds < 86400:
        return f"{seconds/3600:.1f} h"
    if seconds < 7 * 86400:
        return f"{seconds/86400:.1f} d"
    if seconds < 365 * 86400:
        return f"{seconds/(7*86400):.1f} wk"
    return f"{seconds/(365*86400):.2g} yr"


def _fmt_machine_fraction(nodes: float, total: float) -> str:
    if total <= 0:
        return "—"
    frac = nodes / total
    if frac < 1.0:
        return f"{frac:.2f} ×"
    if frac < 100.0:
        return f"{frac:.1f} × full machine"
    return f"{frac:.2e} × full machine"


def _dispatch(total_ram: float, total_pushes: float,
              ram_per_node_bytes: float, pushes_per_s_per_node: float,
              power_kw: float, total_nodes_partition: float) -> dict:
    """Compute memory-bound node count, wall-time, energy, CO₂ for one
    (model × partition) cell."""
    import math
    nodes = math.ceil(total_ram / ram_per_node_bytes) if ram_per_node_bytes > 0 else float("inf")
    wall_s = (total_pushes / (nodes * pushes_per_s_per_node)
              if (nodes > 0 and pushes_per_s_per_node > 0) else float("nan"))
    energy_kwh = nodes * power_kw * (wall_s / 3600.0) * pue
    co2_t = energy_kwh * carbon_g_per_kwh / 1e6
    return {
        "nodes_mem": nodes,
        "wall_time_s": wall_s,
        "energy_kwh": energy_kwh,
        "co2_t": co2_t,
        "machine_frac": _fmt_machine_fraction(nodes, total_nodes_partition),
    }


# Build (model, total_ram, total_pushes) tuples once.
models = [("AMR Σ", amr_ram, amr_pushes)]
for ref in report.uniform_references:
    ref_steps  = N * ref.steps_per_finest
    ref_pushes = ref_steps * ref.n_particles
    models.append((f"uniform {ref.dx_km:.1f} km", ref.ram_bytes, ref_pushes))

gpu_ram_per_node = hbm_per_gpu_gb * gpus_per_node * 1e9  # bytes
gpu_pps_per_node = gpu_throughput_g * gpus_per_node * 1e9
cpu_ram_per_node = cpu_ram_gb * 1e9
cpu_pps_per_node = (1.0 / cfg.sec_per_particle_per_step) * cores_per_cpu_node

# Baseline reference for the "vs AMR-GPU" energy ratio.
baseline = _dispatch(amr_ram, amr_pushes,
                     gpu_ram_per_node, gpu_pps_per_node,
                     gpu_power_kw, total_gpu_nodes)
baseline_kwh = baseline["energy_kwh"] or 1.0

hpc_rows = []
for model_name, ram_b, pushes in models:
    for partition_name, specs in (
        ("GPU", (gpu_ram_per_node, gpu_pps_per_node, gpu_power_kw,
                 total_gpu_nodes)),
        ("CPU", (cpu_ram_per_node, cpu_pps_per_node, cpu_power_kw,
                 total_cpu_nodes)),
    ):
        d = _dispatch(ram_b, pushes, *specs)
        hpc_rows.append({
            "model": model_name,
            "partition": partition_name,
            "nodes (memory)": _sci(d["nodes_mem"]),
            "× Alice Recoque": d["machine_frac"],
            "wall-time": _fmt_walltime(d["wall_time_s"]),
            "energy [MWh]": d["energy_kwh"] / 1000.0,
            "tCO₂": d["co2_t"],
            "vs AMR-GPU (energy)": (d["energy_kwh"] / baseline_kwh
                                    if baseline_kwh else float("nan")),
        })

st.dataframe(
    hpc_rows,
    hide_index=True,
    column_config={
        "energy [MWh]": st.column_config.NumberColumn(format="%.2f"),
        "tCO₂":         st.column_config.NumberColumn(format="%.3f"),
        "vs AMR-GPU (energy)": st.column_config.NumberColumn(
            format="%.2f ×",
            help="Energy ratio relative to running the AMR hierarchy on the "
                 "GPU partition (the natural baseline). >1 means more expensive."),
    },
)

st.caption(
    "Caveats: memory-bound sizing only (perf might require more nodes if "
    "load balance is poor); throughput per GPU/core is a first-order "
    "estimate; linear scaling assumed (real efficiency at >10⁴ nodes is "
    "~50–70 %). Edit the defaults in the sidebar **HPC sizing (Alice Recoque)** "
    "expander to refine.")


# ----- HPC dispatch on Adastra ---------------------------------------------
st.subheader("HPC dispatch on Adastra (CINES)")
st.caption(
    "Memory-bound sizing on the in-production CINES system (3 partitions: "
    "MI250X, MI300A, Genoa). All values editable in the sidebar.")

ad_mi250_ram_per_node = ad_mi250_hbm_gb * ad_mi250_gcds_per_node * 1e9
ad_mi250_pps_per_node = ad_mi250_throughput_g * ad_mi250_gcds_per_node * 1e9
ad_mi300_ram_per_node = ad_mi300_hbm_gb * ad_mi300_apus_per_node * 1e9
ad_mi300_pps_per_node = ad_mi300_throughput_g * ad_mi300_apus_per_node * 1e9
ad_genoa_ram_per_node = ad_genoa_ram_gb * 1e9
ad_genoa_pps_per_node = (1.0 / cfg.sec_per_particle_per_step) * ad_genoa_cores

# Baseline = AMR on MI250X (the largest Adastra partition) for the energy
# ratio column.
ad_baseline = _dispatch(amr_ram, amr_pushes,
                        ad_mi250_ram_per_node, ad_mi250_pps_per_node,
                        ad_mi250_power_kw, ad_mi250_nodes)
ad_baseline_kwh = ad_baseline["energy_kwh"] or 1.0

ad_rows = []
for model_name, ram_b, pushes in models:
    for partition_name, specs in (
        ("MI250X", (ad_mi250_ram_per_node, ad_mi250_pps_per_node,
                    ad_mi250_power_kw, ad_mi250_nodes)),
        ("MI300A", (ad_mi300_ram_per_node, ad_mi300_pps_per_node,
                    ad_mi300_power_kw, ad_mi300_nodes)),
        ("Genoa",  (ad_genoa_ram_per_node, ad_genoa_pps_per_node,
                    ad_genoa_power_kw, ad_genoa_nodes)),
    ):
        d = _dispatch(ram_b, pushes, *specs)
        ad_rows.append({
            "model": model_name,
            "partition": partition_name,
            "nodes (memory)": _sci(d["nodes_mem"]),
            "× Adastra partition": d["machine_frac"],
            "wall-time": _fmt_walltime(d["wall_time_s"]),
            "energy [MWh]": d["energy_kwh"] / 1000.0,
            "tCO₂": d["co2_t"],
            "vs AMR-MI250X (energy)": (d["energy_kwh"] / ad_baseline_kwh
                                       if ad_baseline_kwh else float("nan")),
        })

st.dataframe(
    ad_rows,
    hide_index=True,
    column_config={
        "energy [MWh]": st.column_config.NumberColumn(format="%.2f"),
        "tCO₂":         st.column_config.NumberColumn(format="%.3f"),
        "vs AMR-MI250X (energy)": st.column_config.NumberColumn(
            format="%.2f ×",
            help="Energy ratio relative to running the AMR hierarchy on "
                 "the MI250X partition (Adastra's flagship, 356 nodes)."),
    },
)

st.caption(
    "Adastra peak ≈ 79 PFlop FP64 (68 PF MI250X + 7 PF MI300A + 4 PF Genoa). "
    "Same caveats as the Alice Recoque table (memory-bound, linear scaling, "
    "rough per-GPU throughput estimates).")

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
