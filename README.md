# PHARE load estimator

Estimate memory and CPU load for a global magnetosphere PIC run with PHARE,
comparing an AMR hierarchy (any number of levels) to a uniform reference run.

## Domain (user convention)

`-x` points toward the Sun. Default domain: `x ∈ [-30, +150] Re`,
`y, z ∈ [-30, +30] Re`. Internally the Shue (MP) and Jelínek (BS) models are
evaluated in standard GSE (`+x` sunward); coordinates are mirrored on the
plot.

## Configuration

Everything is driven by a TOML config file (see `config.toml`). You can:

- add, remove, or rename levels
- change resolutions (in km, the `delta_i_km` field sets the ionic unit)
- pick each level's region: `"full"` (whole domain), `"shell"` with
  `pad_re` (sheath ±pad), or `"band"` with `band_re` (band around MP and BS)
- toggle dayside-only PIC
- change the solar-wind parameters and the domain box
- change the run target, particle cost model, and the reference uniform dx

Run with the bundled config:

```bash
.venv/bin/python -m phare_load.cli
```

Or with your own:

```bash
.venv/bin/python -m phare_load.cli --config my_setup.toml --out my_plot.png
```

## Interactive web app (no install)

A browser version of the estimator (Streamlit + Pyodide, via
[stlite](https://stlite.net)) lives in `docs/` and is served by GitHub Pages
at:

> **https://pharehub.github.io/pharculator/**

It runs entirely client-side — Python executes in your browser via
WebAssembly, no server is needed. First load is ~30 MB (numpy, scipy,
scikit-image, plotly, Streamlit runtime), cached for subsequent visits.

To enable / re-enable GitHub Pages on this repository: Settings → Pages →
Source = `Deploy from a branch`, Branch = `main`, Folder = `/docs`.

After changing anything in `src/phare_load/`, sync the copy used by the web
app:

```bash
bash docs/sync_source.sh
```

To preview the web app locally without deploying:

```bash
cd docs && python -m http.server 8000
# open http://localhost:8000/
```

## 3D visualization (optional)

Install the extra deps once:

```bash
.venv/bin/pip install -e ".[viz3d]"
```

Then add `--plot3d` to produce a standalone interactive HTML alongside the
2D figure:

```bash
.venv/bin/python -m phare_load.cli --plot3d
```

This writes `outputs/load_estimate_3d.html` — open it in any browser to
rotate the nested PIC shells, the Shue magnetopause, and the Jelínek bow
shock around Earth. The default view is a `Y < 0` cutaway so the nested
L1→Lₙ onion is exposed; pass `--no-cutaway` to render full surfaces.

For static slide-friendly PNGs, append one or more camera presets
(`front`, `oblique`, `tail`, `top`):

```bash
.venv/bin/python -m phare_load.cli --plot3d \
  --plot3d-png oblique --plot3d-png front
```

Each PNG is saved next to the HTML as `load_estimate_3d_<preset>.png`.

## Tests

```bash
.venv/bin/pytest
```

## Models

- Magnetopause: Shue et al. 1998.
- Bow shock: Jelínek et al. 2012 (paraboloid form).

The MP/BS surfaces flare unphysically deep into the tail; the dayside-only
flag is on by default to keep PIC where the fits are reliable.

## Subcycling

Each coarser AMR level uses `dt_ratio_per_level` × the dt of the next finer
level (default 4, i.e. `dt ∝ dx²`). The finest level shares its dt with the
uniform reference, so timestep counts are directly comparable.
