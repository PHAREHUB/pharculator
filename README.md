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
