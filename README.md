# PHARE load estimator

Estimates memory and CPU load for a global magnetosphere PIC run with PHARE,
comparing a uniform 100 km Cartesian baseline to an AMR scenario where only a
shell around the magnetopause/bow shock is treated kinetically.

## Domain (user convention)

- `-x` points toward the Sun.
- `x ∈ [-30 Re, +150 Re]` (Sun side at -30 Re, tail at +150 Re).
- `y, z ∈ [-30 Re, +30 Re]`.

Internally, the Shue (magnetopause) and Jelínek (bow shock) models are
evaluated in standard GSE (`+x` sunward); coordinates are mirrored on the
plot.

## Models

- **Magnetopause:** Shue et al. 1998.
- **Bow shock:** Jelínek et al. 2012 (paraboloid form).
- Nominal solar wind: n = 5 cm⁻³, V = 400 km/s, Bz = 0 nT → Pdyn ≈ 1.94 nPa.

The models are reliable on the dayside and flanks. Deep on the night side
they extrapolate; we clamp `θ` to `θ_max = 130°` for the volume integration
of the shell. This is a known approximation.

## AMR scenario

- **L0**: 40 km, fills the shell from 2 Re upstream of the bow shock to 2 Re
  earthward of the magnetopause. Outside L0 the code is MHD — no particles.
- **L1**: 20 km, on 10% of L0 volume.
- **L2**: 10 km, on 50% of L1 volume.
- 100 particles per cell on every level.

## Usage

```bash
pip install -e .
python -m phare_load.cli
```

Outputs:
- A comparison table on stdout.
- `outputs/load_estimate.png` with meridional + equatorial slices.

## Tests

```bash
pytest
```
