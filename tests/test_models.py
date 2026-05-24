import numpy as np
from pathlib import Path

from phare_load.config import (
    Config, LevelSpec, Patch, SolarWind, load_config,
)
from phare_load.models import subsolar_mp, subsolar_bs, shue_mp, jelinek_bs
from phare_load.geometry import sample_regions
from phare_load.load import build_report


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.toml"


def _default_cfg() -> Config:
    return load_config(CONFIG_PATH)


def test_default_config_loads():
    cfg = _default_cfg()
    assert cfg.delta_i_km == 100.0
    assert cfg.dt_ratio_per_level == 4.0
    assert len(cfg.levels) == 5
    assert cfg.levels[0].kind == "mhd"
    assert cfg.levels[-1].name == "L4"


def test_steps_per_finest():
    cfg = _default_cfg()
    # finest level fires every base step; each coarser level every 4 of the next finer
    assert cfg.steps_per_finest("L4") == 1.0
    assert cfg.steps_per_finest("L3") == 0.25
    assert cfg.steps_per_finest("L2") == 1.0 / 16.0
    assert cfg.steps_per_finest("L1") == 1.0 / 64.0
    assert cfg.steps_per_finest("L0") == 1.0 / 256.0


def test_shue_subsolar_nominal():
    r = subsolar_mp(SolarWind())
    assert 9.5 < r < 11.5


def test_jelinek_subsolar_nominal():
    r = subsolar_bs(SolarWind())
    assert 13.0 < r < 15.5


def test_bs_outside_mp_everywhere():
    sw = SolarWind()
    theta = np.linspace(0, np.deg2rad(125), 50)
    assert np.all(jelinek_bs(theta, sw) > shue_mp(theta, sw))


def test_nested_masks_default():
    cfg = _default_cfg()
    cfg.sample_dx_re = 1.0
    s = sample_regions(cfg)
    m1, m2, m3, m4 = s.masks["L1"], s.masks["L2"], s.masks["L3"], s.masks["L4"]
    assert np.all(m4 <= m3)
    assert np.all(m3 <= m2)
    assert np.all(m2 <= m1)
    assert (s.volumes_Re3["L1"] > s.volumes_Re3["L2"]
            > s.volumes_Re3["L3"] > s.volumes_Re3["L4"] > 0)


def test_mhd_has_no_particles():
    cfg = _default_cfg()
    cfg.sample_dx_re = 1.0
    report, _ = build_report(cfg)
    L0 = report.by_name("L0")
    assert not L0.is_pic and L0.n_particles == 0


def test_amr_cheaper_than_reference():
    cfg = _default_cfg()
    cfg.sample_dx_re = 1.0
    report, _ = build_report(cfg)
    pic_work = sum(L.steps_per_finest * L.n_particles for L in report.pic_levels)
    ref_work = report.uniform_reference.n_particles
    assert ref_work / pic_work > 30


def test_two_level_config_via_dataclass():
    cfg = Config(reference_dx_km=10.0)
    cfg.levels = [
        LevelSpec(name="L0", kind="mhd", dx_km=40.0, region="full"),
        LevelSpec(name="L1", kind="pic", dx_km=20.0, region="shell", pad_re=2.0),
    ]
    cfg.sample_dx_re = 1.0
    report, _ = build_report(cfg)
    assert len(report.levels) == 2
    assert report.by_name("L1").is_pic


def test_dx_di_scales_with_delta_i():
    cfg = Config(reference_dx_di=0.2)
    cfg.delta_i_km = 50.0
    cfg.levels = [
        LevelSpec(name="L1", kind="pic", dx_di=0.4, region="full"),
    ]
    cfg.sample_dx_re = 1.0
    report, _ = build_report(cfg)
    # 0.4 * 50 = 20 km
    assert report.by_name("L1").dx_km == 20.0
    # reference 0.2 * 50 = 10 km
    assert report.uniform_reference.dx_km == 10.0


def _patch_cfg(**kwargs) -> Config:
    """Two-level config with L1 = full-MP band, L2 = patched MP band.

    Uses dayside_only=True so the L1 baseline is well-defined (one
    hemisphere); otherwise the MP-band extends far down the tail because
    Shue diverges as θ → 180°. L2 is nested under L1 by construction."""
    cfg = Config(reference_dx_di=0.1, dayside_only=True)
    cfg.sample_dx_re = 0.5
    cfg.levels = [
        LevelSpec(name="L1", kind="pic", dx_di=0.2,
                  region="band", band_re=1.0, boundaries=["mp"]),
        LevelSpec(name="L2", kind="pic", dx_di=0.1,
                  region="band", band_re=1.0, boundaries=["mp"],
                  **kwargs),
    ]
    return cfg


def test_subsolar_cap_shrinks_to_expected_fraction():
    full = _patch_cfg()
    capped = _patch_cfg(theta_max_deg=30.0)
    v_full = sample_regions(full).volumes_Re3["L2"]
    v_cap = sample_regions(capped).volumes_Re3["L2"]
    # On the dayside hemisphere (θ ∈ [0, 90°]), the θ ≤ 30° spherical cap
    # covers a fraction (1 - cos 30°) of the area. The band thickness is the
    # same, so the volume ratio should match.
    expected = 1 - np.cos(np.deg2rad(30.0))
    assert 0.5 * expected < v_cap / v_full < 2.0 * expected


def test_phi_arcs_union_matches_arc_fraction():
    # Two 60° arcs cover 120/360 = 1/3 of the φ ring.
    full = _patch_cfg()
    arcs = _patch_cfg(phi_ranges_deg=[(60.0, 120.0), (240.0, 300.0)])
    v_full = sample_regions(full).volumes_Re3["L2"]
    v_arcs = sample_regions(arcs).volumes_Re3["L2"]
    assert 0.2 < v_arcs / v_full < 0.5


def test_multi_patch_union_geq_each_patch():
    cap = _patch_cfg(patches=[Patch(theta_max_deg=30.0)])
    flank = _patch_cfg(patches=[Patch(theta_min_deg=60.0, theta_max_deg=120.0,
                                      phi_ranges_deg=[(60.0, 120.0),
                                                      (240.0, 300.0)])])
    both = _patch_cfg(patches=[
        Patch(theta_max_deg=30.0),
        Patch(theta_min_deg=60.0, theta_max_deg=120.0,
              phi_ranges_deg=[(60.0, 120.0), (240.0, 300.0)]),
    ])
    v_cap = sample_regions(cap).volumes_Re3["L2"]
    v_flank = sample_regions(flank).volumes_Re3["L2"]
    v_both = sample_regions(both).volumes_Re3["L2"]
    # Patches are disjoint, so the union volume equals the sum (within
    # sampling discretization).
    assert v_both >= v_cap and v_both >= v_flank
    assert abs(v_both - (v_cap + v_flank)) / (v_cap + v_flank) < 0.05


def test_patches_and_top_level_are_exclusive():
    import pytest
    from phare_load.load import build_report
    cfg = _patch_cfg(theta_max_deg=30.0,
                     patches=[Patch(theta_max_deg=60.0)])
    with pytest.raises(ValueError):
        build_report(cfg)


def test_default_config_l4_has_subsolar_and_flank_patches():
    cfg = _default_cfg()
    L4 = cfg.levels[-1]
    assert L4.name == "L4"
    patches = L4.resolved_patches()
    # Subsolar reconnection cap + dawn/dusk KH flanks = 2 patches.
    assert len(patches) == 2
    assert patches[0].theta_max_deg == 30.0
    assert patches[1].phi_ranges_deg == [(60.0, 120.0), (240.0, 300.0)]


def test_dx_underspecified_rejected():
    import pytest
    cfg = Config(reference_dx_km=10.0)
    cfg.levels = [LevelSpec(name="L1", kind="pic", region="full")]
    with pytest.raises(ValueError):
        build_report(cfg)
