import numpy as np

from phare_load.constants import (
    NOMINAL_SW, L0_DX_KM, L1_DX_KM, L2_DX_KM, L3_DX_KM, STEPS_PER_L3,
)
from phare_load.models import subsolar_mp, subsolar_bs, shue_mp, jelinek_bs
from phare_load.geometry import sample_shell
from phare_load.load import uniform_reference, amr_load


def test_shue_subsolar_nominal():
    r = subsolar_mp(NOMINAL_SW)
    assert 9.5 < r < 11.5


def test_jelinek_subsolar_nominal():
    r = subsolar_bs(NOMINAL_SW)
    assert 13.0 < r < 15.5


def test_bs_outside_mp_everywhere():
    theta = np.linspace(0, np.deg2rad(125), 50)
    assert np.all(jelinek_bs(theta) > shue_mp(theta))


def test_resolutions():
    assert L0_DX_KM == 40.0
    assert L1_DX_KM == 40.0
    assert L2_DX_KM == 20.0
    assert L3_DX_KM == 10.0


def test_subcycling_ratios():
    assert STEPS_PER_L3["L0"] == 1.0 / 64.0
    assert STEPS_PER_L3["L1"] == 1.0 / 16.0
    assert STEPS_PER_L3["L2"] == 1.0 / 4.0
    assert STEPS_PER_L3["L3"] == 1.0


def test_shell_volumes_monotone():
    s = sample_shell(sample_dx_Re=1.0)
    assert s.volume_L1_Re3 > s.volume_L2_Re3 > s.volume_L3_Re3 > 0


def test_nested_masks():
    s = sample_shell(sample_dx_Re=1.0)
    assert np.all(s.in_L3 <= s.in_L2)
    assert np.all(s.in_L2 <= s.in_L1)


def test_uniform_reference_baseline():
    L = uniform_reference(10.0)
    assert 1e16 < L.n_particles < 3e16
    assert L.is_pic


def test_l0_has_no_particles():
    L0, L1, L2, L3, total, _ = amr_load(sample_dx_Re=1.0)
    assert L0.n_particles == 0
    assert not L0.is_pic
    for L in (L1, L2, L3):
        assert L.is_pic and L.n_particles > 0
    assert total.n_particles == L1.n_particles + L2.n_particles + L3.n_particles


def test_amr_cheaper_than_uniform_reference():
    ref = uniform_reference(10.0)
    L0, L1, L2, L3, _, _ = amr_load(sample_dx_Re=1.0)
    N = 1
    pic_work = N * (
        L1.steps_per_L3 * L1.n_particles
        + L2.steps_per_L3 * L2.n_particles
        + L3.steps_per_L3 * L3.n_particles
    )
    ref_work = N * ref.n_particles
    assert ref_work / pic_work > 30
