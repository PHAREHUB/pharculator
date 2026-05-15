import numpy as np

from phare_load.constants import NOMINAL_SW
from phare_load.models import subsolar_mp, subsolar_bs, shue_mp, jelinek_bs
from phare_load.geometry import sample_shell
from phare_load.load import uniform_load, amr_load


def test_shue_subsolar_nominal():
    r = subsolar_mp(NOMINAL_SW)
    assert 9.5 < r < 11.5, f"Shue subsolar MP out of range: {r}"


def test_jelinek_subsolar_nominal():
    r = subsolar_bs(NOMINAL_SW)
    assert 13.0 < r < 15.5, f"Jelinek subsolar BS out of range: {r}"


def test_bs_outside_mp_everywhere():
    theta = np.linspace(0, np.deg2rad(125), 50)
    assert np.all(jelinek_bs(theta) > shue_mp(theta))


def test_shell_volume_nonzero():
    s = sample_shell(sample_dx_Re=1.0)
    assert s.volume_Re3 > 0


def test_uniform_load_baseline():
    L = uniform_load(10.0)
    # 10 km uniform: ~1.67e16 particles, ~9e17 bytes
    assert 1e16 < L.n_particles < 3e16
    assert 5e17 < L.ram_bytes < 2e18


def test_amr_vs_equivalent_uniform():
    uni10 = uniform_load(10.0)
    _, total, _ = amr_load(sample_dx_Re=1.0)
    assert total.n_particles < uni10.n_particles
    assert uni10.n_particles / total.n_particles > 50


def test_l1_l2_volume_fractions():
    _, _, shell = amr_load(sample_dx_Re=1.0)
    # L1 ~ 10% of L0, L2 ~ 5% of L0 (i.e. 50% of L1). Allow tolerance for
    # discretization (sampling at 1 Re gives stepwise targets).
    f1 = shell.volume_L1_Re3 / shell.volume_Re3
    f2 = shell.volume_L2_Re3 / shell.volume_L1_Re3
    assert 0.08 < f1 < 0.12, f1
    assert 0.45 < f2 < 0.55, f2
