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
    L = uniform_load(100.0)
    # Order of magnitude: ~1.67e13 particles, ~9e14 bytes
    assert 1e13 < L.n_particles < 3e13
    assert 5e14 < L.ram_bytes < 2e15


def test_amr_vs_equivalent_uniform():
    # AMR with finest level dx=10 km should be much cheaper than
    # a 10 km uniform run over the whole domain.
    uni10 = uniform_load(10.0)
    _, total, _ = amr_load(sample_dx_Re=1.0)
    assert total.n_particles < uni10.n_particles
    assert uni10.n_particles / total.n_particles > 50
