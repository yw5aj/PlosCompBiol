# -*- coding: utf-8 -*-
import pytest
import os
import numpy as np

from model_constants import MC_GROUPS
from setup_test_data import load_test_data, params
from stress_to_spike import stress_to_inst_fr, stress_to_group_current


@pytest.fixture(scope='module')
def load_data():
    vname_list = ['fine_time', 'fine_stress', 'inst_fr_time', 'inst_fr',
                  'group_gen_current']
    return load_test_data(vname_list)


def test_stress_to_group_current(load_data):
    group_gen_current = stress_to_group_current(
        load_data['fine_time'], load_data['fine_stress'], MC_GROUPS, **params)
    assert np.allclose(group_gen_current, load_data['group_gen_current'])


def test_stress_to_spike(load_data):
    inst_fr_time, inst_fr = stress_to_inst_fr(
        load_data['fine_time'], load_data['fine_stress'], MC_GROUPS, **params)
    assert np.allclose(load_data['inst_fr_time'], inst_fr_time)
    assert np.allclose(load_data['inst_fr'], inst_fr)


if __name__ == '__main__':
    pytest.main([os.path.basename(__file__)])
