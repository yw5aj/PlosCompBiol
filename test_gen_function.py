# -*- coding: utf-8 -*-
import numpy as np
import pytest
import os

from setup_test_data import load_test_data, tau1, tau2, tau3, k1, k3, a, b, c
from gen_function import stress_to_current_new
from model_constants import MC_GROUPS


@pytest.fixture(scope='module')
def load_data():
    vname_list = ['fine_stress', 'fine_time', 'gen_current',
                  'nr_current', 'mc_current']
    return load_test_data(vname_list)


def test_stress_to_current(load_data):
    params = {
        'tau_nr': tau1,
        'tau_mc': tau2,
        'tau_ad': tau3,
        'k_nr': a,
        'k_nr_1': k3,
        'k_mc': b,
        'k_mc_1': k1,
        'k_ad': c,
        'k_ad_1': 1,
            }
    current_dict = stress_to_current_new(
        load_data['fine_time'], load_data['fine_stress'],
        **params)
    assert np.allclose(current_dict['gen_current'],
                       load_data['gen_current'][:, 0] / MC_GROUPS[0])
    assert np.allclose(current_dict['mc_current'],
                       load_data['mc_current'][:, 0] / MC_GROUPS[0])
    assert np.allclose(current_dict['nr_current'] + current_dict['ad_current'],
                       load_data['nr_current'][:, 0] / MC_GROUPS[0])


if __name__ == '__main__':
    pytest.main([os.path.basename(__file__)])
