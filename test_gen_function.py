# -*- coding: utf-8 -*-
import numpy as np
import pytest
import os

from setup_test_data import load_test_data, params
from gen_function import stress_to_current
from model_constants import MC_GROUPS


@pytest.fixture(scope='module')
def load_data():
    vname_list = ['fine_stress', 'fine_time', 'gen_current',
                  'nr_current', 'mc_current', 'ad_current']
    return load_test_data(vname_list)


def test_stress_to_current(load_data):
    tau_arr = np.array((params['tau_nr'], params['tau_mc'],
                        params['tau_ad'], np.inf))
    k_arr = np.array((params['k_nr'] * params['k_nr_1'],
                      params['k_mc'] * params['k_mc_1'],
                      params['k_ad'] * params['k_ad_1'],
                      params['k_nr'] * (1 - params['k_nr_1']) +
                      params['k_mc'] * (1 - params['k_mc_1']) +
                      params['k_ad'] * (1 - params['k_ad_1'])))
    current_arr = stress_to_current(
        load_data['fine_time'], load_data['fine_stress'],
        tau_arr, k_arr)
    assert np.allclose(current_arr.sum(axis=1),
                       load_data['gen_current'][:, 0] / MC_GROUPS[0])


if __name__ == '__main__':
    pytest.main([os.path.basename(__file__)])
