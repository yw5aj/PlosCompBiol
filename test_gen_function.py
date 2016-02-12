# -*- coding: utf-8 -*-
import numpy as np
import pytest
import os

from setup_test_data import (load_test_data, tau_nr, tau_mc, tau_ad,
                             k_mc_1, k_nr_1, k_ad_1, k_mc, k_nr, k_ad)
from gen_function import stress_to_current
from model_constants import MC_GROUPS


@pytest.fixture(scope='module')
def load_data():
    vname_list = ['fine_stress', 'fine_time', 'gen_current',
                  'nr_current', 'mc_current', 'ad_current']
    return load_test_data(vname_list)


def test_stress_to_current(load_data):
    params_key_list = ['tau_nr', 'tau_mc', 'tau_ad', 'k_nr', 'k_mc', 'k_ad',
                       'k_nr_1', 'k_mc_1', 'k_ad_1']
    params = {}
    for key in params_key_list:
        params[key] = globals()[key]
    current_dict = stress_to_current(
        load_data['fine_time'], load_data['fine_stress'],
        **params)
    for key, item in current_dict.items():
        assert np.allclose(current_dict[key],
                           load_data[key][:, 0] / MC_GROUPS[0])


if __name__ == '__main__':
    pytest.main([os.path.basename(__file__)])
