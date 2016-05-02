# -*- coding: utf-8 -*-
import numpy as np
import pytest
import os

from setup_test_data import (load_test_csv, params, interp_static_displ,
                             extrap_static_displ)
from gen_function import stress_to_current, get_interp_stress


@pytest.fixture(scope='module')
def load_data():
    vname_list = ['fine_stress', 'fine_time', 'current_arr',
                  'interp_time', 'interp_stress',
                  'extrap_time', 'extrap_stress']
    return load_test_csv(vname_list)


def test_stress_to_current(load_data):
    current_arr = stress_to_current(
        load_data['fine_time'], load_data['fine_stress'], **params)
    assert np.allclose(current_arr, load_data['current_arr'])


def test_get_interp_stress(load_data):
    interp_time, interp_stress = get_interp_stress(interp_static_displ)
    extrap_time, extrap_stress = get_interp_stress(extrap_static_displ)
    assert np.allclose(interp_stress, load_data['interp_stress'])
    assert np.allclose(extrap_stress, load_data['extrap_stress'])


if __name__ == '__main__':
    pytest.main([os.path.basename(__file__)])
