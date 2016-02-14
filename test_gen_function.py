# -*- coding: utf-8 -*-
import numpy as np
import pytest
import os

from setup_test_data import load_test_data, params
from gen_function import stress_to_current


@pytest.fixture(scope='module')
def load_data():
    vname_list = ['fine_stress', 'fine_time', 'current_arr']
    return load_test_data(vname_list)


def test_stress_to_current(load_data):
    current_arr = stress_to_current(
        load_data['fine_time'], load_data['fine_stress'], **params)
    assert np.allclose(current_arr.sum(axis=1),
                       load_data['current_arr'].sum(axis=1) * 1e9)


if __name__ == '__main__':
    pytest.main([os.path.basename(__file__)])
