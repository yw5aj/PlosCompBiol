# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:56:48 2016

@author: Administrator
"""

import numpy as np
import pytest
import os

from setup_test_data import load_test_csv, lmpars
from fit_model import get_single_residual, static_displ_to_residual


@pytest.fixture(scope='module')
def load_data():
    vname_list = ['groups', 'time', 'stress', 'rec_spike_time', 'rec_fr_roll',
                  'single_residual']
    data_dict = load_test_csv(vname_list)
    data_dict['lmpars'] = lmpars
    return data_dict


def test_static_displ_to_residual(load_data):
    assert True


def test_get_single_residual(load_data):
    residual = get_single_residual(**load_data)
    assert np.allclose(residual, load_data['single_residual'])


if __name__ == '__main__':
    pytest.main([os.path.basename(__file__)])
