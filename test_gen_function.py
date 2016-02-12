# -*- coding: utf-8 -*-
import numpy as np
import pytest
import os

from setup_test_data import load_test_data


@pytest.fixture(scope='module')
def load_data():
    vname_list = ['fine_stress', 'fine_time', 'gen_current',
                  'nr_current', 'mc_current']
    return load_test_data(vname_list)


def test_stress_to_current(load_data):
    pass


if __name__ == '__main__':
    pytest.main([os.path.basename(__file__)])
