# -*- coding: utf-8 -*-
import numpy as np
import shutil

from stress_to_spike import stress_to_group_current
from gen_function import stress_to_current
import cy_lif_model as lif_model
from model_constants import MC_GROUPS
from stress_to_spike import stress_to_inst_fr


TEST_DATA_PATH = './data/test/'


# Commonly used constants
params = {
    'tau_arr': np.array([8, 500, 1000, np.inf]),
    'k_arr': np.array([1.35, 2, .15, 1.5])}


def load_test_data(vname_list):
    data = {}
    for vname in vname_list:
        data[vname] = np.genfromtxt('%s%s.csv' % (TEST_DATA_PATH, vname),
                                    delimiter=',')
    return data


def save_test_data(data):
    for key, item in data.items():
        np.savetxt('%s%s.csv' % (TEST_DATA_PATH, key), item, delimiter=',')


def setup_lif_model(data):
    data['group_gen_current'] = stress_to_group_current(
        data['fine_time'], data['fine_stress'], MC_GROUPS, **params)
    data['spike_time'] = lif_model.get_spikes(data['group_gen_current'])


def copy_stress():
    shutil.copy('./data/fem/dcon_disp3_stress.csv',
                TEST_DATA_PATH + 'fine_stress.csv')
    shutil.copy('./data/fem/dcon_disp3_time.csv',
                TEST_DATA_PATH + 'fine_time.csv')


def setup_gen_function(data):
    # Generator function decay parameters
    data['current_arr'] = stress_to_current(
        data['fine_time'], data['fine_stress'], **params)


def setup_stress_to_spike(data):
    data['spike_time'], data['inst_fr'] = stress_to_inst_fr(
        data['fine_time'], data['fine_stress'], MC_GROUPS, **params)


if __name__ == '__main__':
    copy_stress()
    vname_list = ['fine_time', 'fine_stress']
    data = load_test_data(vname_list)
    setup_lif_model(data)
    setup_gen_function(data)
    setup_stress_to_spike(data)
    save_test_data(data)
