# -*- coding: utf-8 -*-
import numpy as np
import shutil
from lmfit import Parameters

from stress_to_spike import stress_to_group_current, stress_to_fr_inst
from gen_function import stress_to_current, get_interp_stress
import cy_lif_model as lif_model
from model_constants import MC_GROUPS, REF_DISPL, REF_STIM
from fit_model import get_data_dicts, get_single_residual


TEST_DATA_PATH = './data/test/'


# Commonly used constants
params = {
    'tau_arr': np.array([8, 500, 1000, np.inf]),
    'k_arr': np.array([1.35, 2, .15, 1.5])}
lmpars = Parameters()
lmpars.add_many(('tau1', 8), ('tau2', 200), ('tau3', 1832), ('tau4', np.inf),
                ('k1', 0.782), ('k2', 0.304), ('k3', 0.051), ('k4', 0.047))
interp_static_displ = .35
extrap_static_displ = .65


def load_test_csv(vname_list):
    data = {}
    for vname in vname_list:
        data[vname] = np.genfromtxt('%s%s.csv' % (TEST_DATA_PATH, vname),
                                    delimiter=',')
    return data


def save_test_csv(data):
    for key, item in data.items():
        np.savetxt('%s%s.csv' % (TEST_DATA_PATH, key), item, delimiter=',')


def setup_lif_model(data):
    data['group_gen_current'] = stress_to_group_current(
        data['fine_time'], data['fine_stress'], MC_GROUPS, **params)
    data['spike_time'] = lif_model.get_spikes(data['group_gen_current'])


def copy_stress():
    shutil.copy('./data/fem/LindsayThesisData/dcon_disp3_stress.csv',
                TEST_DATA_PATH + 'fine_stress.csv')
    shutil.copy('./data/fem/LindsayThesisData/dcon_disp3_time.csv',
                TEST_DATA_PATH + 'fine_time.csv')


def setup_gen_function(data):
    # Generator function decay parameters
    data['current_arr'] = stress_to_current(
        data['fine_time'], data['fine_stress'], **params)


def setup_stress_to_spike(data):
    data['spike_time'], data['fr_inst'] = stress_to_fr_inst(
        data['fine_time'], data['fine_stress'], MC_GROUPS, **params)
    data['interp_time'], data['interp_stress'] = get_interp_stress(
        interp_static_displ)
    data['extrap_time'], data['extrap_stress'] = get_interp_stress(
        extrap_static_displ)


def setup_fit_lif(data):
    # Input data
    data_dicts = get_data_dicts(REF_STIM, 'Piezo2CONT')
    for key, item in data_dicts.items():
        data.update(item)
    # Output data
    data['single_residual'] = get_single_residual(
        lmpars, **data_dicts['fit_data_dict'])


if __name__ == '__main__':
    copy_stress()
    vname_list = ['fine_time', 'fine_stress']
    data = load_test_csv(vname_list)
    setup_lif_model(data)
    setup_gen_function(data)
    setup_stress_to_spike(data)
    setup_fit_lif(data)
    save_test_csv(data)
