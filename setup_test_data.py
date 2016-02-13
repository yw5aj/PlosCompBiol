# -*- coding: utf-8 -*-
import numpy as np
import shutil

from stress_to_spike import stress_to_group_current
import lif_model
from model_constants import MC_GROUPS
from stress_to_spike import stress_to_inst_fr


TEST_DATA_PATH = './csvs/test/'


# Commonly used constants
params = {'k_ad': 2e-12,
          'k_ad_1': 1,
          'k_mc': 1.5e-12,
          'k_mc_1': 0.1,
          'k_nr': 1.5e-12,
          'k_nr_1': 0.9,
          'tau_ad': 0.5,
          'tau_mc': 1,
          'tau_nr': 0.008}


def load_test_data(vname_list):
    data = {}
    for vname in vname_list:
        data[vname] = np.genfromtxt('%s%s.csv' % (TEST_DATA_PATH, vname),
                                    delimiter=',')
    return data


def setup_lif_model(data):
    gen_current = data['gen_current']
    spike_time = lif_model.get_spikes(gen_current)
    np.savetxt(TEST_DATA_PATH + 'spike_time.csv', spike_time, delimiter=',')


def copy_stress():
    shutil.copy('./csvs/fem/dcon_disp3_stress.csv',
                TEST_DATA_PATH + 'fine_stress.csv')
    shutil.copy('./csvs/fem/dcon_disp3_time.csv',
                TEST_DATA_PATH + 'fine_time.csv')


def setup_gen_function(data):
    # Generator function decay parameters
    current_dict = stress_to_group_current(
        data['fine_time'], data['fine_stress'], MC_GROUPS, **params)
    for key, item in current_dict.items():
        np.savetxt('%s%s.csv' % (TEST_DATA_PATH, key), item, delimiter=',')


def setup_stress_to_spike(data):
    inst_fr_time, inst_fr = stress_to_inst_fr(
        data['fine_time'], data['fine_stress'], MC_GROUPS, **params)
    np.savetxt('%sinst_fr_time.csv' % TEST_DATA_PATH, inst_fr_time,
               delimiter=',')
    np.savetxt('%sinst_fr.csv' % TEST_DATA_PATH, inst_fr, delimiter=',')


if __name__ == '__main__':
    copy_stress()
    vname_list = ['fine_time', 'fine_stress', 'gen_current']
    data = load_test_data(vname_list)
    setup_lif_model(data)
    setup_gen_function(data)
    setup_stress_to_spike(data)
