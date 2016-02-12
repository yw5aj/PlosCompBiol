# -*- coding: utf-8 -*-
import numpy as np

from stress_to_spike import stress_to_group_current
import lif_model
from model_constants import MC_GROUPS


TEST_DATA_PATH = './csvs/test/'


# Commonly used constants
tau_nr = 0.008
tau_mc = 1
tau_ad = 0.5  # in sec, new approach 2
k_mc_1 = 0.1
k_nr_1 = 0.9
k_ad_1 = 1
k_nr = 1.5e-12  # in Pa/mA, new approach 3
k_mc = 1.5e-12  # in Pa/mA, new approach 3
k_ad = 2e-12  # in Pa/mA, new approach 2


def load_test_data(vname_list):
    data = {}
    for vname in vname_list:
        data[vname] = np.genfromtxt('%s%s.csv' % (TEST_DATA_PATH, vname),
                                    delimiter=',')
    return data


def setup_lif_model():
    gen_current = setup_gen_function()
    spike_time = lif_model.get_spikes(gen_current)
    np.savetxt(TEST_DATA_PATH + 'spike_time.csv', spike_time, delimiter=',')


def setup_gen_function():
    fine_stress = np.genfromtxt('./csvs/fem/dcon_disp3_stress.csv',
                                delimiter=',')
    fine_time = np.genfromtxt('./csvs/fem/dcon_disp3_time.csv',
                              delimiter=',')
    # Re-save the fine_stress and fine_time
    np.savetxt(TEST_DATA_PATH + 'fine_stress.csv', fine_stress, delimiter=',')
    np.savetxt(TEST_DATA_PATH + 'fine_time.csv', fine_time, delimiter=',')
    # Generator function decay parameters
    params_key_list = ['tau_nr', 'tau_mc', 'tau_ad', 'k_nr', 'k_mc', 'k_ad',
                       'k_nr_1', 'k_mc_1', 'k_ad_1']
    params = {}
    for key in params_key_list:
        params[key] = globals()[key]
    current_dict = stress_to_group_current(fine_time, fine_stress, MC_GROUPS,
                                           **params)
    for key, item in current_dict.items():
        np.savetxt('%s%s.csv' % (TEST_DATA_PATH, key), item, delimiter=',')
    return current_dict['gen_current']


if __name__ == '__main__':
    setup_lif_model()
    setup_gen_function()
