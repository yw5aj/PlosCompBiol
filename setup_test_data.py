# -*- coding: utf-8 -*-
import numpy as np

from stress_to_spike import stress_to_group_current
import lif_model
from model_constants import MC_GROUPS


TEST_DATA_PATH = './csvs/test/'


# Commonly used constants
tau1 = 0.008
tau2 = 1
tau3 = 0.5  # in sec, new approach 2
k1 = 0.1
k3 = 0.9
a = 1.5e-12  # in Pa/mA, new approach 3
b = 1.5e-12  # in Pa/mA, new approach 3
c = 2e-12  # in Pa/mA, new approach 2


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
    tau1_m = tau1
    tau2_ap2 = tau2
    k1_ap2 = k1
    gen_current = stress_to_group_current(fine_time, fine_stress, tau1_m,
                                          tau2_ap2, k1_ap2, 'gen', MC_GROUPS)
    nr_current = stress_to_group_current(fine_time, fine_stress, tau1_m,
                                         tau2_ap2, k1_ap2, 'nr', MC_GROUPS)
    mc_current = stress_to_group_current(fine_time, fine_stress, tau1_m,
                                         tau2_ap2, k1_ap2, 'mc', MC_GROUPS)
    np.savetxt(TEST_DATA_PATH + 'gen_current.csv', gen_current, delimiter=',')
    np.savetxt(TEST_DATA_PATH + 'nr_current.csv', nr_current, delimiter=',')
    np.savetxt(TEST_DATA_PATH + 'mc_current.csv', mc_current, delimiter=',')
    return gen_current


if __name__ == '__main__':
    setup_lif_model()
    setup_gen_function()
