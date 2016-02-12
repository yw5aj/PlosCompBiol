import numpy as np
import pytest
import os

from stress_to_spike import stress_to_group_current
import lif_model
from model_constants import MC_GROUPS

import setpyximport
import cy_lif_model


def test_cy_lif_model():
    fine_stress = np.genfromtxt('./csvs/fem/dcon_disp3_stress.csv',
                                delimiter=',')
    fine_time = np.genfromtxt('./csvs/fem/dcon_disp3_time.csv',
                              delimiter=',')
    # %% Generator function decay parameters
    tau1_m = 0.008
    tau2_ap2 = 1
    k1_ap2 = 0.1
    gen_current = stress_to_group_current(fine_time, fine_stress, tau1_m,
                                          tau2_ap2, k1_ap2, 'nr', MC_GROUPS)
    spike_time = lif_model.get_spikes(gen_current)
    cy_spike_time = cy_lif_model.get_spikes(gen_current)
    assert np.all(np.array(spike_time) == np.array(cy_spike_time))


if __name__ == '__main__':
    pytest.main([os.path.basename(__file__)])
    # %% Test speed
    '''
    %timeit -n3 spike_time = lif_model.get_spikes(gen_current)
    %timeit -n3 spike_time = cy_lif_model.get_spikes(gen_current)
    '''
