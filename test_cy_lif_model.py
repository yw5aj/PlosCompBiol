import numpy as np
import setpyximport

from stress_to_spike import sed_to_group_current
import lif_model
import cy_lif_model
from model_constants import MC_GROUPS


if __name__ == '__main__':
    fine_stress = np.genfromtxt('./Shawn model data/dcon_disp3_stress.csv',
                                delimiter=',')
    fine_time = np.genfromtxt('./Shawn model data/dcon_disp3_time.csv',
                              delimiter=',')
    # %% Generator function decay parameters
    tau1_m = 0.008
    tau2_ap2 = 1
    k1_ap2 = 0.1
    gen_current = sed_to_group_current(fine_time, fine_stress, tau1_m,
                                       tau2_ap2, k1_ap2, 'nr', MC_GROUPS)
    spike_time = lif_model.get_spikes(gen_current)
    cy_spike_time = cy_lif_model.get_spikes(gen_current)
    assert np.all(np.array(spike_time) == np.array(cy_spike_time))
    # %% Test speed
    '''
    %timeit -n3 spike_time = lif_model.get_spikes(gen_current)
    %timeit -n3 spike_time = cy_lif_model.get_spikes(gen_current)
    '''
