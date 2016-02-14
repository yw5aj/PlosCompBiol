import numpy as np
import pytest
import os

import setpyximport
import cy_lif_model

import lif_model
from setup_test_data import load_test_data


@pytest.fixture(scope='module')
def load_data():
    vname_list = ['group_gen_current', 'spike_time']
    return load_test_data(vname_list)


def test_lif_model(load_data):
    gen_current = load_data['group_gen_current']
    spike_time = load_data['spike_time']
    py_spike_time = np.array(lif_model.get_spikes(gen_current))
    cy_spike_time = np.array(cy_lif_model.get_spikes(gen_current))
    assert np.allclose(py_spike_time, cy_spike_time)
    assert np.allclose(spike_time, py_spike_time)
    assert np.allclose(spike_time, cy_spike_time)


if __name__ == '__main__':
    pytest.main([os.path.basename(__file__)])
    # %% Test speed
    '''
    %timeit -n3 spike_time = lif_model.get_spikes(gen_current)
    %timeit -n3 spike_time = cy_lif_model.get_spikes(gen_current)
    '''
