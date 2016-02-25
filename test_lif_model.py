import numpy as np
import pytest
import os
import imp

import setpyximport
import cy_lif_model

import lif_model
from setup_test_data import load_test_csv, lmpars


@pytest.fixture(scope='module')
def load_data():
    vname_list = ['group_gen_current', 'spike_time']
    return load_test_csv(vname_list)


def test_dudt():
    assert cy_lif_model.dudt(15, 20) == lif_model.dudt(15, 20)


def test_runge_kutta(load_data):
    current = load_data['group_gen_current']
    assert np.allclose(lif_model.runge_kutta(current, 0),
                       np.array(cy_lif_model.runge_kutta(current, 0)))


def test_get_spikes(load_data):
    gen_current = load_data['group_gen_current']
    spike_time = load_data['spike_time']
    py_spike_time = np.array(lif_model.get_spikes(gen_current))
    cy_spike_time = np.array(cy_lif_model.get_spikes(gen_current))
    assert np.allclose(py_spike_time, cy_spike_time)
    assert np.allclose(spike_time, py_spike_time)
    assert np.allclose(spike_time, cy_spike_time)


if __name__ == '__main__':
    imp.reload(cy_lif_model)
    pytest.main([os.path.basename(__file__)])
    # %% Test speed
    '''
    %timeit -n3 spike_time = lif_model.get_spikes(gen_current)
    %timeit -n3 spike_time = cy_lif_model.get_spikes(gen_current)
    '''
