# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:25:32 2016

@author: Administrator
"""

import numpy as np
import os
from lmfit import Parameters, minimize, fit_report
import matplotlib.pyplot as plt

from stress_to_spike import stress_to_inst_fr
from model_constants import MC_GROUPS


def get_fine_fr():
    return time, fine_fr


def load_rec_fr(animal, stim):
    fname = os.path.join('data', 'rec', '%s_fr_%d.csv' % (animal, stim))
    spike_time, inst_fr = np.genfromtxt(fname, delimiter=',').T
    return exp_fr


if __name__ == '__main__':
    from setup_test_data import params, load_test_data
    vname_list = ['fine_time', 'fine_stress']
    data = load_test_data(vname_list)
    spike_time, inst_fr = stress_to_inst_fr(
        data['fine_time'], data['fine_stress'], MC_GROUPS, **params)
    plt.plot(spike_time, inst_fr)
    animal = 'Piezo2CONT'
    stim = 0
    fname = os.path.join('data', 'rec', '%s_fr_%d.csv' % (animal, stim))
    spike_time, inst_fr = np.genfromtxt(fname, delimiter=',').T
    plt.plot(spike_time, inst_fr, '.r')
    plt.plot(data['fine_time'], data['fine_stress'] / 100, '-g')

