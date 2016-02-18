# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:25:32 2016

@author: Administrator
"""

import numpy as np
import os
from lmfit import Parameters, minimize, fit_report
import matplotlib.pyplot as plt

from stress_to_spike import (stress_to_fr_inst, spike_time_to_fr_roll,
                             spike_time_to_fr_inst)
from model_constants import MC_GROUPS, FS, ANIMAL_LIST, STIM_NUM
from gen_function import get_fine_stress


def get_single_residual(lmpars, groups,
                        time, stress, rec_spike_time, rec_fr_inst):
    mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress)
    mod_fr_inst_interp = np.interp(rec_spike_time,
                                   mod_spike_time, mod_fr_inst)
    residual = mod_fr_inst_interp - rec_fr_inst
    print((residual**2).sum())
    return residual


def get_mod_spike(lmpars, groups, time, stress):
    params = lmpars_to_params(lmpars)
    mod_spike_time, mod_fr_inst = stress_to_fr_inst(time, stress,
                                                    groups, **params)
    return (mod_spike_time, mod_fr_inst)


def lmpars_to_params(lmpars):
    lmpars_dict = lmpars.valuesdict()
    # Export parameters to separate dicts and use indices as keys
    separate_dict = {'tau': {}, 'k': {}}
    for var, val in lmpars_dict.items():
        for param in separate_dict.keys():
            if param in var:
                index = int(var.split(param)[-1])
                separate_dict[param][index] = val
    # Convert to final format of parameter dict
    params = {}
    for param, indexed_dict in separate_dict.items():
        params[param + '_arr'] = np.array(
            np.array([indexed_dict[index]
                      for index in sorted(indexed_dict.keys())]))
    return params


def load_rec(animal):
    fname = os.path.join('data', 'rec', '%s_spike.csv' % animal)
    spike_arr = np.genfromtxt(fname, delimiter=',')
    spike_time_list = [spike.nonzero()[0] / FS for spike in spike_arr.T]
    fr_inst_list = [spike_time_to_fr_inst(spike_time)
                    for spike_time in spike_time_list]
    max_time_list = [spike_time[spike_time_to_fr_roll(spike_time, 5).argmax()]
                     for spike_time in spike_time_list]
    rec_dict = {
        'spike_time_list': spike_time_list,
        'fr_inst_list': fr_inst_list,
        'max_time_list': max_time_list}
    return rec_dict


def adjust_stress_ramp_time(time, stress, max_time_target):
    """
    Stretch the ramp phase of the stress such that the ramp time matches the
    `max_time_target`. The hold phase of the stress is unchanged.

    Parameters
    ----------
    time : 1xN array
    stress : 1xN array
    max_time_target : float

    Returns
    -------
    stress_new : 1XN array
    """
    # Clean the zeros in the beginning of the stress trace, if any
    start_idx = (stress == 0).nonzero()[0][-1]
    time = time[start_idx:] - time[start_idx]
    stress = stress[start_idx:]
    # Do the stretch
    max_idx = stress.argmax()
    max_time_original = time[max_idx]
    time_new = time.copy()
    time_new[:max_idx + 1] *= max_time_target / max_time_original
    time_new[max_idx + 1:] += max_time_target - max_time_original
    stress_new = np.interp(time, time_new, stress)
    return stress_new


if __name__ == '__main__':
    time, stress = get_fine_stress()
    groups = MC_GROUPS
    lmpars = Parameters()
    lmpars.add_many(('tau1', 8, False, 0, None),
                    ('tau2', 500, False, 0, None),
                    ('tau3', 1000, False, 0, None),
                    ('tau4', np.inf, False, 0, None),
                    ('k1', .025, True, 0, None),
                    ('k2', .05, True, 0, None),
                    ('k3', .004, True, 0, None),
                    ('k4', 0.04, True, 0, None))
    animal = 'Piezo2CONT'
    rec_dict = load_rec(animal)
    rec_fr_inst = rec_dict['fr_inst_list'][-1]
    rec_spike_time = rec_dict['spike_time_list'][-1]
    max_time = rec_dict['max_time_list'][-1]
    stress = adjust_stress_ramp_time(time, stress, max_time)
    # %% Try to minimize one single trace
    res = minimize(get_single_residual, lmpars,
                   args=(groups, time, stress, rec_spike_time, rec_fr_inst),
                   epsfcn=1e-4)
    # %% Plot the fitting result
    lmpars_fit = res.params
    mod_spike_time, mod_fr_inst = get_mod_spike(lmpars_fit, groups,
                                                time, stress)
    fig, axs = plt.subplots()
    axs.plot(mod_spike_time, mod_fr_inst * 1e3, '-r', label='Experiment')
    axs.plot(rec_spike_time, rec_fr_inst * 1e3, '.k', label='Model')
    axs.set_xlabel('Time (msec)')
    axs.set_ylabel('Instantaneous firing (Hz)')
    fig.tight_layout()
