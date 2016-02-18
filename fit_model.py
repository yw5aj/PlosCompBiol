# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:25:32 2016

@author: Administrator
"""

import numpy as np
import os
from lmfit import minimize, fit_report, Parameters
import matplotlib.pyplot as plt
import pickle
import datetime
import re

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


def get_data_dicts(rec_dict, stim):
    # Read recording data
    rec_fr_inst = rec_dict['fr_inst_list'][stim]
    rec_spike_time = rec_dict['spike_time_list'][stim]
    rec_data_dict = {
        'rec_spike_time': rec_spike_time,
        'rec_fr_inst': rec_fr_inst}
    # Read model data
    time, stress = get_fine_stress()
    max_time = rec_dict['max_time_list'][stim]
    stress = adjust_stress_ramp_time(time, stress, max_time)
    mod_data_dict = {
        'groups': MC_GROUPS,
        'time': time,
        'stress': stress}
    fit_data_dict = dict(list(rec_data_dict.items()) +
                         list(mod_data_dict.items()))
    data_dicts = {
        'rec_data_dict': rec_data_dict,
        'mod_data_dict': mod_data_dict,
        'fit_data_dict': fit_data_dict}
    return data_dicts


def fit_single_rec(lmpars, fit_data_dict):
    result = minimize(get_single_residual,
                      lmpars, kws=fit_data_dict, epsfcn=1e-4)
    return result


def plot_single_fit(result, groups, time, stress, rec_spike_time, rec_fr_inst):
    lmpars_fit = result.params
    mod_spike_time, mod_fr_inst = get_mod_spike(lmpars_fit, groups,
                                                time, stress)
    fig, axs = plt.subplots()
    axs.plot(mod_spike_time, mod_fr_inst * 1e3, '-r', label='Experiment')
    axs.plot(rec_spike_time, rec_fr_inst * 1e3, '.k', label='Model')
    axs.set_xlabel('Time (msec)')
    axs.set_ylabel('Instantaneous firing (Hz)')
    fig.tight_layout()
    return fig, axs


def export_fit_result(result, fit_data_dict):
    timestamp = ''.join(re.findall('\d+', str(datetime.datetime.now()))[:-1])
    fname_report = 'fit_report_%s.txt' % timestamp
    fname_pickle = 'fit_result_%s.pkl' % timestamp
    fname_plot = 'fit_plot_%s.png' % timestamp
    pname = os.path.join('data', 'fit')
    with open(os.path.join(pname, fname_report), 'w') as f:
        f.write(fit_report(result))
    with open(os.path.join(pname, fname_pickle), 'wb') as f:
        pickle.dump(result, f)
    # Plot
    fig, axs = plot_single_fit(result, **fit_data_dict)
    fig.savefig(os.path.join(pname, fname_plot), dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    # Load relavent data
    data_dicts_dict = {}
    for animal in ANIMAL_LIST:
        rec_dict = load_rec(animal)
        data_dicts = get_data_dicts(rec_dict, STIM_NUM - 1)
        data_dicts_dict[animal] = data_dicts
    # %% Approach 0: use Adrienne's data
    # Define lmpar
    lmpars = Parameters()
    lmpars.add('tau1', value=8, vary=False)
    lmpars.add('tau2', value=200, vary=False)
    lmpars.add('tau3', value=np.inf, vary=False)
    lmpars.add('k1', value=.025, vary=True)
    lmpars.add('k2', value=.05, vary=True)
    lmpars.add('k3', value=.04, vary=True)
    # Run fitting for the control standard
    result = fit_single_rec(lmpars,
                            data_dicts_dict['Piezo2CONT']['fit_data_dict'])
    export_fit_result(result, data_dicts['fit_data_dict'])
    # %% Approach 4: add ultra-slow adapting constant
    lmpars = Parameters()
    lmpars.add('tau1', value=8, vary=False)
    lmpars.add('tau2', value=200, vary=False)
    lmpars.add('tau3', value=1832, vary=False)
    lmpars.add('tau4', value=np.inf, vary=False)
    lmpars.add('k1', value=.025, vary=True)
    lmpars.add('k2', value=.05, vary=True)
    lmpars.add('k3', value=.05, vary=True)
    lmpars.add('k4', value=.04, vary=True)
    # Run fitting
    result = fit_single_rec(lmpars, data_dicts['fit_data_dict'])
    export_fit_result(result, data_dicts['fit_data_dict'])
    # %% Playing with Approach 4
    with open('data/fit/fit_result_20160218165404.pkl', 'rb') as f:
        result = pickle.load(f)
    # Have a function that generates different responses given the result