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
from model_constants import (MC_GROUPS, FS, ANIMAL_LIST, STIM_NUM, REF_DISPL,
                             REF_STIM, WINDOW)
from gen_function import get_interp_stress


def get_log_sample_after_peak(spike_time, fr_roll, n_sample):
    maxidx = fr_roll.argmax()
    log_range = np.logspace(0, np.log10(spike_time.size - maxidx),
                            n_sample).astype(np.int) - 1
    sample_range = maxidx + log_range
    return spike_time[sample_range], fr_roll[sample_range]


def get_single_residual(lmpars, groups,
                        time, stress, rec_spike_time, rec_fr_roll,
                        **kwargs):
    sample_spike_time, sample_fr_roll = get_log_sample_after_peak(
        rec_spike_time, rec_fr_roll, 50)
    mod_spike_time, mod_fr_inst = get_mod_spike(lmpars, groups, time, stress)
    mod_fr_inst_interp = np.interp(sample_spike_time,
                                   mod_spike_time, mod_fr_inst)
    residual = mod_fr_inst_interp - sample_fr_roll
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
    fr_roll_list, max_time_list, max_fr_roll_list = [], [], []
    for spike_time in spike_time_list:
        fr_roll = spike_time_to_fr_roll(spike_time, WINDOW)
        fr_roll_list.append(fr_roll)
        max_time_list.append(spike_time[fr_roll.argmax()])
        max_fr_roll_list.append(fr_roll.max())
    rec_dict = {
        'spike_time_list': spike_time_list,
        'fr_inst_list': fr_inst_list,
        'fr_roll_list': fr_roll_list,
        'max_time_list': max_time_list,
        'max_fr_roll_list': max_fr_roll_list}
    return rec_dict


def adjust_stress_ramp_time(time, stress, max_time_spike, stretch_coeff=1.25):
    """
    Stretch the ramp phase of the stress such that the ramp time matches the
    `max_time_target`. The hold phase of the stress is unchanged.

    Parameters
    ----------
    time : 1xN array
    stress : 1xN array
    max_time_spike : float
        When does the peak firing happens for the recording
    stretch_coeff : float
        The actual `max_time_target` is stretched by this coeff.

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
    max_time_target = max_time_spike * stretch_coeff
    time_new[:max_idx + 1] *= max_time_target / max_time_original
    time_new[max_idx + 1:] += max_time_target - max_time_original
    stress_new = np.interp(time, time_new, stress)
    return stress_new


def get_data_dicts(animal, stim, static_displ, rec_dict=None):
    if rec_dict is None:
        rec_dict = load_rec(animal)
    # Read recording data
    rec_fr_inst = rec_dict['fr_inst_list'][stim]
    rec_spike_time = rec_dict['spike_time_list'][stim]
    rec_fr_roll = rec_dict['fr_roll_list'][stim]
    rec_data_dict = {
        'rec_spike_time': rec_spike_time,
        'rec_fr_inst': rec_fr_inst,
        'rec_fr_roll': rec_fr_roll}
    # Read model data
    time, stress = get_interp_stress(static_displ)
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


def plot_single_fit(lmpars_fit, groups, time, stress,
                    rec_spike_time, roll=True, **kwargs):
    mod_spike_time, mod_fr_inst = get_mod_spike(lmpars_fit, groups,
                                                time, stress)
    if roll:
        rec_fr = kwargs['rec_fr_roll']
    else:
        rec_fr = kwargs['rec_fr_inst']
    fig, axs = plt.subplots()
    axs.plot(mod_spike_time, mod_fr_inst * 1e3, '-r', label='Experiment')
    axs.plot(rec_spike_time, rec_fr * 1e3, '.k', label='Model')
    axs.set_xlabel('Time (msec)')
    axs.set_ylabel('Instantaneous firing (Hz)')
    fig.tight_layout()
    return fig, axs


def get_time_stamp():
    time_stamp = ''.join(re.findall(
        '\d+', str(datetime.datetime.now()))[:-1])
    return time_stamp


def export_fit_result(result, fit_data_dict, label_str=None):
    if label_str is None:
        label_str = get_time_stamp()
    fname_report = 'fit_report_%s.txt' % label_str
    fname_pickle = 'fit_result_%s.pkl' % label_str
    fname_plot = 'fit_plot_%s.png' % label_str
    pname = os.path.join('data', 'fit')
    with open(os.path.join(pname, fname_report), 'w') as f:
        f.write(fit_report(result))
    with open(os.path.join(pname, fname_pickle), 'wb') as f:
        pickle.dump(result, f)
    # Plot
    fig, axs = plot_single_fit(result.params, **fit_data_dict)
    fig.savefig(os.path.join(pname, fname_plot), dpi=300)
    plt.close(fig)


def plot_static_displ_to_mod_spike(lmdispl, lmpars, animal, stim):
    data_dicts = get_data_dicts(animal, stim, lmdispl['static_displ'].value)
    return plot_single_fit(lmpars, **data_dicts['fit_data_dict'])


def static_displ_to_mod_spike(lmdispl, lmpars, animal, stim):
    """
    Given the model fitting parameters `lmpars`, we can find a stress trace
    that can fit to a certain fiber. This fiber is from `stim` in `animal`,
    and we fit by adjusting the `static_displ` to interpolate the stress.
    """
    data_dicts = get_data_dicts(animal, stim, lmdispl['static_displ'].value)
    return get_mod_spike(lmpars, **data_dicts['mod_data_dict'])


def static_displ_to_residual(lmdispl, lmpars, fit_data_dict):
    return get_single_residual(lmpars, **fit_data_dict)


def fit_static_displ(lmdispl, lmpars, animal, stim):
    data_dicts = get_data_dicts(animal, stim, lmdispl['static_displ'].value)
    result = minimize(static_displ_to_residual,
                      lmdispl, args=(lmpars, data_dicts['fit_data_dict']),
                      epsfcn=1e-4)
    return result


class FitApproach():

    def __init__(self, lmpars_init, label=None):
        self.lmpars_init = lmpars_init
        if label is None:
            self.label = get_time_stamp()
        else:
            self.label = label
        # Load data
        self.load_rec_dicts()

    def load_rec_dicts(self):
        self.rec_dicts = {animal: load_rec(animal) for animal in ANIMAL_LIST}

    def get_data_dicts(self, animal, stim, static_displ):
        data_dicts = get_data_dicts(animal, stim, static_displ,
                                    self.rec_dicts[animal])
        return data_dicts

    def fit_ref(self, export=True, plot=False):
        data_dicts = self.get_data_dicts('Piezo2CONT', REF_STIM, REF_DISPL)
        self.ref_result = fit_single_rec(self.lmpars_init,
                                         data_dicts['fit_data_dict'])
        self.lmpars_fit = self.ref_result.params
        if export:
            export_fit_result(self.ref_result, data_dicts['fit_data_dict'],
                              label_str=self.label)
        if plot:
            fig, axs = plot_single_fit(self.ref_result.params,
                                       **data_dicts['fit_data_dict'])
            return fig, axs


# Define parameters for fitting
# E.g., t3f1v23 means: 3 taus, fix tau1, vary tau2, tau3
lmpars_init_dict = {}
# Approach t2f12: use Adrienne's data
lmpars = Parameters()
lmpars.add('tau1', value=8, vary=False)
lmpars.add('tau2', value=200, vary=False)
lmpars.add('tau3', value=np.inf, vary=False)
lmpars.add('k1', value=1, vary=True)
lmpars.add('k2', value=.3, vary=True)
lmpars.add('k3', value=.04, vary=True)
lmpars_init_dict['t2f12'] = lmpars

# Approach t2v12: let tau1 and tau2 float
lmpars = Parameters()
lmpars.add('tau1', value=8, vary=True, min=0, max=5000)
lmpars.add('tau2', value=200, vary=True, min=0, max=5000)
lmpars.add('tau3', value=np.inf, vary=False)
lmpars.add('k1', value=1, vary=True)
lmpars.add('k2', value=.3, vary=True)
lmpars.add('k3', value=.04, vary=True)
lmpars_init_dict['t2v12'] = lmpars

# Approach t3f1v23: fix tau1 and float tau2, 3
# Define lmpar
lmpars = Parameters()
lmpars.add('tau1', value=8, vary=False, min=0, max=5000)
lmpars.add('tau2', value=200, vary=True, min=0, max=5000)
lmpars.add('tau3', value=1000, vary=True, min=0, max=5000)
lmpars.add('tau4', value=np.inf, vary=False)
lmpars.add('k1', value=1., vary=True)
lmpars.add('k2', value=.3, vary=True)
lmpars.add('k3', value=.04, vary=True)
lmpars.add('k4', value=.04, vary=True)
lmpars_init_dict['t3f1v23'] = lmpars

# Approach t3f123: add ultra-slow adapting constant and fix tau1, 2, 3
lmpars = Parameters()
lmpars.add('tau1', value=8, vary=False)
lmpars.add('tau2', value=200, vary=False)
lmpars.add('tau3', value=1832, vary=False)
lmpars.add('tau4', value=np.inf, vary=False)
lmpars.add('k1', value=1., vary=True)
lmpars.add('k2', value=.3, vary=True)
lmpars.add('k3', value=.04, vary=True)
lmpars.add('k4', value=.04, vary=True)
lmpars_init_dict['t3f123'] = lmpars

if __name__ == '__main__':
    run_fitting = True
    # %%
    if run_fitting:
        fitApproach_dict = {}
        for approach, lmpars_init in lmpars_init_dict.items():
            fitApproach = FitApproach(lmpars_init, approach)
            fitApproach.fit_ref()
            fitApproach_dict[approach] = fitApproach

    # %% Playing with Approach t3f123
#    fitApproach = fitApproach_dict['t3f123']
    with open('data/fit/fit_result_approach_t3f123.pkl', 'rb') as f:
        result = pickle.load(f)
    fitApproach.ref_result = result
    fitApproach.lmpars_fit = result.params
    lmdispl = Parameters()
    lmdispl.add('static_displ', value=.33, min=0, max=.6, vary=True)
    plot_static_displ_to_mod_spike(lmdispl, fitApproach.lmpars_fit,
                                   'Piezo2CONT', 2)
#    fit_static_displ(lmdispl, lmpars, 'Piezo2CONT', 0)
