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
                             REF_STIM, REF_ANIMAL, WINDOW)
from gen_function import get_interp_stress


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


def adjust_stress_ramp_time(time, stress, max_time_spike, stretch_coeff):
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


def get_data_dicts(stim, static_displ, animal=None, rec_dict=None):
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
    stretch_coeff = (5 + stim) / 4.
    stress = adjust_stress_ramp_time(time, stress, max_time, stretch_coeff)
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
                    rec_spike_time, plot_kws=None, roll=True,
                    fig=None, axs=None, **kwargs):
    mod_spike_time, mod_fr_inst = get_mod_spike(lmpars_fit, groups,
                                                time, stress)
    if roll:
        rec_fr = kwargs['rec_fr_roll']
    else:
        rec_fr = kwargs['rec_fr_inst']
    if fig is None and axs is None:
        fig, axs = plt.subplots()
        axs0 = axs
        axs1 = axs
    elif isinstance(axs, np.ndarray):
        axs0 = axs[0]
        axs1 = axs[1]
    else:
        axs0 = axs
        axs1 = axs
    axs0.plot(mod_spike_time, mod_fr_inst * 1e3, '-', **plot_kws)
    axs1.plot(rec_spike_time, rec_fr * 1e3, '.', **plot_kws)
    axs0.set_xlabel('Time (msec)')
    axs1.set_xlabel('Time (msec)')
    axs1.set_ylabel('Instantaneous firing (Hz)')
    axs0.set_ylabel('Instantaneous firing (Hz)')
    fig.tight_layout()
    return fig, axs


def get_time_stamp():
    time_stamp = ''.join(re.findall(
        '\d+', str(datetime.datetime.now()))[:-1])
    return time_stamp


def export_ref_fit(result, fit_data_dict, label_str=None,
                   subfolder=''):
    if label_str is None:
        label_str = get_time_stamp()
    fname_report = 'ref_fit_%s.txt' % label_str
    fname_pickle = 'ref_fit_%s.pkl' % label_str
    fname_plot = 'ref_fit_%s.png' % label_str
    pname = os.path.join('data', 'fit', subfolder)
    with open(os.path.join(pname, fname_report), 'w') as f:
        f.write(fit_report(result))
    with open(os.path.join(pname, fname_pickle), 'wb') as f:
        pickle.dump(result, f)
    # Plot
    fig, axs = plot_single_fit(result.params, **fit_data_dict)
    fig.savefig(os.path.join(pname, fname_plot), dpi=300)
    plt.close(fig)


def export_displ_fit(result_static_displ, lmpars, stim, pname,
                     animal=None, rec_dict=None):
    with open(pname + '.txt', 'w') as f:
        f.write(fit_report(result_static_displ))
    with open(pname + '.pkl', 'wb') as f:
        pickle.dump(result_static_displ, f)
    fig, axs = plot_static_displ_to_mod_spike(
        result_static_displ.params, lmpars, stim,
        animal=animal, rec_dict=rec_dict)
    fig.savefig(pname + '.png')
    plt.close(fig)


def plot_static_displ_to_mod_spike(lmdispl, lmpars, stim, plot_kws={},
                                   roll=True, animal=None, rec_dict=None,
                                   fig=None, axs=None):
    data_dicts = get_data_dicts(stim, lmdispl['static_displ'].value, animal,
                                rec_dict)
    fig, axs = plot_single_fit(
        lmpars, roll=roll, plot_kws=plot_kws, fig=fig, axs=axs,
        **data_dicts['fit_data_dict'])
    return fig, axs


def static_displ_to_residual(lmdispl, lmpars, stim,
                             animal=None, rec_dict=None):
    data_dicts = get_data_dicts(stim, lmdispl['static_displ'].value,
                                animal, rec_dict)
    return get_single_residual(lmpars, **data_dicts['fit_data_dict'])


def fit_static_displ(lmdispl, lmpars, stim, animal=None, rec_dict=None):
    result = minimize(static_displ_to_residual,
                      lmdispl, args=(lmpars, stim, animal, rec_dict),
                      epsfcn=1e-4)
    return result


class FitApproach():
    """
    If the parameters are stored in `data/fit`, then will load from file;
    Otherwise, fitting will be performed.
    """

    def __init__(self, lmpars_init, label=None):
        self.lmpars_init = lmpars_init
        if label is None:
            self.label = get_time_stamp()
        else:
            self.label = label
        # Load data
        self.load_rec_dicts()
        self.get_ref_fit()
        self.get_displ_fit()

    def get_ref_fit(self):
        fname = 'data/fit/ref_fit_%s.pkl' % self.label
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                self.ref_result = pickle.load(f)
                self.lmpars_fit = self.ref_result.params
        else:
            self.fit_ref()

    def get_displ_fit(self):
        # Create property if not there
        self.lmdispl_fit = {}
#        for animal in ANIMAL_LIST:
        for animal in ['Piezo2CONT']:
            self.lmdispl_fit[animal] = np.empty(STIM_NUM, dtype='object')
            for stim in range(STIM_NUM):
                fname = os.path.join('data', 'fit', self.label,
                                     '%s_%d.pkl' % (animal, stim))
                if os.path.exists(fname):
                    with open(fname, 'rb') as f:
                        lmdispl_fit = pickle.load(f).params
                        self.lmdispl_fit[animal][stim] = lmdispl_fit
                else:
                    self.fit_static_displ(animal, stim)

    def load_rec_dicts(self):
        self.rec_dicts = {animal: load_rec(animal) for animal in ANIMAL_LIST}

    def get_data_dicts(self, animal, stim, static_displ):
        data_dicts = get_data_dicts(stim, static_displ,
                                    rec_dict=self.rec_dicts[animal])
        return data_dicts

    def fit_ref(self, export=True):
        data_dicts = self.get_data_dicts(REF_ANIMAL, REF_STIM, REF_DISPL)
        self.ref_result = fit_single_rec(self.lmpars_init,
                                         data_dicts['fit_data_dict'])
        self.lmpars_fit = self.ref_result.params
        if export:
            export_ref_fit(self.ref_result, data_dicts['fit_data_dict'],
                           label_str=self.label)

    def fit_static_displ(self, animal, stim, export=True):
        lmdispl_init = Parameters()
        if animal == REF_ANIMAL and stim == REF_STIM:
            lmdispl_init.add('static_displ', value=REF_DISPL)
            lmdispl_fit = lmdispl_init
        else:
            lmdispl_init.add('static_displ', value=.5, min=0, max=.6,
                             vary=True)
            result_static_displ = fit_static_displ(
                lmdispl_init, self.lmpars_fit, stim,
                rec_dict=self.rec_dicts[animal])
            lmdispl_fit = result_static_displ.params
            if export:
                sub_folder = 'data/fit/%s' % self.label
                pname = os.path.join(sub_folder, '%s_%d' % (animal, stim))
                if not os.path.exists(sub_folder):
                    os.mkdir(sub_folder)
                export_displ_fit(result_static_displ, self.lmpars_fit, stim,
                                 pname, rec_dict=self.rec_dicts[animal])
        self.lmdispl_fit[animal][stim] = lmdispl_fit

    def plot_all_displ(self, animal):
        fig, axs = plt.subplots(2, 1, figsize=(3.5, 6))
        for stim in range(STIM_NUM):
#            alpha = 1 - stim * .4
            alpha = 1
            label = str(stim)
            color = ['k', 'r', 'g'][stim]
            plot_kws = {'alpha': alpha, 'color': color, 'label': label}
            lmdispl = self.lmdispl_fit[animal][stim]
            fig, axs = plot_static_displ_to_mod_spike(
                lmdispl, self.lmpars_fit, stim,
                plot_kws=plot_kws, rec_dict=self.rec_dicts[animal],
                fig=fig, axs=axs, roll=False)
        for axes in axs:
            axes.set_ylim(top=180)
        return fig, axs


if __name__ == '__main__':
    pass
    # %%
    fitApproach_dict = {}
#    for approach, lmpars_init in lmpars_init_dict.items():
    for approach, lmpars_init in lmpars_init_dict.items():
        if approach in ['t3f123', 't2f12']:
            lmpars_init = lmpars_init_dict[approach]
            fitApproach = FitApproach(lmpars_init, approach)
            fitApproach_dict[approach] = fitApproach
            fig, axs = fitApproach.plot_all_displ(REF_ANIMAL)
            fig.savefig('./data/fit/%s/Piezo2CONT.png' % approach)
    # %% Playing with Approach t3f123

