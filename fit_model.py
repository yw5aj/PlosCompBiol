# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:25:32 2016

@author: Administrator
"""

import numpy as np
import os
from multiprocessing import Pool
from collections import defaultdict
from itertools import combinations
from lmfit import minimize, fit_report, Parameters
import matplotlib.pyplot as plt
import pickle
import datetime
import re
import copy

from stress_to_spike import (stress_to_fr_inst, spike_time_to_fr_roll,
                             spike_time_to_fr_inst)
from model_constants import (MC_GROUPS, FS, ANIMAL_LIST, STIM_NUM,
                             REF_ANIMAL, REF_STIM_LIST, WINDOW, REF_DISPL,
                             COLOR_LIST, CKO_ANIMAL_LIST)
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
    def get_fname(animal, datatype):
        return os.path.join('data', 'rec', '%s_%s.csv' % (animal, datatype))
    fname_dict = {datatype: get_fname(animal, datatype)
                  for datatype in ['spike', 'displ']}
    displ_arr = np.genfromtxt(fname_dict['displ'], delimiter=',')
    static_displ_list = np.round(displ_arr[-1], 2).tolist()
    spike_arr = np.genfromtxt(fname_dict['spike'], delimiter=',')
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
        'static_displ_list': static_displ_list,
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


def get_data_dicts(stim, animal=None, rec_dict=None):
    if rec_dict is None:
        rec_dict = load_rec(animal)
    # Read recording data
    rec_fr_inst = rec_dict['fr_inst_list'][stim]
    rec_spike_time = rec_dict['spike_time_list'][stim]
    rec_fr_roll = rec_dict['fr_roll_list'][stim]
    static_displ = rec_dict['static_displ_list'][stim]
    rec_data_dict = {
        'rec_spike_time': rec_spike_time,
        'rec_fr_inst': rec_fr_inst,
        'rec_fr_roll': rec_fr_roll}
    # Read model data
    time, stress = get_interp_stress(static_displ)
    max_time = rec_dict['max_time_list'][stim]
    stretch_coeff = 1 + 0.25 * static_displ / REF_DISPL
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


def fit_single_rec_mp(args):
    return fit_single_rec(*args)


def plot_single_fit(lmpars_fit, groups, time, stress,
                    rec_spike_time, plot_kws={}, roll=True,
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


def get_mean_lmpar(lmpar_list):
    if isinstance(lmpar_list, Parameters):
        return lmpar_list
    lmpar_dict_list = [lmpar.valuesdict() for lmpar in lmpar_list]
    all_param_dict = defaultdict(list)
    for lmpar_dict in lmpar_dict_list:
        for key, value in lmpar_dict.items():
            all_param_dict[key].append(value)
    mean_lmpar = copy.deepcopy(lmpar_list[0])
    for key, value in all_param_dict.items():
        mean_lmpar[key].set(value=np.mean(value))
    return mean_lmpar


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
        self.load_data_dicts_dicts()
        self.get_ref_fit()

    def get_ref_fit(self):
        pname = os.path.join('data', 'fit', self.label)
        if os.path.exists(pname):
            with open(os.path.join(pname, 'ref_mean_lmpars.pkl'), 'rb') as f:
                self.ref_mean_lmpars = pickle.load(f)
            self.ref_result_list = []
            for fname in os.listdir(pname):
                if fname.startswith('ref_fit') and fname.endswith('.pkl'):
                    with open(os.path.join(pname, fname), 'rb') as f:
                        self.ref_result_list.append(pickle.load(f))
        else:
            self.fit_ref()

    def load_rec_dicts(self):
        self.rec_dicts = {animal: load_rec(animal) for animal in ANIMAL_LIST}

    def get_data_dicts(self, animal, stim):
        data_dicts = get_data_dicts(stim, rec_dict=self.rec_dicts[animal])
        return data_dicts

    def load_data_dicts_dicts(self):
        self.data_dicts_dicts = {}
        for animal in ANIMAL_LIST:
            self.data_dicts_dicts[animal] = {}
            for stim in range(STIM_NUM):
                self.data_dicts_dicts[animal][stim] = self.get_data_dicts(
                    animal, stim)

    def fit_ref(self, export=True):
        data_dicts_dict = self.data_dicts_dicts[REF_ANIMAL]
        # Prepare data for multiprocessing
        fit_mp_list = []
        for stim in REF_STIM_LIST:
            fit_mp_list.append([self.lmpars_init,
                                data_dicts_dict[stim]['fit_data_dict']])
        with Pool(5) as p:
            self.ref_result_list = p.map(fit_single_rec_mp, fit_mp_list)
        lmpar_list = [result.params for result in self.ref_result_list]
        self.ref_mean_lmpars = get_mean_lmpar(lmpar_list)
        # Plot the fit for multiple displacements
        if export:
            self.export_ref_fit()

    def export_ref_fit(self):
        pname = os.path.join('data', 'fit', self.label)
        os.mkdir(pname)
        for stim, result in zip(REF_STIM_LIST, self.ref_result_list):
            fname_report = 'ref_fit_%d.txt' % stim
            fname_pickle = 'ref_fit_%d.pkl' % stim
            with open(os.path.join(pname, fname_report), 'w') as f:
                f.write(fit_report(result))
            with open(os.path.join(pname, fname_pickle), 'wb') as f:
                pickle.dump(result, f)
        with open(os.path.join(pname, 'ref_mean_lmpars.pkl'), 'wb') as f:
            pickle.dump(self.ref_mean_lmpars, f)
        # Plot
        fig, axs = self.plot_ref_fit(roll=True)
        fig.savefig(os.path.join(pname, 'ref_fit_roll.png'), dpi=300)
        plt.close(fig)
        fig, axs = self.plot_ref_fit(roll=False)
        fig.savefig(os.path.join(pname, 'ref_fit_inst.png'), dpi=300)
        plt.close(fig)

    def plot_ref_fit(self, roll=True):
        fig, axs = plt.subplots(2, 1, figsize=(3.5, 6))
        for stim, ref_result in zip(REF_STIM_LIST, self.ref_result_list):
            lmpars_fit = ref_result.params
            color = COLOR_LIST[stim]
            plot_single_fit(
                lmpars_fit, fig=fig, axs=axs[0], roll=roll,
                plot_kws={'color': color},
                **self.data_dicts_dicts[REF_ANIMAL][stim]['fit_data_dict'])
            plot_single_fit(
                self.ref_mean_lmpars, fig=fig, axs=axs[1], roll=roll,
                plot_kws={'color': color},
                **self.data_dicts_dicts[REF_ANIMAL][stim]['fit_data_dict'])
        axs[0].set_title('Individual fitting parameters')
        axs[1].set_title('Using the average fitting parameter')
        fig.tight_layout()
        return fig, axs

    def plot_cko(self, animal, k_out_list):
        label = str(k_out_list).replace('\'', '').replace('(', '').replace(
            ')', '').replace(',', '').replace(' ', '')
        lmpars_piezo2cko = copy.deepcopy(self.ref_mean_lmpars)
        for k_out in k_out_list:
            lmpars_piezo2cko[k_out].set(value=0)
        fig, axs = plt.subplots()
        for stim in range(STIM_NUM):
            color = COLOR_LIST[stim]
            plot_single_fit(
                lmpars_piezo2cko, fig=fig, axs=axs, roll=False,
                plot_kws={'color': color},
                **self.data_dicts_dicts[animal][stim]['fit_data_dict'])
        axs.set_title('Eliminating %s for %s' % (label, animal))
        fig.tight_layout()
        fig.savefig('./data/output/%s_%s.png' % (animal, label))
        plt.close(fig)

    def plot_atoh1cko(self):
        pass


if __name__ == '__main__':
    pass
    # %%
    fitApproach_dict = {}
    for approach, lmpars_init in lmpars_init_dict.items():
        lmpars_init = lmpars_init_dict[approach]
        fitApproach = FitApproach(lmpars_init, approach)
        fitApproach_dict[approach] = fitApproach
    # %% Play with t3f123
    fitApproach = fitApproach_dict['t3f123']
    k_list = [key for key in fitApproach.ref_mean_lmpars.keys()
              if key.startswith('k')]
    k_combination_list = []
    for i in range(1, len(k_list)):
        k_combination_list.extend(list(combinations(k_list, i)))
    for animal in CKO_ANIMAL_LIST:
        for k_out_tuple in k_combination_list:
            try:
                fitApproach.plot_cko(animal, k_out_tuple)
            except ValueError:
                pass
