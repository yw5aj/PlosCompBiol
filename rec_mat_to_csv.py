# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 17:15:54 2016

@author: Administrator
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


from model_constants import (FS, ANIMAL_LIST, MAT_FNAME_DICT, STIM_LIST_DICT,
                             DURATION)


def extract_trace_arr_dict(data_dict, stim_list, fs, duration):
    trace_arr_dict = {'force': [], 'displ': [], 'spike': [], 'time': []}
    for block_id, stim_id in stim_list:
        # Find the first spike index
        spike_time = data_dict['OUT_PUT_CS%d' % block_id].T[stim_id]
        start_index = spike_time.nonzero()[0][0]
        end_index = start_index + duration * fs
        data_range = range(start_index, end_index)
        # Spike times
        trace_arr_dict['spike'].append(spike_time[data_range])
        # Force in mN
        trace_arr_dict['force'].append(
            data_dict['COUT_PUT_F%d' % block_id].T[stim_id][data_range])
        # Displ in mm
        displ = data_dict['COUT_PUT_D%d' % block_id].T[stim_id][data_range]
        displ -= displ[0]
        trace_arr_dict['displ'].append(displ * 1e-3)
        # Time in ms
        trace_arr_dict['time'].append(
            np.arange(trace_arr_dict['force'][-1].size) / fs * 1e3)
    for key, item in trace_arr_dict.items():
        trace_arr_dict[key] = np.column_stack(item)
    return trace_arr_dict


def save_arr_dict_to_csv(arr_dict, animal):
    for key, item in arr_dict.items():
        fname = '%s_%s.csv' % (animal, key)
        pname = os.path.join('data', 'rec', fname)
        np.savetxt(pname, item, delimiter=',')


def export_trace_to_csv(animal_list, mat_fname_dict, stim_list_dict,
                        fs, duration):
    trace_arr_dict_dict = {animal: [] for animal in animal_list}
    for (i, animal) in enumerate(animal_list):
        mat_fname = mat_fname_dict[animal]
        stim_list = stim_list_dict[animal]
        data_dict = loadmat(os.path.join('data', 'rec', 'raw', mat_fname))
        trace_arr_dict = extract_trace_arr_dict(data_dict, stim_list,
                                                fs, duration)
        save_arr_dict_to_csv(trace_arr_dict, animal)
        trace_arr_dict_dict[animal] = trace_arr_dict
    return trace_arr_dict_dict


def plot_trace(trace_arr_dict_dict):
    for animal, trace_arr_dict in trace_arr_dict_dict.items():
        for key, trace_arr in trace_arr_dict.items():
            if key != 'time':
                fig, axs = plt.subplots()
                axs.plot(trace_arr)
                fig.savefig('./data/rec/plots/%s_%s.png' % (animal, key),
                            dpi=300)
                plt.close(fig)


if __name__ == '__main__':
    trace_arr_dict_dict = export_trace_to_csv(
        ANIMAL_LIST, MAT_FNAME_DICT, STIM_LIST_DICT, FS, DURATION)
    plot_trace(trace_arr_dict_dict)
