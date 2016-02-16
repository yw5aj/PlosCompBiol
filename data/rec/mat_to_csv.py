# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 17:15:54 2016

@author: Administrator
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


fs = int(1.6e3)
animal_list = ['Piezo2CONT', 'Piezo2CKO', 'Atoh1CKO']
mat_fname_dict = {
    'Piezo2CONT': '2013-12-07-01Piezo2CONT_calibrated.mat',
    'Piezo2CKO': '2013-12-13-02Piezo2CKO_calibrated.mat',
    'Atoh1CKO': '2013-10-16-01Atoh1CKO_calibrated.mat'}
stim_list_dict = {
    'Piezo2CONT': [(101, 2), (101, 1), (101, 3)],
    'Piezo2CKO': [(201, 2), (201, 7), (201, 4)],
    'Atoh1CKO': [(101, 2), (101, 1), (101, 5)]}


def extract_trace_arr_dict(data_dict, stim_list, fs):
    trace_arr_dict = {'force': [], 'displ': [], 'spike': [], 'time': []}
    for block_id, stim_id in stim_list:
        trace_arr_dict['force'].append(
            data_dict['COUT_PUT_F%d' % block_id].T[stim_id])
        trace_arr_dict['displ'].append(
            data_dict['COUT_PUT_D%d' % block_id].T[stim_id] * 1e-3)  # to mm
        trace_arr_dict['spike'].append(
            data_dict['OUT_PUT_CS%d' % block_id].T[stim_id])
        trace_arr_dict['time'].append(
            np.arange(trace_arr_dict['force'][-1].size) / fs * 1e3)  # to ms
    for key, item in trace_arr_dict.items():
        trace_arr_dict[key] = np.column_stack(item)
    return trace_arr_dict


def save_arr_dict_to_csv(arr_dict, animal):
    for key, item in arr_dict.items():
        fname = '%s_%s.csv' % (animal, key)
        np.savetxt(fname, item, delimiter=',')


def export_all_to_csv(animal_list, mat_fname_dict, stim_list_dict, fs):
    trace_arr_dict_dict = {animal: [] for animal in animal_list}
    for (i, animal) in enumerate(animal_list):
        mat_fname = mat_fname_dict[animal]
        stim_list = stim_list_dict[animal]
        data_dict = loadmat(os.path.join('./raw', mat_fname))
        trace_arr_dict = extract_trace_arr_dict(data_dict, stim_list, fs)
        save_arr_dict_to_csv(trace_arr_dict, animal)
        trace_arr_dict_dict[animal] = trace_arr_dict
    return trace_arr_dict_dict


def plot_all(trace_arr_dict_dict):
    for animal, trace_arr_dict in trace_arr_dict_dict.items():
        for key, trace_arr in trace_arr_dict.items():
            if key != 'time':
                fig, axs = plt.subplots()
                axs.plot(trace_arr)
                fig.savefig('./plots/%s_%s.png' % (animal, key), dpi=300)
                plt.close(fig)


if __name__ == '__main__':
    trace_arr_dict_dict = export_all_to_csv(
        animal_list, mat_fname_dict, stim_list_dict, fs)
    plot_all(trace_arr_dict_dict)
