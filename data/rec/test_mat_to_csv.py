# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 17:15:54 2016

@author: Administrator
"""

import numpy as np
import pytest
import os

from mat_to_csv import (export_all_to_csv, animal_list, mat_fname_dict,
                        stim_list_dict, fs)


def test_compare_kate_data():
    trace_arr_dict_dict = export_all_to_csv(animal_list, mat_fname_dict,
                                            stim_list_dict, fs)
    key_mapping = {'force': 'force', 'disp': 'displ', 'cs': 'spike'}
    for fname in os.listdir('kate'):
        if 'Atoh1CONT' not in fname:
            animal, key_kate, stim = fname[:-4].split('_')
            stim = int(stim)
            key = key_mapping[key_kate]
            kate_data = np.genfromtxt(os.path.join('kate', fname),
                                      delimiter=',')
            assert np.allclose(kate_data.T[0][:100],
                               trace_arr_dict_dict[animal][
                                   key].T[stim - 1][:100])
    return


if __name__ == '__main__':
    pytest.main([os.path.basename(__file__)])
