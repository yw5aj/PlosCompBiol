# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 10:56:20 2014

@author: Lindsay

This module contains all the constants needed in the LIF Model calculation.
"""

import numpy as np

LIF_RESOLUTION = 0.1  # in msec
DURATION = 5000  # in msec
REFRACTORY_PERIOD = 1  # in msec
MC_GROUPS = np.array([8, 5, 3, 1])

# %% LIF_PARAMS:
#    threshold (mV)
#    membrane capacitance (cm, in pF)
#    membrane resistance (rm, in Gohm)
LIF_PARAMS = np.array([30, 30, 1.667])

# Recording constants
FS = 16  # kHz
ANIMAL_LIST = ['Piezo2CONT', 'Piezo2CKO', 'Atoh1CKO']
MAT_FNAME_DICT = {
    'Piezo2CONT': '2013-12-07-01Piezo2CONT_calibrated.mat',
    'Piezo2CKO': '2013-12-13-02Piezo2CKO_calibrated.mat',
    'Atoh1CKO': '2013-10-16-01Atoh1CKO_calibrated.mat'}
STIM_LIST_DICT = {
    'Piezo2CONT': [(101, 2), (101, 1), (101, 3)],
    'Piezo2CKO': [(201, 2), (201, 7), (201, 4)],
    'Atoh1CKO': [(101, 2), (101, 1), (101, 5)]}
STIM_NUM = len(next(iter(STIM_LIST_DICT.values())))
