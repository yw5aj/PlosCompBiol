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
