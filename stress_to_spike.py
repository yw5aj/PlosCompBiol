# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 13:46:55 2014

@author: Lindsay

This module takes stress files into a generator function to generate groups of
generator currents, and then call the LIF Model to output spikes.
"""

import setpyximport
from cy_lif_model import get_spikes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model_constants import (LIF_RESOLUTION, DURATION, MC_GROUPS)
from gen_function import stress_to_current


# %% Convert fine stress to current in all groups of Merkel Cells
def stress_to_group_current(fine_time, fine_stress, groups, **params):
    """
    Convert fine stress to current in all groups of Merkel Cells.

    Parameters
    ----------
    fine_time : 1xM array
        Interpolated time.
    fine_stress : 1xM array
        Interpolated stress.
    groups : 1xN array
        Groups of Merkel cells.
        Example: [8, 5, 3, 1] has 4 groups, 8, 5, 3, and 1 cell in each group.

    Returns
    -------
    group_gen_current : MxN array
        Generator current generated by stress in groups.
    """
    single_gen_current = stress_to_current(
        fine_time, fine_stress, **params).sum(axis=1)
    group_gen_current = np.multiply(single_gen_current[None].T, groups)
    return group_gen_current


# %% Transfer spike times to spike trace
def spike_time_to_trace(spike_time):
    """
    Transfer spike time to spike trace.

    Parameters
    ----------
    spike_time : 1d-array
        The time points of all the spikes in msec.

    Returns
    -------
    spike_trace : 2d-array
        [0] = time increments in the total duration.
        [1] = 1 if there is a spike at this time, 0 otherwise.
    """
    spike_trace = np.zeros([int(DURATION/LIF_RESOLUTION)+1, 2])
    spike_trace[:, 0] = np.arange(0, DURATION+LIF_RESOLUTION, LIF_RESOLUTION)
    spike_trace[(spike_time / LIF_RESOLUTION).astype(int), 1] = 1
    return spike_trace


def spike_time_to_fr_roll(spike_time, window_size):
    spike_time = np.array(spike_time)
    isi_inst = np.r_[np.inf, np.diff(spike_time)]
    isi_roll = pd.Series(isi_inst).rolling(window=window_size).mean()
    isi_roll[np.isnan(isi_roll)] = np.inf
    fr_roll = 1 / isi_roll
    return fr_roll


def spike_time_to_fr_inst(spike_time):
    return spike_time_to_fr_roll(spike_time, 1)


def stress_to_fr_inst(fine_time, fine_stress, groups, **params):
    group_gen_current = stress_to_group_current(fine_time, fine_stress,
                                                groups, **params)
    spike_time = get_spikes(group_gen_current)
    return np.array(spike_time), spike_time_to_fr_inst(spike_time)


# %% Main function
if __name__ == '__main__':
    pass
