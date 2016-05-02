# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:34:37 2014

@author: Lindsay

This script is used to construct and test the generator function for a single
Merkel cell-neurite connection.
The input of the function is stress, and output is current.
"""

import numpy as np
import os
from scipy.interpolate import interp1d
from model_constants import LIF_RESOLUTION, FE_NUM


# %% Generate Fine stress
def get_single_fine_stress(fe_id):
    rough_time, rough_displ, rough_stress = get_single_rough_fea(fe_id)
    fine_time, fine_stress = interpolate_stress(rough_time, rough_stress)
    fine_time, fine_displ = interpolate_stress(rough_time, rough_displ)
    return fine_time, fine_displ, fine_stress


def get_single_rough_fea(fe_id):
    fname = 'TipOneFive%02dDispl.csv' % fe_id
    pname = os.path.join('data', 'fem', fname)
    time, force, displ, stress, strain, sener = np.genfromtxt(
        pname, delimiter=',').T
    time *= 1e3  # sec to msec
    stress *= 1e-3  # Pa to kPa
    displ *= 1e3  # m to mm
    return time, displ, stress


def get_interp_stress(static_displ):
    """
    Get interpolated stress from FE model. Will do linear extrapolation.

    Parameters
    ----------
    static_displ : float
        The steady-state displ to scale the stress.

    Returns
    -------
    time : 1xN array
        Time array corresponding with the stress.
    stress : 1xN array
        Stress array.
    """
    time, static_displ_arr, stress_table = get_stress_table()
    stress = np.empty_like(time)
    # Use numpy for performance reasons
    if static_displ <= static_displ_arr.max():
        for i in range(stress.size):
            stress[i] = np.interp(static_displ, static_displ_arr,
                                  stress_table[i])
    else:
        for i in range(stress.size):
            interp_func = interp1d(static_displ_arr, stress_table[i],
                                   kind='linear', fill_value='extrapolate')
            stress[i] = interp_func(static_displ)
    return time, stress


def get_stress_table(fe_num=FE_NUM):
    """
    Parameters
    ----------
    fe_num : int
        Total number of fe runs.

    Returns
    -------
    time : 1xN array
        Time points.

    static_displ_arr : 1xM array
        A list for static displacements.

    stress_table : NxM array
        A table with columns as stress traces for each displacement.
    """
    time_list, displ_list, stress_list = [], [], []
    for fe_id in range(fe_num):
        time, displ, stress = get_single_fine_stress(fe_id)
        time_list.append(time)
        displ_list.append(displ)
        stress_list.append(stress)
    size_min = np.min([time.size for time in time_list])
    stress_table = np.column_stack(
        [np.zeros(size_min)] + [stress[:size_min] for stress in stress_list])
    static_displ_arr = np.r_[0, [displ[-1] for displ in displ_list]]
    return time[:size_min], static_displ_arr, stress_table


def interpolate_stress(rough_time, rough_stress):
    """
    Generate fine stress from rough stress using Linear Spline.

    Parameters
    ----------
    rough_time : 1d-array
        Timecourse from input stress file.
    rough_stress : 1d-array
        Stress from input stress file.

    Returns
    -------
    output_time_stress : 2d-array
        Fine time and Fine stress from Linear Spline of rough time and stress.
    """
    fine_time = np.arange(0, rough_time[-1], LIF_RESOLUTION)
    fine_spline = interp1d(rough_time, rough_stress, kind='linear')
    fine_stress = fine_spline(fine_time)
    return fine_time, fine_stress


# %% Convert stress to current
def stress_to_current(fine_time, fine_stress, tau_arr, k_arr):
    """
    Generate current from the stress of a single Merkel cell.

    Parameters
    ----------
    fine_time : 1xM array
        Time array of the indentation process.
    fine_stress : 1xM array
        Stress from a single Merkel cell.
    tau_arr : 1xN array
        Decay time constant for different adaptation mechanisms:
            tau_0, tau_1., ...., tau_inf
    k_arr : 1xN array
        Peak/steady ratio for different adaptation mechanisms.

    Returns
    -------
    current_arr : MxN array
        Generator current array from the generator function;
        each column represent one component.
    """
    ds = np.r_[0, np.diff(fine_stress)]
    k_func_arr = k_arr * np.exp(np.divide(-fine_time[None].T, tau_arr))
    current_arr = np.column_stack(
        [np.convolve(k_func_col, ds, mode='full')[:fine_time.size]
         for k_func_col in k_func_arr.T])
    current_arr[current_arr < 0] = 0
    return current_arr


# %% Main function
if __name__ == '__main__':
    pass
