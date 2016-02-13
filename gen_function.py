# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:34:37 2014

@author: Lindsay

This script is used to construct and test the generator function for a single
Merkel cell-neurite connection.
The input of the function is stress, and output is current.
"""

import numpy as np
from scipy.interpolate import interp1d
from model_constants import LIF_RESOLUTION


# %% Generate Fine stress
def prepare_stress(rough_time, rough_stress):
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
    fine_time = np.arange(0, rough_time[-1], LIF_RESOLUTION * 1e-3)
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
