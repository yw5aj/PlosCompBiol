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
def stress_to_current(fine_time, fine_stress, tau1, tau2, k1, mode):
    """
    Generate current from the stress of a single Merkel cell.

    Parameters
    ----------
    fine_time : 1d-array
        Timecourse of the indentation process.
    fine_stress : 1d-array
        Stress from a single Merkel cell.
    tau1 : double
        Decay time constant for the neurite mechanism.
    tau2 : double
        Decay time constant for the Merkel cell mechanism.
    k1 : double
        Peak/steady ratio for the Merkel cell mechanism.
    mode : string
        Decide what current is output.
        "gen" = output total generator current
        "mc" = output Merkel cell mechanism current
        "nr" = output neurite mechanism current

    Returns
    -------
    gen_current : 1d-array
        Generator current from the generator function.
    """
    # Neuron current parameter
#    a = 30e-12  # in Pa/mA, basic
#    a = 2.1e-12  # in Pa/mA, new approach 1
#    a = 4e-12  # in Pa/mA, new approach 2
    a = 1.5e-12  # in Pa/mA, new approach 3
    # Merkel cell voltage parameter
#    b = 1.875e-12  # in Pa/mA
#    b = 1e-12  # in Pa/mA, new approach 1
#    b = 1e-12  # in Pa/mA, new approach 2
    b = 1.5e-12  # in Pa/mA, new approach 3
    c = 2e-12  # in Pa/mA, new approach 2
    tau3 = 0.5  # in sec, new approach 2
    k2 = 1-k1
    k3 = 0.9
    k4 = 1 - k3
    dt = LIF_RESOLUTION * 1e-3  # in sec
    dsdt = np.diff(fine_stress) / dt
    dsdt = np.r_[0., dsdt]
    # The K function
    mc_k = b * (k1 * np.exp(-fine_time / tau2) + k2)
#    nr_k = a * np.exp(-fine_time / tau1)
    nr_k = a * (k3 * np.exp(-fine_time / tau1) + k4)  # New approach 3 only
    ia_k = c * np.exp(-fine_time / tau3)
    # Two parts of currents
    mc_current = np.convolve(mc_k, dsdt, mode='full')[:fine_time.shape[0]] \
        * dt
    nr_current = np.convolve(nr_k, dsdt, mode='full')[:fine_time.shape[0]] \
        * dt
    ia_current = np.convolve(ia_k, dsdt, mode='full')[:fine_time.shape[0]] \
        * dt
    nr_current[nr_current < 0] = 0
    mc_current[mc_current < 0] = 0
    ia_current[ia_current < 0] = 0
    gen_current = mc_current + nr_current
    gen_current = gen_current + ia_current  # new approach 2 only
    nr_current = nr_current + ia_current  # new approach 2 only
    if mode == "nr":
        return nr_current
    elif mode == "mc":
        return mc_current
    elif mode == "gen":
        return gen_current


def stress_to_current_new(fine_time, fine_stress, tau_nr, tau_mc, tau_ad,
                          k_nr, k_nr_1, k_mc, k_mc_1, k_ad, k_ad_1):
    """
    Generate current from the stress of a single Merkel cell.

    Parameters
    ----------
    fine_time : 1d-array
        Timecourse of the indentation process.
    fine_stress : 1d-array
        Stress from a single Merkel cell.
    tau_nr, tau_mc, tau_ad : double
        Decay time constant for the neurite/Merkel cell/adaptation mechanism.
    k_nr, k_mc, k_ad : double
        Peak/steady ratio for the neurite/Merkel cell/adaptation mechanism.
    k_mc_1, k_nr_1, k_ad_1 : double
        1st sub-component of the `k_mc`, `k_nr`, and `k_ad`.

    Returns
    -------
    current_dict : dict
        Generator current from the generator function.
        'gen' : output total generator current
        'mc' : output Merkel cell mechanism current
        'nr' : output neurite mechanism current
        'ad' : output adaptation mechanism current
    """
    ds = np.r_[0, np.diff(fine_stress)]

    def get_sub_current(k, k_sub_list, tau):
        k_func = k * (k_sub_list[0] * np.exp(-fine_time / tau) + k_sub_list[1])
        current = np.convolve(k_func, ds, mode='full')[:fine_time.shape[0]]
        current[current < 0] = 0
        return current
    # Gen current
    mc_current = get_sub_current(k_mc, [k_mc_1, 1 - k_mc_1], tau_mc)
    nr_current = get_sub_current(k_nr, [k_nr_1, 1 - k_nr_1], tau_nr)
    ad_current = get_sub_current(k_ad, [k_ad_1, 1 - k_ad_1], tau_ad)
    gen_current = mc_current + nr_current + ad_current
    current_dict = {}
    key = None  # Such that locals() won't change size during runtime
    for key in locals():
        if '_current' in key:
            current_dict[key] = locals()[key]
    return current_dict


# %% Main function
if __name__ == '__main__':
    pass
