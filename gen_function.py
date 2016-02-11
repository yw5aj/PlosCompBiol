# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:34:37 2014

@author: Lindsay

This script is used to construct and test the generator function for a single
Merkel cell-neurite connection.
The input of the function is stress, and output is current.
"""

import numpy as np
import matplotlib.pyplot as plt
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
    a = 4e-12  # in Pa/mA, new approach 2
#    a = 1.5e-12  # in Pa/mA, new approach 3
    # Merkel cell voltage parameter
#    b = 1.875e-12  # in Pa/mA
#    b = 1e-12  # in Pa/mA, new approach 1
    b = 1e-12  # in Pa/mA, new approach 2
#    b = 1.5e-12  # in Pa/mA, new approach 3
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
    nr_k = a * np.exp(-fine_time / tau1)
#    nr_k = a * (k3 * np.exp(-fine_time / tau1) + k4)  # New approach 3 only
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


# %% Main function
if __name__ == '__main__':
    rough_time = np.genfromtxt('./Shawn model data/dcon_disp2_time.csv',
                               delimiter=',')
    rough_stress = np.genfromtxt('./Shawn model data/'
                                 'dcon_disp2_stress.csv', delimiter=',')
    fine_time, fine_stress = prepare_stress(rough_time, rough_stress)
    # tau1, tau2, k1 values
    tau1_s = 0.003
    tau1_m = 0.008
    tau1_l = 0.013
    tau2_s = 0.05
    tau2_m = 0.2
    tau2_l = 0.35
    k1_s = 0.1
    k1_m = 0.5
    k1_l = 0.9
    tau1_ap1 = 0.5  # New approach 1
    tau2_ap1 = 1  # New approach 1
    k1_ap1 = 0.1  # New approach 1
    tau2_ap2 = 1  # New approach 2
    k1_ap2 = 0.1  # New approach 2
    tau1_ap3 = 0.5  # New approach 3
    tau2_ap3 = 1  # New approach 3
    k1_ap3 = 0.5  # New approach 3
    # %% Choose current with desired parameters
    # Standard parameters
#    gen_current = stress_to_current(fine_time, fine_stress, tau1_m, tau2_m, k1_m, 'gen')
#    nr_current = stress_to_current(fine_time, fine_stress, tau1_m, tau2_m, k1_m,  'nr')
#    mc_current = stress_to_current(fine_time, fine_stress, tau1_m, tau2_m, k1_m,  'mc')
#    # Change tau1
#    gen_current_tau1s = stress_to_current(fine_time, fine_stress, tau1_s, tau2_m, k1_m, 'gen')
#    gen_current_tau1l = stress_to_current(fine_time, fine_stress, tau1_l, tau2_m, k1_m, 'gen')
#    # Change tau2
#    gen_current_tau2s = stress_to_current(fine_time, fine_stress, tau1_m, tau2_s, k1_m, 'gen')
#    gen_current_tau2l = stress_to_current(fine_time, fine_stress, tau1_m, tau2_l, k1_m, 'gen')
#    # Change k1
#    gen_current_k1s = stress_to_current(fine_time, fine_stress, tau1_m, tau2_m, k1_s, 'gen')
#    gen_current_k1l = stress_to_current(fine_time, fine_stress, tau1_m, tau2_m, k1_l, 'gen')
#    # New approach 1: modify parameters
#    gen_current = stress_to_current(fine_time, fine_stress, tau1_ap1, tau2_ap1, k1_ap1, 'nr')
    # New approach 2: add tau 3 and c
#    gen_current = stress_to_current(fine_time, fine_stress, tau1_m, tau2_ap2, k1_ap2, 'gen')
    # New approach 2: add tau 3 and c
    gen_current = stress_to_current(fine_time, fine_stress, tau1_ap3, tau2_ap3, k1_ap3, 'gen')
    # %% Save files
#    np.savetxt("disp2_fine_time.csv", fine_time, delimiter=",")
#    np.savetxt("disp2_gen_current.csv", gen_current, delimiter=",")
#    np.savetxt("disp2_nr_current.csv", nr_current, delimiter=",")
#    np.savetxt("disp2_mc_current.csv", mc_current, delimiter=",")
#    np.savetxt("disp2_tau1s_gen_current.csv", gen_current_tau1s, delimiter=",")
#    np.savetxt("disp2_tau1l_gen_current.csv", gen_current_tau1l, delimiter=",")
#    np.savetxt("disp2_tau2s_gen_current.csv", gen_current_tau2s, delimiter=",")
#    np.savetxt("disp2_tau2l_gen_current.csv", gen_current_tau2l, delimiter=",")
#    np.savetxt("disp2_k1s_gen_current.csv", gen_current_k1s, delimiter=",")
#    np.savetxt("disp2_k1l_gen_current.csv", gen_current_k1l, delimiter=",")
    # %% Plot
    fig, axs = plt.subplots()
    d2_cur_t = np.genfromtxt('./figure files/data/disp2_fine_time.csv',
                             delimiter=',')
    d2_gen_cur = np.genfromtxt('./figure files/data/disp2_gen_current.csv',
                               delimiter=',')
    axs.plot(d2_cur_t, -d2_gen_cur*1e9, color='0')
    axs.plot(fine_time, -gen_current*1e9, color='0.5')
    axs.set_xlim(0, 5)
    axs.set_ylim(-15, 0)
    axs.set_yticks([-15, -10, -5, 0])
    axs.set_xlabel('Time (sec)')
    axs.set_ylabel('Current (pA)')
#    axs.plot(fine_time, -gen_current*1e9)  # in pA
