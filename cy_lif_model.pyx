# -*- coding: utf-8 -*-
# cython: profile = False
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True


import numpy as np
cimport numpy as np
import model_constants
from libc.math cimport M_PI


#%% Define imported constants
cdef double [::1] LIF_PARAMS = model_constants.LIF_PARAMS
cdef unsigned int [::1] MC_GROUPS = model_constants.MC_GROUPS.astype(np.uint)
cdef double LIF_RESOLUTION = model_constants.LIF_RESOLUTION
cdef double DURATION = model_constants.DURATION
cdef double REFRACTORY_PERIOD = model_constants.REFRACTORY_PERIOD


# %% Calculate du/dt
cpdef double dudt(double pot, double current):
    """
    Calculate du/dt using:
    du/dt = current / cm - pot / (rm * cm)

    Parameters
    ----------
    pot : double
        Potential at one heminode at the timepoint.
    current : double
        Transduction current at the timepoint.

    Returns
    -------
    ut : double
        The value of du/dt.
    """
    cdef double cm = LIF_PARAMS[1]
    cdef double rm = LIF_PARAMS[2]
    cdef double ut = 0.
    ut = current / cm - pot / (rm * cm)
    return ut


# %% Runge-Kutta calculation
cpdef double [::1] runge_kutta(double [:, ::1] current, double start_time):
    """
    Use Runge-Kutta 4 to calculate the differential equation of LIF
    model and return timepoint and potential at the spike.

    Parameters
    ----------
    current : nd-array [timepoints, group number]
        Groups of transduction current in the total time duration.
        Group number = size of MC_GROUPS
    start_time : double
        The starting time for voltage calculation in the whole time duration.
        Voltage is 0 at the starting time.

    Returns
    -------
    output : array of 2 doubles
        [0] = timepoint when voltage at one heminode exceeds the threshold.
        [1] = largest voltage among all heminodes at the time.
    """
    cdef double threshold = LIF_PARAMS[0]
    cdef unsigned int mc_size = MC_GROUPS.shape[0]
    cdef double h = LIF_RESOLUTION
    cdef double time_span = DURATION - start_time
    cdef double temp_time = 0.
    cdef unsigned int current_index = <unsigned int>(start_time / h)
    cdef double [::1] each_pot = np.zeros(mc_size, dtype=np.double)
    cdef double max_pot = 0.
    cdef double mid_current = 0.
    cdef double [::1]  output = np.empty(2)
    cdef double k1 = 0.
    cdef double k2 = 0.
    cdef double k3 = 0.
    cdef double k4 = 0.
    cdef unsigned int i, j
    cdef unsigned int end_pot_argmax
    while temp_time <= time_span - h:
        for i in range(mc_size):
            mid_current = 0.5 * (current[current_index, i] +
                                 current[current_index + 1, i])
            k1 = dudt(each_pot[i], current[current_index, i])
            k2 = dudt(each_pot[i] + 0.5 * h * k1, mid_current)
            k3 = dudt(each_pot[i] + 0.5 * h * k2, mid_current)
            k4 = dudt(each_pot[i] + h * k3, current[current_index + 1, i])
            each_pot[i] = each_pot[i] + (h * (k1 + 2 * k2 + 2 * k3 + k4) / 6)
        max_pot = np.max(each_pot)
        if max_pot > threshold:
            break
        temp_time = temp_time + h
        current_index += 1
    output[0] = start_time + temp_time
    output[1] = max_pot
    return output


#%% Generate output spikes from input transduction currents
cpdef list get_spikes(double [:, ::1] current):
    """
    Generate output spikes from input transduction currents.

    Parameters
    ----------
    current : nd-array
        Grouped transduction current in the total time duration.
        Each column represents current from one group of Merkel cells.

    Returns
    -------
    spike_time : 1d_array
        Timepoints where there is a spike.
    """
    cdef double threshold = LIF_PARAMS[0]
    cdef double ini_time = 0.0
    cdef double integration_start = 0.0
    cdef double [::1] timestamp_finalpot = np.zeros(2)
    cdef unsigned int trace_length = <unsigned int>(DURATION // LIF_RESOLUTION
                                                    + 1)
    cdef list spike_time = []
    cdef double larger_time
    while ini_time <= DURATION:
        integration_start = ini_time
        timestamp_finalpot = runge_kutta(current[0:trace_length, :],
                                         integration_start)
        larger_time = max(timestamp_finalpot[0], ini_time + REFRACTORY_PERIOD)
        larger_time = round(larger_time / LIF_RESOLUTION) * LIF_RESOLUTION
        if timestamp_finalpot[1] > threshold:
            spike_time.append(larger_time)
        ini_time = LIF_RESOLUTION + larger_time
    return spike_time


# %% main function: test of LIF model
if __name__ == '__main__':
    current = np.ones(5/LIF_RESOLUTION+1) * 5e-10
    current = np.c_[current]
    start_time = 0
    mc_size = MC_GROUPS.shape[0]
    time_pot = runge_kutta(current, start_time)