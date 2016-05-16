# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:13:46 2015

@author: Lindsay

This module plots sample Merkel cell potential and neuron current traces from
recordings of Adrienne Dubin, as well as their fitted traces.
"""

import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(1, 2, figsize=(6, 2.9))
for axes_id, axes in enumerate(axs.ravel()):
    axes.text(-.2, 1.04, chr(65+axes_id), transform=axes.transAxes,
              fontsize=12, fontweight='bold', va='top')
axs0 = axs[0]
axs1 = axs[1]


# %% Plot 1: Merkel cell potential recordings
mc_rcd = np.loadtxt('./data/Vol_614D 028 mechano in CC.csv', delimiter=',')
mc_disp = np.loadtxt('./data/Disp_614D 028 mechano in CC.csv', delimiter=',')
mc_fit_time = np.loadtxt('./data/adrienne_voltage_fit_time.csv', delimiter=',')
mc_fit = np.loadtxt('./data/adrienne_voltage_fit.csv', delimiter=',')
mc_pot_rcd = mc_rcd[:, 6]
mc_disp_rcd = mc_disp[:, 6]
axs0.plot(mc_rcd[:, 0], mc_pot_rcd, color='0.7')
axs0.plot(mc_fit_time, mc_fit, color='0')
axs0.set_xlim(0, 0.25)
axs0.set_ylim(-80, 0)
axs0.set_yticks([-80, -60, -40, -20, 0])
axs0.set_xlabel('Time (sec)')
axs0.set_ylabel('Voltage (mV)')


# %% Plot 2: Neuron current recordings
nr_rcd = np.loadtxt('./data/Cur_4092014P102RA_2dprocesses.csv', delimiter=',')
nr_disp = np.loadtxt('./data/Disp_4092014P102RA_2dprocesses.csv',
                     delimiter=',')
nr_fit_time = np.loadtxt('./data/adrienne_current_fit_time.csv', delimiter=',')
nr_fit = np.loadtxt('./data/adrienne_current_fit.csv', delimiter=',')
nr_current_rcd = nr_rcd[:, 8]
nr_disp_rcd = nr_disp[:, 8]
axs1.plot(nr_rcd[:, 0], nr_current_rcd, color='0.7')
axs1.plot(nr_fit_time, nr_fit, color='0')
axs1.set_xlim(0.07, 0.12)
axs1.set_ylim(-1200, -600)
axs1.set_yticks([-1200, -1000, -800, -600])
axs1.set_xlabel('Time (sec)')
axs1.set_ylabel('Current (pA)')


# %%
fig.tight_layout()
fig.savefig('./F12 adrienne recordings/F12 adrienne recordings.tif', dpi=600)
