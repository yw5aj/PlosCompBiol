# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 22:27:25 2015

@author: Lindsay

This module plots sinusoidal stimulations with 5 and 20 Hz frequency,
outputting stress, current, and spikes.
"""

import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(2, 2, figsize=(6, 3.5))
for axes_id, axes in enumerate(axs.ravel()):
    axes.text(-.15, 1.1, chr(65+axes_id), transform=axes.transAxes,
              fontsize=12, fontweight='bold', va='top')
axs0 = axs[0, 0]
axs1 = axs[0, 1]
axs2 = axs[1, 0]
axs3 = axs[1, 1]


# %% Plot 1: 5Hz stress and spike
d1_5hz_t = np.genfromtxt('./data/stress650_sin_5hz_time.csv', delimiter=',')
d1_5hz_stress = np.genfromtxt('./data/stress650_sin_5hz_stress.csv',
                              delimiter=',')
d1_5hz_spike = np.genfromtxt('./data/stress650_sin_5hz_gen_spike.csv',
                             delimiter=',')
axs0.plot(d1_5hz_t, d1_5hz_stress/1000, color='0')
axs0.vlines(d1_5hz_spike, ymin=0, ymax=0.5, linewidth=0.5, color='0.5')
axs0.set_xlim(0, 0.8)
axs0.set_ylim(0, 1)
axs0.set_xlabel('Time (sec)')
axs0.set_ylabel('Stress (kPa)')


# %% Plot 2: 20hz stress and spike
d1_20hz_t = np.genfromtxt('./data/stress900_sin_20hz_time.csv', delimiter=',')
d1_20hz_stress = np.genfromtxt('./data/stress900_sin_20hz_stress.csv',
                               delimiter=',')
d1_20hz_spike = np.genfromtxt('./data/stress900_sin_20hz_gen_spike.csv',
                              delimiter=',')
axs1.plot(d1_20hz_t, d1_20hz_stress/1000, color='0')
axs1.vlines(d1_20hz_spike, ymin=0, ymax=0.5, linewidth=0.5, color='0.5')
axs1.set_xlim(0, 0.2)
axs1.set_ylim(0, 1)
axs1.set_xlabel('Time (sec)')
axs1.set_ylabel('Stress (kPa)')


# %% Plot 3: 5hz current
d1_5hz_cur = np.genfromtxt('./data/stress650_sin_5hz_gen_current.csv',
                           delimiter=',')
d1_5hz_cur = d1_5hz_cur[:, 0]
axs2.plot(d1_5hz_t, -d1_5hz_cur*1e9, color='0')
axs2.set_xlim(0, 0.8)
axs2.set_ylim(-20, 0)
axs2.set_xlabel('Time (sec)')
axs2.set_ylabel('Current (pA)')


# %% Plot 4: 20hz current
d1_20hz_cur = np.genfromtxt('./data/stress900_sin_20hz_gen_current.csv',
                            delimiter=',')
d1_20hz_cur = d1_20hz_cur[:,0]
axs3.plot(d1_20hz_t, -d1_20hz_cur*1e9, color='0')
axs3.set_xlim(0, 0.2)
axs3.set_ylim(-20, 0)
axs3.set_xlabel('Time (sec)')
axs3.set_ylabel('Current (pA)')


# %%
fig.tight_layout()
fig.savefig('./Unused figures/sine spikes.tif', dpi=600)
