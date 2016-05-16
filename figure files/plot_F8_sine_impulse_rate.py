# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 10:29:48 2015

@author: Lindsay

This module plots impulse rate comparisons between model and two recordings
from Johnson and Bensmaia under sine stimulation, 5, 10, and 20 Hz.
"""

import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(1, 3, figsize=(7.5, 2.8))
for axes_id, axes in enumerate(axs.ravel()):
    axes.text(-.1, 1.05, chr(65+axes_id), transform=axes.transAxes,
              fontsize=12, fontweight='bold', va='top')
ax0 = axs[0]
ax1 = axs[1]
ax2 = axs[2]
sin_stress = np.genfromtxt('./data/sin_stress.csv', delimiter=',')


# %% Plot 1: 5Hz
jon_5hz_stim, jon_5hz_rate = np.genfromtxt('./data/sin_Johnson_5Hz.csv',
                                           delimiter=',').T
mod_5hz_rate = np.genfromtxt('./data/sin_spike_per_cyc_5hz.csv', delimiter=',')
jon_5hz_norm = jon_5hz_stim[7]
mod_5hz_norm = 650
jon_5hz_stim = jon_5hz_stim/jon_5hz_norm
sin_stress_5hz = sin_stress/mod_5hz_norm
ax0.plot(jon_5hz_stim, jon_5hz_rate, marker='o', linestyle='None', color='0')
ax0.plot(sin_stress_5hz, mod_5hz_rate, marker='+', color='0')
ax0.set_xlim(0, 3)
ax0.set_xticks([0, 1, 2])
ax0.set_ylim(0, 2.1)
ax0.set_yticks([0, 1, 2])
ax0.set_ylabel('Impluse rate (impulses/cycle)', fontsize=11)
ax0.text(0.5, 1.5, '5 Hz', fontsize=12)


# %% Plot 2: 10Hz
jon_10hz_stim, jon_10hz_rate = np.genfromtxt('./data/sin_Johnson_10Hz.csv',
                                             delimiter=',').T
ben_10hz_stim, ben_10hz_rate = np.genfromtxt('./data/sin_Bensmaia_10Hz.csv',
                                             delimiter=',').T
mod_10hz_rate = np.genfromtxt('./data/sin_spike_per_cyc_10hz.csv',
                              delimiter=',')
jon_10hz_norm = jon_10hz_stim[6]
ben_10hz_norm = ben_10hz_stim[8]
mod_10hz_norm = 600
jon_10hz_stim = jon_10hz_stim/jon_10hz_norm
ben_10hz_stim = ben_10hz_stim/jon_10hz_norm
sin_stress_10hz = sin_stress/mod_10hz_norm
ax1.plot(jon_10hz_stim, jon_10hz_rate, marker='o',
         label='Freeman & Johnson, 1982', linestyle='None', color='0')
ax1.plot(ben_10hz_stim, ben_10hz_rate, marker='^', label='Kim et al., 2010',
         linestyle='None', color='0')
ax1.set_xlim(0, 3)
ax1.set_xticks([0, 1, 2])
ax1.set_ylim(0, 2.1)
ax1.set_yticks([0, 1, 2])
ax1.plot(sin_stress_10hz, mod_10hz_rate, label='model', marker='+', color='0')
ax1.set_xlabel('Normalized amplitude', fontsize=11)
ax1.text(0.5, 1.5,'10 Hz', fontsize=12)


# %% Plot 3: 20Hz
jon_20hz_stim, jon_20hz_rate = np.genfromtxt('./data/sin_Johnson_20Hz.csv',
                                             delimiter=',').T
mod_20hz_rate = np.genfromtxt('./data/sin_spike_per_cyc_20hz.csv',
                              delimiter=',')
jon_20hz_norm = jon_20hz_stim[8]
mod_20hz_norm = 700
jon_20hz_stim = jon_20hz_stim/jon_20hz_norm
sin_stress_20hz = sin_stress/mod_20hz_norm
ax2.plot(jon_20hz_stim, jon_20hz_rate, marker='o',
         label='Freeman & Johnson, 1982', linestyle='None', color='0')
ax2.set_xlim(0, 3)
ax2.set_xticks([0, 1, 2])
ax2.set_ylim(0, 2.1)
ax2.set_yticks([0, 1, 2])
ax2.plot(5, 5, marker='^', linestyle='None', label='Kim et al., 2010',
         color='0')  # only in the purpose of putting text on this subplot
ax2.plot(sin_stress_20hz, mod_20hz_rate, marker='+', label='model', color='0')
ax2.text(0.5, 1.5, '20 Hz', fontsize=12)
leg2 = ax2.legend(loc=4, prop={'size': 6})


# %%
fig.tight_layout()
fig.savefig('./F8 sine impulse rate/F8 sine impulse rate.tif', dpi=600)
