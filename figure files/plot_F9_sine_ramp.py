# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 15:14:49 2015

@author: Lindsay

This module plots the ramp-up phase comparison of stress, current, and spikes
between a disp-controlled ramp-and-hold and a sinusoid stimulation of 1 Hz.
"""

import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(3, 1, figsize=(3, 5))
for axes_id, axes in enumerate(axs.ravel()):
    axes.text(-.2, 1.1, chr(65+axes_id), transform=axes.transAxes,
              fontsize=12, fontweight='bold', va='top')
axs0 = axs[0]
axs1 = axs[1]
axs2 = axs[2]


# %% Plot 1: Sine vs disp stress
d3_time = np.genfromtxt('./data/dcon_disp3_time.csv', delimiter=',')
d3_stress = np.genfromtxt('./data/dcon_disp3_stress.csv', delimiter=',')
s_time = np.genfromtxt('./data/stress7200_sin_1hz_time.csv', delimiter=',')
s_stress = np.genfromtxt('./data/stress7200_sin_1hz_stress.csv', delimiter=',')
axs0.plot(s_time, s_stress*1e-3, color='0.5')
axs0.plot(d3_time, d3_stress*1e-3, color='0')
axs0.set_xlim(0, 1)
axs0.set_ylim(0, 8)
axs0.set_yticks([0, 2, 4, 6, 8])
axs0.set_xlabel('Time (sec)')
axs0.set_ylabel('Stress (kPa)')


# %% Plot 2: Sine vs disp current
d3_gen_cur = np.genfromtxt('./data/disp3_gen_current.csv', delimiter=',')
s_cur = np.genfromtxt('./data/stress7200_sin_1hz_gen_current.csv',
                      delimiter=',')
s_cur = s_cur[:, 0]
axs1.plot(s_time, -s_cur*1e9, color='0.5')
axs1.plot(d3_time, -d3_gen_cur*1e9, color='0')
axs1.set_xlim(0, 1)
axs1.set_ylim(-20, 0)
axs1.set_yticks([-20, -15, -10, -5, 0])
axs1.set_xlabel('Time (sec)')
axs1.set_ylabel('Current (pA)')


# %% Plot 3: Sine vs disp spikes
d3_spike = np.genfromtxt('./data/disp3_gen_spike.csv', delimiter=',')
s_spike = np.genfromtxt('./data/stress7200_sin_1hz_gen_spike.csv',
                        delimiter=',')
axs2.vlines(s_spike, ymin=-1.5, ymax=-0.5, linewidth=0.5, color='0.5')
axs2.axhline(y=-1, xmin=0, xmax=5, linewidth=0.5, color='0.5')
axs2.vlines(d3_spike, ymin=0.5, ymax=1.5, linewidth=0.5, color='0')
axs2.axhline(y=1, xmin=0, xmax=5, linewidth=0.1, color='0')
axs2.set_xlim(0, 1)
axs2.set_ylim(-2, 2)
axs2.set_xlabel('Time (sec)')
axs2.set_ylabel('Spikes')
axs2.get_yaxis().set_ticks([])


# %%
fig.tight_layout()
fig.savefig('./F9 sine ramp/F9 sine ramp.tif', dpi=600)
