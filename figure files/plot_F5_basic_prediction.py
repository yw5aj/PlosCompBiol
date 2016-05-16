# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:10:22 2015

@author: Lindsay

This module plots the comparison of spikes and firing rates between model
prediction and electrophysiological recordings for normal SAI afferents.
"""

import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(2, 2, figsize=(6, 3.3))
for axes_id, axes in enumerate(axs.ravel()):
    axes.text(-.2, 1, chr(65+axes_id), transform=axes.transAxes,
              fontsize=12, fontweight='bold', va='top')
axs0 = axs[0, 0]
axs1 = axs[0, 1]
axs2 = axs[1, 0]
axs3 = axs[1, 1]


# %% Plot 1: Model spike with generator current
d2_gen_spike = np.genfromtxt('./data/disp2_gen_spike.csv', delimiter=',')
d3_gen_spike = np.genfromtxt('./data/disp3_gen_spike.csv', delimiter=',')
axs0.vlines(d2_gen_spike, ymin=-1.5, ymax=-0.5, linewidth=0.5, color='0.5')
axs0.axhline(y=-1, xmin=0, xmax=5, linewidth=0.5, color='0.5')
axs0.vlines(d3_gen_spike, ymin=0.5, ymax=1.5, linewidth=0.5, color='0')
axs0.axhline(y=1, xmin=0, xmax=5, linewidth=0.1, color='0')
axs0.set_xlim(0, 2)
axs0.set_ylim(-2, 2)
axs0.set_xlabel('Time (sec)')
axs0.set_ylabel('Spikes')
axs0.get_yaxis().set_ticks([])


# %% Plot 2: Recording spike with generator current
d2_rec_gen_spike = np.genfromtxt('./data/rec_disp2_run2_gen_spike.csv',
                                 delimiter=',')
d4_rec_gen_spike = np.genfromtxt('./data/rec_disp4_run3_gen_spike.csv',
                                 delimiter=',')
axs1.vlines(d2_rec_gen_spike, ymin=-1.5, ymax=-0.5, linewidth=0.5, color='0.5')
axs1.axhline(y=-1, xmin=0, xmax=5, linewidth=0.5, color='0.5')
axs1.vlines(d4_rec_gen_spike, ymin=0.5, ymax=1.5, linewidth=0.5, color='0')
axs1.axhline(y=1, xmin=0, xmax=5, linewidth=0.1, color='0')
axs1.set_xlim(0, 2)
axs1.set_ylim(-2, 2)
axs1.set_xlabel('Time (sec)')
axs1.set_ylabel('Spikes')
axs1.get_yaxis().set_ticks([])


# %% Plot 3: Model fr with generator current
d2_gen_fr_time = np.genfromtxt('./data/disp2_gen_inst_fr_time.csv',
                               delimiter=',')
d3_gen_fr_time = np.genfromtxt('./data/disp3_gen_inst_fr_time.csv',
                               delimiter=',')
d2_gen_fr_fr = np.genfromtxt('./data/disp2_gen_inst_fr_fr.csv', delimiter=',')
d3_gen_fr_fr = np.genfromtxt('./data/disp3_gen_inst_fr_fr.csv', delimiter=',')
axs2.plot(d2_gen_fr_time, d2_gen_fr_fr, marker='.', linestyle='None',
          color='0.5')
axs2.plot(d3_gen_fr_time, d3_gen_fr_fr, marker='.', linestyle='None',
          color='0')
axs2.set_xlim(0, 2)
axs2.set_ylim(0, 200)
axs2.set_xlabel('Time (sec)')
axs2.set_ylabel('FR (Hz)')


# %% Plot 4: Recording fr with generator current
d2_rec_inst_fr_time = np.genfromtxt('./data/'
                                    'rec_disp2_run2_gen_inst_fr_time.csv',
                                    delimiter=',')
d4_rec_inst_fr_time = np.genfromtxt('./data/'
                                    'rec_disp4_run3_gen_inst_fr_time.csv',
                                    delimiter=',')
d2_rec_inst_fr_fr = np.genfromtxt('./data/rec_disp2_run2_gen_inst_fr_fr.csv',
                                  delimiter=',')
d4_rec_inst_fr_fr = np.genfromtxt('./data/rec_disp4_run3_gen_inst_fr_fr.csv',
                                  delimiter=',')
axs3.plot(d2_rec_inst_fr_time, d2_rec_inst_fr_fr, marker='.', linestyle='None',
          color='0.5')
axs3.plot(d4_rec_inst_fr_time, d4_rec_inst_fr_fr, marker='.', linestyle='None',
          color='0')
axs3.set_xlim(0, 2)
axs3.set_ylim(0, 200)
axs3.set_xlabel('Time (sec)')
axs3.set_ylabel('FR (Hz)')


# %%
fig.tight_layout()
fig.savefig('./F5 basic prediction/F5 basic prediction.tif', dpi=600)
