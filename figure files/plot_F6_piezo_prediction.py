# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:00:55 2015

@author: Lindsay

This module plots currents and firing rate comparisons of model and recordings
under normal and Piezo2 deficient conditions.
"""

import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(3, 2, figsize=(6, 5))
for axes_id, axes in enumerate(axs.ravel()):
    axes.text(-.2, 1.1, chr(65+axes_id), transform=axes.transAxes,
              fontsize=12, fontweight='bold', va='top')
axs0 = axs[0, 0]
axs1 = axs[0, 1]
axs2 = axs[1, 0]
axs3 = axs[1, 1]
axs4 = axs[2, 0]
axs5 = axs[2, 1]


# %% Plot 1: Normal: model currents in pA
d2_cur_t = np.genfromtxt('./data/disp2_fine_time.csv', delimiter=',')
d3_cur_t = np.genfromtxt('./data/disp3_fine_time.csv', delimiter=',')
d2_gen_cur = np.genfromtxt('./data/disp2_gen_current.csv', delimiter=',')
d3_gen_cur = np.genfromtxt('./data/disp3_gen_current.csv', delimiter=',')
axs0.plot(d2_cur_t, -d2_gen_cur*1e9, color='0.5')
axs0.plot(d3_cur_t, -d3_gen_cur*1e9, color='0')
axs0.set_xlim(0, 5)
axs0.set_ylim(-20, 0)
axs0.set_yticks([-20, -15, -10, -5, 0])
axs0.set_xlabel('Time (sec)')
axs0.set_ylabel('Current (pA)')


# %% Plot 2: Piezo2: model currents in pA
d2_nr_cur = np.genfromtxt('./data/disp2_nr_current.csv', delimiter=',')
d3_nr_cur = np.genfromtxt('./data/disp3_nr_current.csv', delimiter=',')
axs1.plot(d2_cur_t, -d2_nr_cur*1e9, color='0.5')
axs1.plot(d3_cur_t, -d3_nr_cur*1e9, color='0')
axs1.set_xlim(0, 5)
axs1.set_ylim(-20, 0)
axs1.set_yticks([-20, -15, -10, -5, 0])
axs1.set_xlabel('Time (sec)')
axs1.set_ylabel('Current (pA)')


# %% Plot 3: Normal: model firing rates
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
axs2.set_xlim(0, 5)
axs2.set_ylim(0, 200)
axs2.set_xlabel('Time (sec)')
axs2.set_ylabel('FR (Hz)')


# %% Plot 4: Piezo2: model firing rates
d2_nr_fr_time = np.genfromtxt('./data/disp2_nr_inst_fr_time.csv',
                              delimiter=',')
d3_nr_fr_time = np.genfromtxt('./data/disp3_nr_inst_fr_time.csv',
                              delimiter=',')
d2_nr_fr_fr = np.genfromtxt('./data/disp2_nr_inst_fr_fr.csv', delimiter=',')
d3_nr_fr_fr = np.genfromtxt('./data/disp3_nr_inst_fr_fr.csv', delimiter=',')
axs3.plot(d2_nr_fr_time, d2_nr_fr_fr, marker='.', linestyle='None',
          color='0.5')
axs3.plot(d3_nr_fr_time, d3_nr_fr_fr, marker='.', linestyle='None', color='0')
axs3.set_xlim(0, 5)
axs3.set_ylim(0, 200)
axs3.set_xlabel('Time (sec)')
axs3.set_ylabel('FR (Hz)')


# %% Plot 5: Normal: recording firing rates
rec_disp1_normal_time, rec_disp1_normal_fr = np.genfromtxt('./data/rec_disp1_'
                                                           'normal_fr.csv',
                                                           delimiter=',').T
rec_disp2_normal_time, rec_disp2_normal_fr = np.genfromtxt('./data/rec_disp2_'
                                                           'normal_fr.csv',
                                                           delimiter=',').T
axs4.plot(rec_disp1_normal_time-1, rec_disp1_normal_fr, marker='.',
          linestyle='None', color='0')
axs4.plot(rec_disp2_normal_time-1, rec_disp2_normal_fr, marker='.',
          linestyle='None', color='0.5')
axs4.set_xlim(0, 5)
axs4.set_ylim(0, 200)
axs4.set_xlabel('Time (sec)')
axs4.set_ylabel('FR (Hz)')


# %% Plot 6: Piezo2: recording firing rates
rec_disp1_piezo_time, rec_disp1_piezo_fr = np.genfromtxt('./data/rec_disp1_'
                                                         'piezo_fr.csv',
                                                         delimiter=',').T
rec_disp2_piezo_time, rec_disp2_piezo_fr = np.genfromtxt('./data/rec_disp2_'
                                                         'piezo_fr.csv',
                                                         delimiter=',').T
axs5.plot(rec_disp1_piezo_time-1, rec_disp1_piezo_fr, marker='.',
          linestyle='None', color='0')
axs5.plot(rec_disp2_piezo_time-1, rec_disp2_piezo_fr, marker='.',
          linestyle='None', color='0.5')
axs5.set_xlim(0, 5)
axs5.set_ylim(0, 200)
axs5.set_xlabel('Time (sec)')
axs5.set_ylabel('FR (Hz)')


# %%
fig.tight_layout()
fig.savefig('./F6 piezo prediction/F6 piezo prediction.tif', dpi=600)
