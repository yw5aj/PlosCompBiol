# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 12:01:09 2015

@author: Lindsay

This module plots firing rate figures for two new approaches of the generator
function to take care of an intermediate adapting response of SAI afferent.
"""

import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(3, 2, figsize=(6, 5))
for axes_id, axes in enumerate(axs.ravel()):
    axes.text(-.2, 1, chr(65+axes_id), transform=axes.transAxes,
              fontsize=12, fontweight='bold', va='top')
axs0 = axs[0, 0]
axs1 = axs[0, 1]
axs2 = axs[1, 0]
axs3 = axs[1, 1]
axs4 = axs[2, 0]
axs5 = axs[2, 1]


# %% Plot 1: Approach 1 normal
ap1_d2_gen_fr_t = np.genfromtxt('./data/ap1_disp2_gen_inst_fr_time.csv',
                                delimiter=',')
ap1_d2_gen_fr_fr = np.genfromtxt('./data/ap1_disp2_gen_inst_fr_fr.csv',
                                 delimiter=',')
ap1_d3_gen_fr_t = np.genfromtxt('./data/ap1_disp3_gen_inst_fr_time.csv',
                                delimiter=',')
ap1_d3_gen_fr_fr = np.genfromtxt('./data/ap1_disp3_gen_inst_fr_fr.csv',
                                 delimiter=',')
axs0.plot(ap1_d2_gen_fr_t, ap1_d2_gen_fr_fr, marker='.', linestyle='None',
          color='0.5')
axs0.plot(ap1_d3_gen_fr_t, ap1_d3_gen_fr_fr, marker='.', linestyle='None',
          color='0')
axs0.set_xlim(0, 2)
axs0.set_ylim(0, 200)
axs0.set_xlabel('Time (sec)')
axs0.set_ylabel('FR (Hz)')


# %% Plot 2: Approach 1 Neurite only
ap1_d2_nr_fr_t = np.genfromtxt('./data/ap1_disp2_nr_inst_fr_time.csv',
                               delimiter=',')
ap1_d2_nr_fr_fr = np.genfromtxt('./data/ap1_disp2_nr_inst_fr_fr.csv',
                                delimiter=',')
ap1_d3_nr_fr_t = np.genfromtxt('./data/ap1_disp3_nr_inst_fr_time.csv',
                               delimiter=',')
ap1_d3_nr_fr_fr = np.genfromtxt('./data/ap1_disp3_nr_inst_fr_fr.csv',
                                delimiter=',')
axs1.plot(ap1_d2_nr_fr_t, ap1_d2_nr_fr_fr, marker='.', linestyle='None',
          color='0.5')
axs1.plot(ap1_d3_nr_fr_t, ap1_d3_nr_fr_fr, marker='.', linestyle='None',
          color='0')
axs1.set_xlim(0, 2)
axs1.set_ylim(0, 200)
axs1.set_xlabel('Time (sec)')
axs1.set_ylabel('FR (Hz)')


# %% Plot 3: Approach 2 normal
ap2_d2_gen_fr_t = np.genfromtxt('./data/ap2_disp2_gen_inst_fr_time.csv',
                                delimiter=',')
ap2_d2_gen_fr_fr = np.genfromtxt('./data/ap2_disp2_gen_inst_fr_fr.csv',
                                 delimiter=',')
ap2_d3_gen_fr_t = np.genfromtxt('./data/ap2_disp3_gen_inst_fr_time.csv',
                                delimiter=',')
ap2_d3_gen_fr_fr = np.genfromtxt('./data/ap2_disp3_gen_inst_fr_fr.csv',
                                 delimiter=',')
axs2.plot(ap2_d2_gen_fr_t, ap2_d2_gen_fr_fr, marker='.', linestyle='None',
          color='0.5')
axs2.plot(ap2_d3_gen_fr_t, ap2_d3_gen_fr_fr, marker='.', linestyle='None',
          color='0')
axs2.set_xlim(0, 2)
axs2.set_ylim(0, 200)
axs2.set_xlabel('Time (sec)')
axs2.set_ylabel('FR (Hz)')


# %% Plot 4: Approach 2 Neurite only
ap2_d2_nr_fr_t = np.genfromtxt('./data/ap2_disp2_nr_inst_fr_time.csv',
                               delimiter=',')
ap2_d2_nr_fr_fr = np.genfromtxt('./data/ap2_disp2_nr_inst_fr_fr.csv',
                                delimiter=',')
ap2_d3_nr_fr_t = np.genfromtxt('./data/ap2_disp3_nr_inst_fr_time.csv',
                               delimiter=',')
ap2_d3_nr_fr_fr = np.genfromtxt('./data/ap2_disp3_nr_inst_fr_fr.csv',
                                delimiter=',')
axs3.plot(ap2_d2_nr_fr_t, ap2_d2_nr_fr_fr, marker='.', linestyle='None',
          color='0.5')
axs3.plot(ap2_d3_nr_fr_t, ap2_d3_nr_fr_fr, marker='.', linestyle='None',
          color='0')
axs3.set_xlim(0, 2)
axs3.set_ylim(0, 200)
axs3.set_xlabel('Time (sec)')
axs3.set_ylabel('FR (Hz)')


# %% Plot 5: Approach 3 normal
ap3_d2_gen_fr_t = np.genfromtxt('./data/ap3_disp2_gen_inst_fr_time.csv',
                                delimiter=',')
ap3_d2_gen_fr_fr = np.genfromtxt('./data/ap3_disp2_gen_inst_fr_fr.csv',
                                 delimiter=',')
ap3_d3_gen_fr_t = np.genfromtxt('./data/ap3_disp3_gen_inst_fr_time.csv',
                                delimiter=',')
ap3_d3_gen_fr_fr = np.genfromtxt('./data/ap3_disp3_gen_inst_fr_fr.csv',
                                 delimiter=',')
axs4.plot(ap3_d2_gen_fr_t, ap3_d2_gen_fr_fr, marker='.', linestyle='None',
          color='0.5')
axs4.plot(ap3_d3_gen_fr_t, ap3_d3_gen_fr_fr, marker='.', linestyle='None',
          color='0')
axs4.set_xlim(0, 2)
axs4.set_ylim(0, 200)
axs4.set_xlabel('Time (sec)')
axs4.set_ylabel('FR (Hz)')


# %% Plot 6: Approach 3 Neurite only
ap3_d2_nr_fr_t = np.genfromtxt('./data/ap3_disp2_nr_inst_fr_time.csv',
                               delimiter=',')
ap3_d2_nr_fr_fr = np.genfromtxt('./data/ap3_disp2_nr_inst_fr_fr.csv',
                                delimiter=',')
ap3_d3_nr_fr_t = np.genfromtxt('./data/ap3_disp3_nr_inst_fr_time.csv',
                               delimiter=',')
ap3_d3_nr_fr_fr = np.genfromtxt('./data/ap3_disp3_nr_inst_fr_fr.csv',
                                delimiter=',')
axs5.plot(ap3_d2_nr_fr_t, ap3_d2_nr_fr_fr, marker='.', linestyle='None',
          color='0.5')
axs5.plot(ap3_d3_nr_fr_t, ap3_d3_nr_fr_fr, marker='.', linestyle='None',
          color='0')
axs5.set_xlim(0, 2)
axs5.set_ylim(0, 200)
axs5.set_xlabel('Time (sec)')
axs5.set_ylabel('FR (Hz)')


# %%
fig.tight_layout()
fig.savefig('./F13 new approach/F13 new approach.tif', dpi=600)
