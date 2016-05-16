# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 21:42:07 2015

@author: Lindsay

This module plots the generator current base case (Figure 3 in thesis).
"""

import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(3, 1, figsize=(3, 5))
for axes_id, axes in enumerate(axs.ravel()):
    axes.text(-.2, 1.1, chr(65+axes_id), transform=axes.transAxes,
              fontsize=12, fontweight='bold', va='top')
axs0 = axs[0]
axs1 = axs[1]
axs3 = axs[2]


# %% Merkel cell mechanism current
d2_cur_t = np.genfromtxt('./data/disp2_fine_time.csv', delimiter=',')
d2_mc_cur = np.genfromtxt('./data/disp2_mc_current.csv', delimiter=',')
axs0.plot(d2_cur_t, -d2_mc_cur*1e9, color='0')
axs0.set_xlim(0, 2)
axs0.set_ylim(-15, 0)
axs0.set_yticks([-15, -10, -5, 0])
axs0.set_xlabel('Time (sec)')
axs0.set_ylabel('Current (pA)')


# %% Neurite mechanism current
d2_nr_cur = np.genfromtxt('./data/disp2_nr_current.csv', delimiter=',')
axs1.plot(d2_cur_t, -d2_nr_cur*1e9, color='0')
axs1.set_xlim(0, 2)
axs1.set_ylim(-15, 0)
axs1.set_yticks([-15, -10, -5, 0])
axs1.set_xlabel('Time (sec)')
axs1.set_ylabel('Current (pA)')


# %% Total generator current
d2_gen_cur = np.genfromtxt('./data/disp2_gen_current.csv', delimiter=',')
axs3.plot(d2_cur_t, -d2_gen_cur*1e9, color='0')
axs3.set_xlim(0, 2)
axs3.set_ylim(-15, 0)
axs3.set_yticks([-15, -10, -5, 0])
axs3.set_xlabel('Time (sec)')
axs3.set_ylabel('Current (pA)')


# %%
fig.tight_layout()
fig.savefig('./F3 current base case/F3 current base case.tif', dpi=600)
