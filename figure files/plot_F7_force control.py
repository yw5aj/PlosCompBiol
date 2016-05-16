# -*- coding: utf-8 -*-
"""
Created on Fri May 15 14:23:29 2015

@author: Lindsay

This module plots model stress and firing rates comparison of displacement- and
force-controlled stimulations.
"""

import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(2, 1, figsize=(3, 3.3))
for axes_id, axes in enumerate(axs.ravel()):
    axes.text(-.2, 1.1, chr(65+axes_id), transform=axes.transAxes,
              fontsize=12, fontweight='bold', va='top')
axs0 = axs[0]
axs1 = axs[1]


# %% Plot 1 part 1: plot disp control: stress
d2_time = np.genfromtxt('./data/dcon_disp2_time.csv', delimiter=',')
d3_time = np.genfromtxt('./data/dcon_disp3_time.csv', delimiter=',')
d2_stress = np.genfromtxt('./data/dcon_disp2_stress.csv', delimiter=',')
d3_stress = np.genfromtxt('./data/dcon_disp3_stress.csv', delimiter=',')
axs0.plot(d2_time, d2_stress*1e-3, color='0.5')
axs0.plot(d3_time, d3_stress*1e-3, color='0')
axs0.set_xlim(0, 5)
axs0.set_ylim(0, 8)
axs0.set_yticks([0, 2, 4, 6, 8])
axs0.set_xlabel('Time (sec)')
axs0.set_ylabel('Stress (kPa)')


# %% Plot 1 part 2: plot force control: stress
fcon_d4_time = np.genfromtxt('./data/fcon_disp4_time.csv', delimiter=',')
fcon_d6_time = np.genfromtxt('./data/fcon_disp6_time.csv', delimiter=',')
fcon_d4_stress = np.genfromtxt('./data/fcon_disp4_stress.csv', delimiter=',')
fcon_d6_stress = np.genfromtxt('./data/fcon_disp6_stress.csv', delimiter=',')
axs0.plot(fcon_d4_time, fcon_d4_stress*1e-3, linestyle='--', color='0.5')
axs0.plot(fcon_d6_time, fcon_d6_stress*1e-3, linestyle='--', color='0')


# %% Plot 2 part 1: plot disp control: fr
d2_gen_fr_time = np.genfromtxt('./data/disp2_gen_inst_fr_time.csv',
                               delimiter=',')
d2_gen_fr_fr = np.genfromtxt('./data/disp2_gen_inst_fr_fr.csv', delimiter=',')
d3_gen_fr_time = np.genfromtxt('./data/disp3_gen_inst_fr_time.csv',
                               delimiter=',')
d3_gen_fr_fr = np.genfromtxt('./data/disp3_gen_inst_fr_fr.csv', delimiter=',')
axs1.plot(d2_gen_fr_time, d2_gen_fr_fr, color='0.5')
axs1.plot(d3_gen_fr_time, d3_gen_fr_fr, color='0')
axs1.set_xlim(0, 5)
axs1.set_ylim(0, 150)
axs1.set_yticks([0, 50, 100, 150])
axs1.set_xlabel('Time (sec)')
axs1.set_ylabel('FR (Hz)')


# %% Plot 2 part 2: plot force control: fr
fcon_d4_gen_fr_time = np.genfromtxt('./data/fcon_disp4_gen_inst_fr_time.csv',
                                    delimiter=',')
fcon_d4_gen_fr_fr = np.genfromtxt('./data/fcon_disp4_gen_inst_fr_fr.csv',
                                  delimiter=',')
fcon_d6_gen_fr_time = np.genfromtxt('./data/fcon_disp6_gen_inst_fr_time.csv',
                                    delimiter=',')
fcon_d6_gen_fr_fr = np.genfromtxt('./data/fcon_disp6_gen_inst_fr_fr.csv',
                                  delimiter=',')
axs1.plot(fcon_d4_gen_fr_time, fcon_d4_gen_fr_fr, linestyle='--', color='0.5')
axs1.plot(fcon_d6_gen_fr_time, fcon_d6_gen_fr_fr, linestyle='--', color='0')


# %%
fig.tight_layout()
fig.savefig('./F7 force control/F7 force control.tif', dpi=600)
