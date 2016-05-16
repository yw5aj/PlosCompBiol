# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:11:58 2015

@author: Lindsay

This module plots a sample stress over time trace as part of Figure 10.
"""
import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(1, 1, figsize=(6, 2))


# %% Plot stress
d2_stress = np.genfromtxt('./data/dcon_disp2_stress.csv', delimiter=',')
d2_time = np.genfromtxt('./data/dcon_disp2_time.csv', delimiter=',')
axs.plot(d2_time, d2_stress*1e-3, color='0')
axs.set_xlim(0, 5)
axs.set_ylim(0, 6)
axs.set_xlabel('Time (sec)')
axs.set_ylabel('Stress (kPa)')

# %%
fig.tight_layout()
fig.savefig('./F10 function concept/F10 function concept.tif', dpi=600)
