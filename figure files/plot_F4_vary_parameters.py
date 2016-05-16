# -*- coding: utf-8 -*-
"""
Created on Thu May 14 23:35:02 2015

@author: Lindsay

This module plots currents and firing rates affected by varying parameters
in the generator function.
"""

import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(3, 2, figsize=(6, 5))
for axes_id, axes in enumerate(axs.ravel()):
    axes.text(-.2, 1.1, chr(65+axes_id), transform=axes.transAxes,
              fontsize=12, fontweight='bold', va='top')
axs0 = axs[0, 0]
axs1 = axs[1, 0]
axs2 = axs[2, 0]
axs3 = axs[0, 1]
axs4 = axs[1, 1]
axs5 = axs[2, 1]


# %% Change tau1: current
d2_time = np.genfromtxt('./data/disp2_fine_time.csv', delimiter=',')
d2_gen_tau1_low = np.genfromtxt('./data/disp2_tau1s_gen_current.csv',
                                delimiter=',')
d2_gen_tau1_med = np.genfromtxt('./data/disp2_gen_current.csv', delimiter=',')
d2_gen_tau1_high = np.genfromtxt('./data/disp2_tau1l_gen_current.csv',
                                 delimiter=',')
axs0.plot(d2_time, -d2_gen_tau1_low*1e9, label=r'$\tau_1$=.003', color='0.8')
axs0.plot(d2_time, -d2_gen_tau1_med*1e9, label=r'$\tau_1$=.008', color='0.4')
axs0.plot(d2_time, -d2_gen_tau1_high*1e9, label=r'$\tau_1$=.013', color='0.0')
leg0 = axs0.legend(loc=4, prop={'size': 8})
colors = ['0.8', '0.4', '0']
for color, text in zip(colors, leg0.get_texts()):
    text.set_color(color)
axs0.set_xlim(0, 2)
axs0.set_ylim(-20, 0)
axs0.set_xlabel('Time (sec)')
axs0.set_ylabel('Current (pA)')


# %% Change tau2: current
d2_gen_tau2_low = np.genfromtxt('./data/disp2_tau2s_gen_current.csv',
                                delimiter=',')
d2_gen_tau2_med = np.genfromtxt('./data/disp2_gen_current.csv', delimiter=',')
d2_gen_tau2_high = np.genfromtxt('./data/disp2_tau2l_gen_current.csv',
                                 delimiter=',')
axs1.plot(d2_time, -d2_gen_tau2_low*1e9, label=r'$\tau_2$=.05', color='0.8')
axs1.plot(d2_time, -d2_gen_tau2_med*1e9, label=r'$\tau_2$=.2', color='0.4')
axs1.plot(d2_time, -d2_gen_tau2_high*1e9, label=r'$\tau_2$=.35', color='0.0')
leg1 = axs1.legend(loc=4, prop={'size': 8})
colors = ['0.8', '0.4', '0']
for color, text in zip(colors, leg1.get_texts()):
    text.set_color(color)
axs1.set_xlim(0, 2)
axs1.set_ylim(-20, 0)
axs1.set_xlabel('Time (sec)')
axs1.set_ylabel('Current (pA)')


# %% Change k1: current
d2_gen_k1_low = np.genfromtxt('./data/disp2_k1s_gen_current.csv',
                              delimiter=',')
d2_gen_k1_med = np.genfromtxt('./data/disp2_gen_current.csv', delimiter=',')
d2_gen_k1_high = np.genfromtxt('./data/disp2_k1l_gen_current.csv',
                               delimiter=',')
axs2.plot(d2_time, -d2_gen_k1_low*1e9, label=r'$K_1$=.1', color='0.8')
axs2.plot(d2_time, -d2_gen_k1_med*1e9, label=r'$K_1$=.5', color='0.4')
axs2.plot(d2_time, -d2_gen_k1_high*1e9, label=r'$K_1$=.9', color='0.0')
leg2 = axs2.legend(loc=4, prop={'size': 8})
colors = ['0.8', '0.4', '0']
for color, text in zip(colors, leg2.get_texts()):
    text.set_color(color)
axs2.set_xlim(0, 2)
axs2.set_ylim(-20, 0)
axs2.set_xlabel('Time (sec)')
axs2.set_ylabel('Current (pA)')


# %% Change tau1: fr
d2_gen_tau1_low_fr_time = np.genfromtxt('./data/disp2_tau1s_gen_inst_fr_time'
                                        '.csv', delimiter=',')
d2_gen_tau1_low_fr = np.genfromtxt('./data/disp2_tau1s_gen_inst_fr_fr.csv',
                                   delimiter=',')
d2_gen_tau1_med_fr_time = np.genfromtxt('./data/disp2_gen_inst_fr_time.csv',
                                        delimiter=',')
d2_gen_tau1_med_fr = np.genfromtxt('./data/disp2_gen_inst_fr_fr.csv',
                                   delimiter=',')
d2_gen_tau1_high_fr_time = np.genfromtxt('./data/disp2_tau1l_gen_inst_fr_time'
                                         '.csv', delimiter=',')
d2_gen_tau1_high_fr = np.genfromtxt('./data/disp2_tau1l_gen_inst_fr_fr.csv',
                                    delimiter=',')
axs3.plot(d2_gen_tau1_low_fr_time, d2_gen_tau1_low_fr, marker='.',
          linestyle='None', color='0.8')
axs3.plot(d2_gen_tau1_med_fr_time, d2_gen_tau1_med_fr, marker='.',
          linestyle='None', color='0.4')
axs3.plot(d2_gen_tau1_high_fr_time, d2_gen_tau1_high_fr, marker='.',
          linestyle='None', color='0')
axs3.set_xlim(0, 2)
axs3.set_ylim(0, 160)
axs3.set_yticks([0, 40, 80, 120, 160])
axs3.set_xlabel('Time (sec)')
axs3.set_ylabel('FR (Hz)')


# %% Change tau2: fr
d2_gen_tau2_low_fr_time = np.genfromtxt('./data/disp2_tau2s_gen_inst_fr_time'
                                        '.csv', delimiter=',')
d2_gen_tau2_low_fr = np.genfromtxt('./data/disp2_tau2s_gen_inst_fr_fr.csv',
                                   delimiter=',')
d2_gen_tau2_med_fr_time = np.genfromtxt('./data/disp2_gen_inst_fr_time.csv',
                                        delimiter=',')
d2_gen_tau2_med_fr = np.genfromtxt('./data/disp2_gen_inst_fr_fr.csv',
                                   delimiter=',')
d2_gen_tau2_high_fr_time = np.genfromtxt('./data/disp2_tau2l_gen_inst_fr_time'
                                         '.csv', delimiter=',')
d2_gen_tau2_high_fr = np.genfromtxt('./data/disp2_tau2l_gen_inst_fr_fr.csv',
                                    delimiter=',')
axs4.plot(d2_gen_tau2_low_fr_time, d2_gen_tau2_low_fr, marker='.',
          linestyle='None', color='0.8')
axs4.plot(d2_gen_tau2_med_fr_time, d2_gen_tau2_med_fr, marker='.',
          linestyle='None', color='0.4')
axs4.plot(d2_gen_tau2_high_fr_time, d2_gen_tau2_high_fr, marker='.',
          linestyle='None', color='0')
axs4.set_xlim(0, 2)
axs4.set_ylim(0, 160)
axs4.set_yticks([0, 40, 80, 120, 160])
axs4.set_xlabel('Time (sec)')
axs4.set_ylabel('FR (Hz)')


# %% Change k1: fr
d2_gen_k1_low_fr_time = np.genfromtxt('./data/disp2_k1s_gen_inst_fr_time.csv',
                                      delimiter=',')
d2_gen_k1_low_fr = np.genfromtxt('./data/disp2_k1s_gen_inst_fr_fr.csv',
                                 delimiter=',')
d2_gen_k1_med_fr_time = np.genfromtxt('./data/disp2_gen_inst_fr_time.csv',
                                      delimiter=',')
d2_gen_k1_med_fr = np.genfromtxt('./data/disp2_gen_inst_fr_fr.csv',
                                 delimiter=',')
d2_gen_k1_high_fr_time = np.genfromtxt('./data/disp2_k1l_gen_inst_fr_time.csv',
                                       delimiter=',')
d2_gen_k1_high_fr = np.genfromtxt('./data/disp2_k1l_gen_inst_fr_fr.csv',
                                  delimiter=',')
axs5.plot(d2_gen_k1_low_fr_time, d2_gen_k1_low_fr, marker='.',
          linestyle='None', color='0.8')
axs5.plot(d2_gen_k1_med_fr_time, d2_gen_k1_med_fr, marker='.',
          linestyle='None', color='0.4')
axs5.plot(d2_gen_k1_high_fr_time, d2_gen_k1_high_fr, marker='.',
          linestyle='None', color='0')
axs5.set_xlim(0, 2)
axs5.set_ylim(0, 160)
axs5.set_yticks([0, 40, 80, 120, 160])
axs5.set_xlabel('Time (sec)')
axs5.set_ylabel('FR (Hz)')


# %%
fig.tight_layout()
fig.savefig('./F4 vary parameters/F4 vary parameters.tif', dpi=600)
