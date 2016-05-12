# -*- coding: utf-8 -*-
"""
Created on Sun May  8 22:09:54 2016

@author: Shawn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    df_dict = {key: pd.read_excel('timeconst.xlsx', key)
               for key in ['nr', 'mc']}
    tau_dict = {}
    for key, item in df_dict.items():
        tau_dict[key] = item.values.ravel()
        tau_dict[key] = tau_dict[key][~np.isnan(tau_dict[key])]
    plt.boxplot(list(tau_dict.values()))
