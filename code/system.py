#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:09:17 2024

System functions

@author: vy902033
"""

import os as os
import matplotlib.pyplot as plt


def _setup_dirs_():
    
    cwd = os.path.abspath(os.path.dirname(__file__))
    root = os.path.dirname(cwd)
    datapath = os.path.join(root,'Data')
    
    dirs = {"datapath": datapath}
    dirs['root'] = root
    
    return dirs

def plot_defaults():
    # Update font size
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,   # Font size for axis labels
        'xtick.labelsize': 12,  # Font size for x-axis tick labels
        'ytick.labelsize': 12,   # Font size for y-axis tick labels
        'legend.fontsize': 12
    })
    plt.rcParams.update({'font.family': 'Tahoma'})
    
def is_scalar(obj):
    # Check if the object is an instance of any scalar type
    return isinstance(obj, (int, float, complex, str, bytes))