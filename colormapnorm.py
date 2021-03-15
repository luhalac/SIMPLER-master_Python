# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:39:18 2021

@author: Lucia
"""
import matplotlib.cm as cm
import matplotlib as matplotlib

def color_map_color(value, cmap_name='viridis', vmin=0, vmax=1):
    # norm = plt.Normalize(vmin, vmax)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)  # PiYG
    rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb
    color = matplotlib.colors.rgb2hex(rgb)
    return color