# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:55:45 2021

@author: Lucia
"""
#%% OBTAIN BG FUNCTION

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir(r'C:\Users\Lucia\Documents\NanoFísica\SIMPLER\SIMPLER-master_MATLAB\SIMPLER-master\Example data')

# Define filename
filename = "example_spectrin_large.hdf5"

# Read H5 file
f = h5.File(filename, "r")
dataset = f['locs']

# Load  input HDF5 file
frame = dataset['frame']
photon_raw = dataset['photons']
bg = dataset['bg']

x = dataset['x'] 
y = dataset['y'] 
sx = dataset['lpx']
sy = dataset['lpy']
sd = (sx*2 + sy*2)**0.5

# Take x,y,sd values in camera subpixels
camera_px = 133

# Convert x,y,sd values from 'camera subpixels' to nanometres
xloc = x * camera_px
yloc = y * camera_px
# sxloc = dataset['lpx'] * camera_px
# syloc = dataset['lpy'] * camera_px
# sdloc = (sxloc*2 + syloc*2)**0.5

        
## Excitation profile calculation and exportation:
Img_bg = np.zeros((np.int(np.ceil(np.max(x))), np.int(np.ceil(np.max(y)))))
# Empty matrix to be filled with background values pixel-wise.
     
Count_molec_px = Img_bg # Empty matrix to be filled with the number of molecules
                             # used to calcualte local background for each pixel.
     
# If the list contains > 1,000,000 localizations, only 500,000 are used in
# order to speed up the analysis and avoid redundancy.
vector_indices = np.arange(0,np.size(x))
c_random = np.random.permutation(np.size(vector_indices))
length_c_random = np.int(np.min([1e+6, np.size(xloc)]))
range_c = range(0,length_c_random)
c = c_random[range_c]


for i in range(0, np.size(xloc[c])):
    
    # The #molecules used to calculate local background in current pixel is updated.
    xind = np.int(np.ceil(x[c[i]]))-1
    yind = np.int(np.ceil(y[c[i]]))-1
    Count_molec_px[xind, yind] = Count_molec_px[xind, yind] + 1
    Count_i = Count_molec_px[xind, yind]
    
    # If the current pixel's background has already been estimated, then its 
    # value is updated through a weighted average:
    count_div = (Count_i-1)/Count_i
    if Count_i > 1:
        Img_bg[xind, yind] = count_div * (Img_bg[xind, yind]) + (1/Count_i) * bg[c[i]] 
    else:
        Img_bg[xind, yind] = bg[c[i]]
        
