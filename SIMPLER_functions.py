#%% IMPORT HDF5 FILE

import numpy as np
import h5py as h5
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import colormaps as cmaps
from matplotlib import cm

os.chdir(r'C:\Users\Lucia\Documents\NanoFísica\SIMPLER\SIMPLER-master_MATLAB\SIMPLER-master\Example data')


# Define filename
filename = "example_mt.hdf5"

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

# Define filename
# filename = "example_mt_Thunderstorm.csv"
# dataset = pd.read_csv(filename)
# headers = dataset.columns.values

# ## Read ThunderSTRORM csv file

# frame = dataset[headers[0]].values
# x = dataset[headers[1]].values 
# y = dataset[headers[2]].values
# photon_raw = dataset[headers[3]].values
# bg = dataset[headers[4]].values

# Take x,y,sd values in camera subpixels
camera_px = 133

# Convert x,y,sd values from 'camera subpixels' to nanometres
xloc = x * camera_px
yloc = y * camera_px
# sxloc = dataset['lpx'] * camera_px
# syloc = dataset['lpy'] * camera_px
# sdloc = (sxloc*2 + syloc*2)**0.5

#%% CORRECT PHOTON COUNTS

# To perform this correction, the linear dependency between local laser
# power intensity and local background photons is used. Photons are
# converted to the value they would have if the whole image was illuminated
# with the maximum laser power intensity. This section is executed if the
# user has chosen to perform correction due to non-flat illumination.

filename_csv = 'excitation_profile_mt.csv'

datacalib = pd.read_csv(filename_csv)
profiledata = pd.DataFrame(datacalib)
profile = profiledata.values
# print(matplotlib.pyplot.imshow(profile))

phot = photon_raw
max_bg = np.percentile(profile, 97.5)
phot_corr = np.zeros(photon_raw.size)

# Correction loop

profx = np.size(profile,0) 
profy = np.size(profile,1) 

xdata = x
ydata = y

for i in np.arange(len(phot)):
    print(i)
    if int((np.ceil(xdata[i]))) < profx and int((np.ceil(ydata[i]))) < profy:
        phot_corr[i] = phot[i]*(max_bg)/(profile[int(np.ceil(xdata[i])),int(np.ceil(ydata[i]))])
    elif int((np.ceil(xdata[i]))) > profx and int((np.ceil(ydata[i]))) < profy:
        phot_corr[i] = phot[i]*(max_bg)/(profile[int(np.floor(xdata[i])),int(np.ceil(ydata[i]))])
    elif int((np.ceil(xdata[i]))) < profx and int((np.ceil(ydata[i]))) > profy:
        phot_corr[i] = phot[i]*(max_bg)/(profile[int(np.ceil(xdata[i])),int(np.floor(ydata[i]))])
    elif int((np.ceil(xdata[i]))) > profx and int((np.ceil(ydata[i]))) > profy:
        phot_corr[i] = phot[i]*(max_bg)/(profile[int(np.floor(xdata[i])),int(np.floor(ydata[i]))])


    
# phot_corr = photon_raw
        
# Build the output array
listLocalizations = np.column_stack((xloc, yloc, frame, phot_corr))

#%% REMOVING LOCS WITH NO LOCS IN i-1 AND i+1 FRAME

# We keep a molecule if there is another one in the previous and next frame,
# located at a distance < max_dist, where max_dist is introduced by user
# (20 nm by default).

min_div = 100.0
# We divide the list into sub-lists of 'min_div' locs to minimize memory usage

Ntimes_min_div = int((listLocalizations[:,0].size/min_div))
truefalse_sum_roi_acum = []
listLocalizations_filtered = np.zeros((int(min_div*Ntimes_min_div),listLocalizations[1,:].size))

    
daa = np.zeros(listLocalizations[:,0].size)
frame_dif = np.zeros(listLocalizations[:,0].size)
    
for N in range(0, Ntimes_min_div+1):

    min_div_N = int(min_div)
    min_range = min_div_N*N
    max_range = (min_div_N*(N+1))
   
    if N == Ntimes_min_div:
        min_div_N = int(listLocalizations[:,0].size - min_div *(Ntimes_min_div))
        max_range = int(listLocalizations[:,0].size)
       
    truefalse = np.zeros((min_div_N,min_div_N))
    # This matrix relates each localization with the rest of localizations.
    # It will take a value of 1 if the molecules i and j (i = row, j = column)
    # are located at distances < max_dist, and are detected in frames N and (N+1)
    # or (N-1). In any other case, it will take a value of 0.

    max_dist = 20 # Value in nanometers
    
    for i in range(min_range, max_range):
        for j in range(min_range, max_range):
            daa[j-min_div_N*(N)] = ((xloc[i]-xloc[j])**2+(yloc[i]-yloc[j])**2)**(1/2)
            frame_dif[j-min_div_N*(N)] = ((listLocalizations[i,2] - listLocalizations[j,2])**2)**(1/2)
            if daa[(j-min_div_N*(N))] < max_dist and frame_dif[(j-min_div_N*(N))] == 1:
                truefalse[(i-min_range),(j-min_range)] = 1
   
    truefalse_sum = truefalse.sum(axis=0)  
    # For each row (i.e. each molecule) we calculate the sum of every
    # column from the 'truefalse' matrix.
   
    truefalse_sum_roi_acum = np.append(truefalse_sum_roi_acum, truefalse_sum)
   
idx_filtered = np.where(truefalse_sum_roi_acum > 1)
# We choose the indexes of those rows whose columnwise sum is > or = to 2

#%% APPLYING FILTER TO THE ORIGINAL LIST 

x_idx = listLocalizations[idx_filtered,0].T
y_idx = listLocalizations[idx_filtered,1].T
frame_idx = listLocalizations[idx_filtered,2].T
photons_idx = listLocalizations[idx_filtered,3].T

#%% Z-Calculation

alphaF = 0.96
N0 = 51000
dF = 87.7
photons1 = photons_idx

# Small ROI and Large ROI cases


# z1 = (np.log(alphaF*N0)-np.log(photons1-(1-alphaF)*N0))/(1/dF)
# z1 = np.real(z1)

# # Scatter plots / Projection r vs. z, x vs. z , y vs. z
# # ------------------------------------------------------------------------
# # If the selected operation is "(r,z) Small ROI", we perform a first step 
# # where the main axis is obtained from a linear fit of the (x,y) data

# x = x_idx.flatten()
# y = y_idx.flatten()
# z = z1.flatten()

# M = np.matrix([x,y,z])
# P = np.polyfit(x,y,1)
# yfit = P[0]*x + P[1]
# def Poly_fun(x):
#     y_polyfunc = P[0]*x + P[1]
#     return y_polyfunc

# Origin_X = 0.99999*min(x)
# Origin_Y = Poly_fun(Origin_X)

# # Change from cartesian to polar coordinates
# tita = np.arctan(P[0])
# tita1 = np.arctan((y-Origin_Y)/(x-Origin_X))
# r = ((x-Origin_X)**2+(y-Origin_Y)**2)**(1/2)
# tita2 = [x - tita for x in tita1]
# proyec_r = np.cos(tita2)*r


# simpler_output = np.column_stack((x_idx, y_idx, proyec_r, z1, photons_idx, frame_idx))



# ind = np.argsort(z)
# xind = x[ind]
# yind = y[ind]
# zind = z[ind]
# rind = r[ind]

# cmapz = cm.get_cmap('viridis', np.size(z))

           
# col = cmapz.colors
# col = np.delete(col, np.s_[3], axis=1)

# plt.figure()
# plt.scatter(rind, zind, c=col)





# N0 Calibration

# if rz_xyz == 5:
    # For the "N0 Calibration" operation, there is no "Z calculation", 
    # because the aim of this procedure is to obtain N0 from a sample which 
    # is supposed to contain molecules located at z ~ 0.
xl = np.array([np.amax(x_idx), np.amin(x_idx)]) 
yl = np.array([np.amax(y_idx), np.amin(y_idx)]) 
c = np.arange(0,np.size(x_idx))

hist, bin_edges = np.histogram(photons_idx[c], bins = 40, density = False)
bin_limits = np.array([bin_edges[0], bin_edges[-1]])
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

# Gaussian fit of the N0 distribution
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
A0 = np.max(hist)
mu0 = np.mean(bin_centres)
sigma0 = np.std(bin_centres)
p0 = [A0, mu0, sigma0]
coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)   

# Get the fitted curve
hist_fit = gauss(bin_centres, *coeff)
plt.plot(bin_centres, hist, label='Non-fit data')
plt.hist(photons_idx[c], bins = 40)
plt.plot(bin_centres, hist_fit, label='Fitted data')
plt.show()