# -*- coding: utf-8 -*-
"""
Created on Wed May 26 14:43:32 2021

@author: Lucia
"""
import numpy as np
import pandas as pd
import circle_fit
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

os.chdir(r'C:\Users\Lucia\Documents\NanoFísica\SIMPLER\SIMPLER-master_MATLAB\SIMPLER-master\Example data')

# Read csv file containing (lateral,axial) positions of known
# structures.

# Define filename
filename = "example_mt_rz_tunning_8mts.csv"
## Read csv file
dataset = pd.read_csv(filename, header=None)

# Lateral positions are obtained from odd columns
lateral_matrix = dataset.values[:, ::2]
lateral_matrix[np.where(np.isnan(lateral_matrix))]=0
# Axial positions are obtained from even columns
axial_matrix = dataset.values[:, 1::2]
axial_matrix[np.where(np.isnan(axial_matrix))]=0

   
# reshape into vectors    
lateral = lateral_matrix.flatten() 
axial = axial_matrix.flatten()
    
# The values used for the axial positions calculation of the known
# structures through SIMPLER are obtained from the 'run_SIMPLER'
# interface. It is important to set them correctly before running the
# operation.

N0_input = 50000
alpha_input = 0.9
angle_input = 69.5
lambda_exc_input = 642
lambda_em_input = 700
nI_input = 1.516
nS_input = 1.33
NA_input = 1.42
dF_input = 87.7
alphaF_input = 0.96

# The number of photons for each localization are retrieved from the 
# axial position and the dF, alphaF and N0 values obtained in the
# above step.

photons_matrix = np.zeros(np.shape(axial_matrix))
photons_median_matrix = np.zeros(np.shape(axial_matrix))
lateral_median_matrix = np.zeros(np.shape(axial_matrix))

# The next function allows to obtain a custom 'median' value, which is
# calculated as the mean value between the p-centile 10% and p-centile
# 90% from a given distribution. We use this function in order to
# re-center the localizations from the known structures around [0,0]

median_perc90_10_center = lambda x: np.mean([np.percentile(x,90), np.percentile(x,10)])

# To calculate the 'median' values, we use valid localizations, 
# i.e. those with axial positions different from 0;
# there are elements filled with z = 0 and lateral = 0 in the 'data' matrix,
# because not every known structures have the same number of localizations

for i in np.arange(np.shape(lateral_matrix)[1]):
    c = np.where(axial_matrix[:,i]!=0) 
    photons_matrix[:,i] = N0_input*(alphaF_input*np.exp(-axial_matrix[:,i]/dF_input)
    + 1-alphaF_input)
    photons_median_matrix[:,i] = (np.ones((np.shape(photons_matrix)[0]))
                                *median_perc90_10_center(photons_matrix[c,i]))
    lateral_median_matrix[:,i] = (np.ones((np.shape(lateral_matrix)[0]))
                                *median_perc90_10_center(lateral_matrix[c,i]))

# Number of photons for each lozalization    
photons = photons_matrix.flatten()
# Median value of the number of photons for the structure to which
# each localization belongs
photons_median = photons_median_matrix.flatten()
# Lateral positions median value for the structure to which
# each localization belongs                                                
lateral_median = lateral_median_matrix.flatten()
# Median value of the axial position for the structure to which 
# each localization belongs                                            
axial_median = (np.log(alphaF_input*N0_input)-np.log(photons_median
                -(1-alphaF_input)*N0_input))/(1/dF_input)

# Some elements from the axial vector are zero because the 'data' matrix
# contains structures with different number of localizations and thus
# there are columns filled with zeros. Now, we remove those elements. 
    
c = np.where(axial == 0)

photons = np.delete(photons,c) 
photons_median = np.delete(photons_median,c)
lateral_median = np.delete(lateral_median,c)
lateral = np.delete(lateral,c)
axial_median = np.delete(axial_median,c)
axial = np.delete(axial,c)
axial = dF_input*(np.log(alphaF_input)-np.log(photons/N0_input-(1-alphaF_input)))


# The output for this function will contain the information of the
# parameters used to retrieve the number of photons from the axial
# positions; the number of photons and lateral position of every
# localization; and the median values for both the number of photons and
# the lateral positions.
# These outputs will be used by the 'Update Scatter' buttom as
# input information, in order to recalculate every axial position when
# either the angle or alpha (or both) are changed.

    
# output_ft_input[:,0] = [lambda_exc_input, NA_input, lambda_em_input, nS_input, nI_input, photons]
# output_ft_input[:,1] = [np.ones((5,1)), lateral]
# output_ft_input[:,2] = [np.ones((5,1)), photons_median]
# output_ft_input[:,3] = [np.ones((5,1)), lateral_median]

# % Scatter plot of the relative (lateral,axial) positions (centered in
# % [0,0])

# axes(handles.axes5); 
# set(handles.axes5,'visible', 'on'); 
# set(handles.panel_ft_input_tag,'visible', 'on');
# set(handles.ft_panel_tag,'visible', 'on'); 
# set(handles.auto_ft_panel_tag,'visible', 'on'); 

axialc = axial-axial_median
lateralc = lateral - lateral_median
plt.scatter(lateralc,axialc)
plt.xlabel('Lateral (nm)');
plt.ylabel('Axial (nm)'); 
plt.title('Combined structures');

xy = np.vstack((lateralc, axialc)).T
circle = circle_fit.least_squares_circle(xy)
xc = circle[0]
yc = circle[1]
Rc = circle[2]
dc = 2*Rc

plotcircle = circle_fit.plot_data_circle(lateralc, axialc, xc, yc, Rc)

# box on
# daspect([1 1 1])
# zlimits = [min((axial-axial_median))-0.1*(max(axial-axial_median)-...
#     min(axial-axial_median)) max((axial-axial_median))+0.1*(max(axial...
#     -axial_median)-min(axial-axial_median))]; %'zlimits' establishes an axial 
#                                        % range centered at the axial
#                                        % 'median' position, and spanning
#                                        % over an axial length 20% greater
#                                        % than the range given by
#                                        % (max-min)axial positions.
# zlimits_axis = ylim;
# set(handles.axes5, 'Units', 'pixels');
# pos_box = plotboxpos(handles.axes5); % This function will let us match
#                                      % the axis of the relative-z
#                                      % histogram with the axial axis of
#                                      % the 'Combined structures'
#                                      % scatter plot.



# lat_lim = xlim;
# set(handles.lateral_range_3_tag,'string',num2str(round(lat_lim(2))));
# set(handles.lateral_range_1_tag,'string',num2str(round(lat_lim(1))));

# % Plot as vertical lines the lateral limits selected for the relative-z
# % histogram.
#     if str2num(get(handles.lateral_range_3_tag,'string'))...
#             >str2num(get(handles.lateral_range_1_tag,'string'))
#         hold on, plot([str2num(get(handles.lateral_range_1_tag,...
#             'string')) str2num(get(handles.lateral_range_1_tag,'string'))],...
#         [zlimits(1)+5 zlimits(2)-5],'LineStyle',':','LineWidth',1.5,'Color','k');
#         hold on, plot([str2num(get(handles.lateral_range_3_tag,...
#             'string')) str2num(get(handles.lateral_range_3_tag,'string'))],...
#         [zlimits(1)+5 zlimits(2)-5],'LineStyle',':','LineWidth',1.5,'Color','k');      
#     end


# % Histogram of axial positions
# axes(handles.axes4);
# set(handles.axes4,'visible', 'on'); 
# if exist('histogram') >0
#     histogram(axial);
# else
#     hist(axial);
# end
# xlabel('z-position (nm)');
# title('z histogram');
# axes(handles.axes6);
# set(handles.axes6,'visible', 'on'); 

# set(handles.lateral_range_1_tag,'visible', 'on'); 
# set(handles.lateral_range_2_tag,'visible', 'on'); 
# set(handles.lateral_range_3_tag,'visible', 'on');

# set(handles.lateral_range_1_tag,'string',num2str(round(min(lateral-lateral_median))));
# set(handles.lateral_range_3_tag,'string',num2str(round(max(lateral-lateral_median))));

# % Histogram of relative-axial positions
# nbins = zlimits(1):str2num(get(handles.z_bin_tag,'string')):zlimits(2);

# if exist('histogram') >0
#     histogram((axial-axial_median),nbins);
# else
#    hist((axial-axial_median),nbins);
# end
#     set(gca,'xlim',zlimits_axis); 
#     set(gca,'view',[90 -90]) % Rotate histogram
#     set(handles.axes6, 'Units', 'pixels');
#         pos_box1 = get(handles.axes6,'Position'); 
#         pos_box1(4) = pos_box(4);% Match of relative-z histogram z-axis with 
#                                  % the corresponding axial axis from the scatter
#                                  % plot.
#         pos_box1(2) = pos_box(2);
#         set(handles.axes6,'Position',pos_box1);
#         title('relative-z histogram');

