# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 13:06:32 2021

@author: Lucia
"""
import os
import numpy as np
import cv2
from skimage import io
from scipy import signal
import skimage
import matplotlib.pyplot as plt


os.chdir(r'C:\Users\Lucia\Documents\NanoFísica\SIMPLER\beam displacement')


ims = io.imread('Stack_172.tif')
for i in np.arange(11):
    im0 = ims[i,:,:]
    im0 = im0.astype(np.uint8)
    
    # # Task a
    # equ = cv2.equalizeHist(im0) 

    
    # # Task b (attempt #1)
    # kernel = np.array([[-1,-1,-1], [-1,25,-1], [-1,-1,-1]])
    # im = cv2.filter2D(equ, -1, kernel)
    

    
    # # Task b (attempt #2)
    # psf = np.ones((1, 1)) / 25
    # equ = signal.convolve2d(im0, psf, 'same')
    # im, _ = skimage.restoration.unsupervised_wiener(im0, psf)
    

    
    # Task c
    equCopy = np.uint8(im0)
    edges = cv2.Canny(equCopy,270,270)
    plt.figure()
    plt.imshow(edges)