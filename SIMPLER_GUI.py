# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:05:18 2019

@author: Lucia Lopez

GUI for SIMPLER analysis 

conda command for converting QtDesigner file to .py:
pyuic5 -x SIMPLER_GUI_designtabs.ui -o SIMPLER_GUI_design.py
    
"""

import os
import copy


os.chdir(r'C:\Users\Lucia\Documents\NanoFísica\SIMPLER\SIMPLER-master_Python')

import ctypes
import h5py as h5
import pandas as pd
from tkinter import Tk, filedialog
from datetime import date, datetime
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import circle_fit
from skimage.morphology import square, dilation, disk


import pyqtgraph as pg
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import pyqtSignal, pyqtSlot

import SIMPLER_GUI_design
import colormaps as cmaps
import tools.viewbox_tools as viewbox_tools
from matplotlib import cm



# see https://stackoverflow.com/questions/1551605
# /how-to-set-applications-taskbar-icon-in-windows-7/1552105#1552105
# to understand why you need the preceeding two lines
myappid = 'mycompany.myproduct.subproduct.version' # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

class Frontend(QtGui.QMainWindow):
    
    paramSignal = pyqtSignal(dict)
    loadfileSignal = pyqtSignal()
    loadcalibfileSignal = pyqtSignal()
    updatecalSignal = pyqtSignal()
    N0calSignal = pyqtSignal()
    runSIMPLERSignal = pyqtSignal()
    renderSignal = pyqtSignal()


    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        
        self.ui = SIMPLER_GUI_design.Ui_MainWindow()
        self.ui.setupUi(self)
        
                
        self.initialDir = r'Desktop'
        
        self.largeROI = self.ui.groupBox_largeROI
        self.largeROI.hide()
        
        
        self.dF = 0.0
        self.alphaF = 0.0
        
        self.illumcorr = self.ui.checkBox_correction
        self.illumcorr.stateChanged.connect(self.emit_param)
        
        
        fileformat_list = ["Picasso hdf5", "ThunderStorm csv", "custom csv"]
        self.fileformat = self.ui.comboBox_fileformat
        self.fileformat.addItems(fileformat_list)
        self.fileformat.currentIndexChanged.connect(self.emit_param)
        
        
        self.N0calfileformat = self.ui.comboBox_N0calfileformat
        self.N0calfileformat.addItems(fileformat_list)
        self.N0calfileformat.currentIndexChanged.connect(self.emit_param)
        
        self.incanglefileformat = self.ui.comboBox_incanglefileformat
        self.incanglefileformat.addItems(fileformat_list)
        self.incanglefileformat.currentIndexChanged.connect(self.emit_param)
        
        
        NA_list = ["1.42", "1.45", "1.49"]
        self.NA = self.ui.comboBox_NA
        self.NA.addItems(NA_list)
        
        self.browsefile = self.ui.pushButton_browsefile
        self.browsefile.clicked.connect(self.select_file)
        
        self.browseN0calibfile = self.ui.pushButton_browsefile_N0cal
        self.browseN0calibfile.clicked.connect(self.select_N0calfile)
        
        self.browsecalibfile = self.ui.pushButton_browsecalibfile
        self.browsecalibfile.clicked.connect(self.select_calibfile)
        
        self.updatecal = self.ui.pushButton_updatecal
        self.updatecal.clicked.connect(self.update_cal)
        
        self.N0cal = self.ui.pushButton_N0cal
        self.N0cal.clicked.connect(self.N0_cal)
        
        self.runSIMPLER = self.ui.pushButton_runSIMPLER
        self.runSIMPLER.clicked.connect(self.run_SIMPLER)
        
       
        self.pushButton_smallROI = self.ui.pushButton_smallROI
        self.pushButton_smallROI.clicked.connect(self.updateROIPlot)
        
        self.buttonxy = self.ui.radioButtonxy
        self.buttonxz = self.ui.radioButtonxz
        self.buttonyz = self.ui.radioButtonyz
        
        self.pushButton_render = self.ui.pushButton_render
        self.pushButton_render.clicked.connect(self.run_render)
        
        operations_list = ["Small ROI (r,z)", 
                           "Small ROI (x,y,z)",
                           "Large ROI"]
        self.selectop =  self.ui.comboBox_selectop
        self.selectop.addItems(operations_list)
        self.selectop.currentIndexChanged.connect(self.emit_param)
        
        # vertical slider to control point size in scatter
        self.slider = self.ui.verticalSlider    
        self.slider.valueChanged.connect(self.valuechange)
        self.pointsize = 5 

        
    def emit_param(self):
        
        params = dict()
        
        params['fileformat'] = int(self.fileformat.currentIndex())
        params['N0calfileformat'] = int(self.N0calfileformat.currentIndex())
        params['filename'] = self.ui.lineEdit_filename.text()
        params['N0calfilename'] = self.ui.lineEdit_filename_N0cal.text()
        params['calibfilename'] = self.ui.lineEdit_calibfilename.text()
        params['illumcorr'] = self.illumcorr.isChecked()
        params['rz_xyz'] = int(self.selectop.currentIndex())
        params['lambdaex'] = float(self.ui.lineEdit_lambdaex.text())
        params['lambdaem'] = float(self.ui.lineEdit_lambdaem.text())  
        params['angle'] = float(self.ui.lineEdit_angle.text())
        params['alpha'] = float(self.ui.lineEdit_alpha.text())
        params['pxsize'] = float(self.ui.lineEdit_pxsize.text())
        params['ni'] = float(self.ui.lineEdit_ni.text())
        params['ns'] = float(self.ui.lineEdit_ns.text())
        params['N0'] = float(self.ui.lineEdit_N0.text())
        params['NA'] = float(self.ui.comboBox_NA.currentText())
        params['maxdist'] = float(self.ui.lineEdit_maxdist.text())
        params['mag'] = float(self.ui.lineEdit_mag.text())
        params['sigmalat'] = float(self.ui.lineEdit_sigmalat.text())
        params['sigmaax'] = float(self.ui.lineEdit_sigmaax.text())
        
  
        
        self.paramSignal.emit(params)
       
            
    
    def select_file(self):
        try:
            root = Tk()
            root.withdraw()
            root.filenamedata = filedialog.askopenfilename(initialdir=self.initialDir,
                                                      title = 'Select file',
                                                      filetypes = [('hdf5 files','.hdf5'),
                                                                   ('csv file', '.csv')])
            if root.filenamedata != '':
                self.ui.lineEdit_filename.setText(root.filenamedata)
                
        except OSError:
            pass
        
        if root.filenamedata == '':
            return
    
    def select_N0calfile(self):
        try:
            root = Tk()
            root.withdraw()
            root.filenameN0cal = filedialog.askopenfilename(initialdir=self.initialDir,
                                                      title = 'Select N0 calib file',
                                                      filetypes = [('hdf5 files','.hdf5'),
                                                                   ('csv file', '.csv')])
            if root.filenameN0cal != '':
                self.ui.lineEdit_filename_N0cal.setText(root.filenameN0cal)
                
        except OSError:
            pass
        
        if root.filenameN0cal == '':
            return
        
    
    def select_calibfile(self):    
        try:
            root = Tk()
            root.withdraw()
            root.filenamecalib = filedialog.askopenfilename(initialdir=self.initialDir,
                                                      title = 'Select calibration file')
            if root.filenamecalib != '':
                self.ui.lineEdit_calibfilename.setText(root.filenamecalib)
                
        except OSError:
            pass
        
        if root.filenamecalib == '':
            return
    
    
    def update_cal(self):
        
        self.emit_param()
        self.updatecalSignal.emit()
        
    def N0_cal(self):
        
        self.emit_param()
        self.N0calSignal.emit()
        
     
        
    
    def run_SIMPLER(self):
        
        self.emit_param()
        self.updatecalSignal.emit()
        self.runSIMPLERSignal.emit()
    
        
    def valuechange(self):
        
        self.pointsize = self.slider.value()
        self.scatterplot(self.simpler_output)
      
    
    @pyqtSlot(np.ndarray, np.ndarray)    
    def scatterplot(self, simpler_output):  
    
        self.simpler_output = simpler_output
        self.scatter = True
        if self.scatter == True:
            rz_xyz = int(self.selectop.currentIndex())    
    
            
            x = simpler_output[:,0]
            y = simpler_output[:,1]
            r = simpler_output[:,2]
            z = simpler_output[:,3]
            
            ind = np.argsort(z)
            self.xind = x[ind]
            self.yind = y[ind]
            self.zind = z[ind]
            rind = r[ind]
            
            cmapz = cm.get_cmap('viridis', np.size(z))
            
                       
            col = cmapz.colors
            col = np.delete(col, np.s_[3], axis=1)
            col = 255*col
            self.col = col
            
            if rz_xyz == 0:
                                
                scatterWidget = pg.GraphicsLayoutWidget()
                plotrz = scatterWidget.addPlot(title="Scatter plot small ROI (r,z)")
                plotrz.setLabels(bottom=('r [nm]'), left=('z [nm]'))
                plotrz.setAspectLocked(True)
                                     
                
                rz = pg.ScatterPlotItem(rind, self.zind, pen=pg.mkPen(None),
                                        brush=[pg.mkBrush(v) for v in col],
                                        size = self.pointsize)
                plotrz.addItem(rz)
                             
                self.empty_layout(self.ui.scatterlayout)
                self.ui.scatterlayout.addWidget(scatterWidget)
                
            elif rz_xyz == 1:
                
                scatterWidgetxz = pg.GraphicsLayoutWidget()
                plotxz = scatterWidgetxz.addPlot(title="Scatter plot small ROI (x,z)")
                plotxz.setLabels(bottom=('x [nm]'), left=('z [nm]'))
                plotxz.setAspectLocked(True)
                
                
                
                xz = pg.ScatterPlotItem(self.xind, self.zind, pen=pg.mkPen(None),
                                            brush=[pg.mkBrush(v) for v in col],
                                            size = self.pointsize)
                plotxz.addItem(xz)
               
                    
                self.empty_layout(self.ui.scatterlayout)
                self.ui.scatterlayout.addWidget(scatterWidgetxz)
                
                
                scatterWidgetyz = pg.GraphicsLayoutWidget()
                plotyz = scatterWidgetyz.addPlot(title="Scatter plot small ROI (y,z)")
                plotyz.setLabels(bottom=('y [nm]'), left=('z [nm]'))
                plotyz.setAspectLocked(True)
                
        
                
                yz = pg.ScatterPlotItem(self.yind, self.zind, pen=pg.mkPen(None),
                                            brush=[pg.mkBrush(v) for v in col],
                                            size = self.pointsize)
                plotyz.addItem(yz)
               
                    
                self.empty_layout(self.ui.scatterlayout_2)
                self.ui.scatterlayout_2.addWidget(scatterWidgetyz)
                
            elif rz_xyz == 2:
                
                self.largeROI.show()
                scatterWidgetlarge = pg.GraphicsLayoutWidget()
                plotxylarge = scatterWidgetlarge.addPlot(title="Scatter plot large ROI (x,y)")
                plotxylarge.setLabels(bottom=('x [nm]'), left=('y [nm]'))
                plotxylarge.setAspectLocked(True)
                
        
                
                self.xy = pg.ScatterPlotItem(self.xind, self.yind, pen=pg.mkPen(None),
                                            brush=[pg.mkBrush(v) for v in col],
                                            size = self.pointsize)
                plotxylarge.addItem(self.xy)
               
                
                self.empty_layout(self.ui.scatterlayout)
                self.ui.scatterlayout.addWidget(scatterWidgetlarge)
                
                npixels = np.size(x)
                ROIpos = (int(min(x)), int(min(y)))
                ROIextent = int(npixels/3)

                
                ROIpen = pg.mkPen(color='b')
                self.roi = pg.ROI(ROIpos, ROIextent, pen = ROIpen)  
                
                self.roi.setZValue(10)
                self.roi.addScaleHandle([0, 0], [1, 1])
                self.roi.addRotateHandle([0, 0], [1, 1])                             
                plotxylarge.addItem(self.roi)         
                                             
            
            else:
                pass
                
    def updateROIPlot(self):
        
                
        scatterWidgetROI = pg.GraphicsLayoutWidget()
        plotROI = scatterWidgetROI.addPlot(title="Scatter plot ROI selected")
        plotROI.setLabels(bottom=('x [nm]'), left=('y [nm]'))
        plotROI.setAspectLocked(True)
        
        xmin, ymin = self.roi.pos()
        xmax, ymax = self.roi.pos() + self.roi.size()
        
        
        indx = np.where((self.xind>xmin) & (self.xind<xmax))
        indy = np.where((self.yind>ymin) & (self.yind<ymax))
        
        mask = np.in1d(indx, indy)
        
        ind = np.nonzero(mask)
        index=indx[0][ind[0]]
        
        xroi = self.xind[index]
        yroi = self.yind[index]
        zroi = self.zind[index]
        
        if self.buttonxy.isChecked():
            self.selected = pg.ScatterPlotItem(xroi, yroi)    
            
        if self.buttonxz.isChecked():
            self.selected = pg.ScatterPlotItem(xroi, zroi)
            
        if self.buttonyz.isChecked():
            self.selected = pg.ScatterPlotItem(yroi, zroi)
        
        else:
            pass
        
        
        plotROI.addItem(self.selected)
        
        self.empty_layout(self.ui.scatterlayout_3)
        self.ui.scatterlayout_3.addWidget(scatterWidgetROI)
        
    
    def empty_layout(self, layout):
        for i in reversed(range(layout.count())): 
            layout.itemAt(i).widget().setParent(None)
    
        
    def run_render(self):
        
        self.emit_param()
        self.renderSignal.emit()
        
        
    @pyqtSlot(np.ndarray)    
    def disprender(self,B):
         
        rz_xyz = int(self.selectop.currentIndex())
                
        if rz_xyz == 0:
            
            renderWidgetrz = pg.GraphicsLayoutWidget()
          
    
            # image widget set-up and layout
            vb = renderWidgetrz.addPlot(row=0, col=0)
    
            img = pg.ImageItem(B)
            vb.clear()
            vb.addItem(img)
            # self.vb.setAspectLocked(True)
            
            #set up histogram for the rendered image
            hist = pg.HistogramLUTItem(image=img)   #set up histogram for the liveview image
            hist.vb.setLimits(yMin=0, yMax=np.max(B)) 
            for tick in hist.gradient.ticks:
                tick.hide()
            renderWidgetrz.addItem(hist, row=0, col=1)
                
            self.empty_layout(self.ui.renderlayout)        
            self.ui.renderlayout.addWidget(renderWidgetrz)
            
        elif rz_xyz == 1:
           
            renderWidgetxz = pg.GraphicsLayoutWidget()

            
            # image widget set-up and layout
            vb1 = renderWidgetxz.addPlot(row=0, col=0)
            img1 = pg.ImageItem(B[0])
            vb1.clear()
            vb1.addItem(img1)
            # self.vb.setAspectLocked(True)
            
            #set up histogram for the rendered image
            hist1 = pg.HistogramLUTItem(image=img1)   #set up histogram for the liveview image
            
            # hist1.vb1.setLimits(yMin=0, yMax=np.max(B[0])) 
            # for tick in hist1.gradient.ticks:
            #     tick.hide()
            renderWidgetxz.addItem(hist1, row=0, col=1)

                
            self.empty_layout(self.ui.renderlayout)        
            self.ui.renderlayout.addWidget(renderWidgetxz)
            
            
            
            renderWidgetyz = pg.GraphicsLayoutWidget()
    
            # image widget set-up and layout
            vb2 = renderWidgetyz.addPlot(row=0, col=0)
    
            img2 = pg.ImageItem(B[1])
            vb2.clear()
            vb2.addItem(img2)
            # self.vb.setAspectLocked(True)
            
            #set up histogram for the rendered image
            hist2 = pg.HistogramLUTItem(image=img2)   #set up histogram for the liveview image
            # hist2.vb2.setLimits(yMin=0, yMax=np.max(B[1])) 
            # for tick in hist2.gradient.ticks:
            #     tick.hide()
            renderWidgetyz.addItem(hist2, row=0, col=1)
                
            self.empty_layout(self.ui.renderlayout_2)        
            self.ui.renderlayout_2.addWidget(renderWidgetyz)
    
    @pyqtSlot(np.float, np.float)    
    def dispupdateparam(self, dF, alphaF):
        
           
        text = 'dF=' + str(np.round(dF, decimals=2)) +\
            ' alphaF=' + str(np.round(alphaF, decimals=2)) 
        self.ui.lineEdit_updatecal.setText(text)
        
    
    @pyqtSlot(np.float, np.float)    
    def dispNO(self):
        
           
        text = 'N0=' + str(np.round(N0m, decimals=2)) +\
            ' sigmaN0=' + str(np.round(sigmaN0, decimals=2)) 
        self.ui.lineEdit_updatecal.setText(text)
        
        
    
    @pyqtSlot(np.ndarray, np.ndarray)    
    def dispframef(self, simpler_output, frame):
        
        framef = simpler_output[:,5]
            
        text = '# of localizations from raw data = ' + str(int(len(frame))) + '\n' +\
            ' # of valid localizations (after filter) = ' + str(int(len(framef))) 
        self.ui.textEdit_framef.setText(text)
        
        
        
    def make_connection(self, backend):
        
        backend.sendupdatecalSignal.connect(self.dispupdateparam)
        backend.sendSIMPLERSignal.connect(self.scatterplot)
        backend.sendSIMPLERSignal.connect(self.dispframef)
        backend.sendrenderSignal.connect(self.disprender)
        

        
                



class Backend(QtCore.QObject):

    paramSignal = pyqtSignal(dict)
    sendupdatecalSignal = pyqtSignal(np.float, np.float)
    sendSIMPLERSignal = pyqtSignal(np.ndarray, np.ndarray)
    sendrenderSignal = pyqtSignal(np.ndarray)
    
        
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
    
        
    @pyqtSlot(dict)
    def get_frontend_param(self, params):
        
        # updates parameters according to what is input in the GUI
        self.fileformat = params['fileformat']
        self.filename = params['filename']
        self.N0calfileformat = params['N0calfileformat']
        self.N0calfilename = params['N0calfilename']
        self.calibfilename = params['calibfilename']
        self.rz_xyz = params['rz_xyz']
        self.illum = params['illumcorr']
        self.lambdaex = params['lambdaex']
        self.lambdaem = params['lambdaem'] 
        self.angle = params['angle']
        self.alpha = params['alpha']
        self.pxsize = params['pxsize']
        self.ni = params['ni']
        self.ns = params['ns']
        self.N0 = params['N0']
        self.NA = params['NA']
        self.maxdist = params['maxdist']
        self.mag = params['mag']
        self.sigma_lat = params['sigmalat']
        self.sigma_ax = params['sigmaax']
        
        
            
    @pyqtSlot()
    def getParameters_SIMPLER(self):
                
        # Angle
        if self.angle == 0:
            self.angle = np.arcsin(self.NA/self.ni)
        else:
            self.angle = np.deg2rad(self.angle)
        
        
        # Alpha
        if self.alpha == 0:
            self.alpha = 0.9
        else:
            self.alpha = self.alpha
            
        # Z
        z_fit = np.arange(5, 500, 0.5)
        z =[5, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        
        # Lambda
        lambda_em_max =[500, 530, 560, 590, 620, 670, 700, 720]
        dif = np.min(abs(lambda_em_max - np.ones(np.size(lambda_em_max))*self.lambdaem))
        i_lambda_em = np.argmin(abs(lambda_em_max - np.ones(np.size(lambda_em_max))*self.lambdaem))
    
        
        # Axial dependece of the excitation field
        d = self.lambdaex/(4 * np.pi * np.sqrt(self.ni**2*(np.sin(self.angle)**2) - self.ns**2))
        I_exc = self.alpha * np.exp(-z_fit/d) + (1-self.alpha)
        
        # Axial dependece of the fraction of fluorescence collected by a microscope objetive
        
        if self.NA == 1.42:
            DF_NA1_42 = np.loadtxt('DF_NA1.42.txt')
            DFi = DF_NA1_42[:,i_lambda_em]
        elif self.NA == 1.45:
            DF_NA1_45 = np.loadtxt('DF_NA1.45.txt')
            DFi = DF_NA1_45[:,i_lambda_em]
        elif self.NA == 1.49:
            DF_NA1_49 = np.loadtxt('DF_NA1.49.txt')
            DFi = DF_NA1_49[:,i_lambda_em]
            
         
        DFi_interp = interp1d(z,DFi)(z_fit)
        I_total = I_exc * DFi_interp
        #plt.plot(z, DFi, 'o', z_fit, DFi_interp, '-')
        #plt.show()
           
        # Fit F to "F = alphaF*exp(-z/dF)+(1-alphaF) 
        def fitF(x, a, b, c):
            return a * np.exp(-b*x) + c
        
                                     
        popt, pcov = curve_fit(fitF, z_fit, I_total, p0 = [0.6, 0.01, 0.05])
        
        # check the goodness of fit by plotting
    #    plt.plot(z_fit, I_total, 'b-', label='data')
    #    plt.plot(z_fit, fitF(z_fit, *popt), 'r-',
    #         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        
        self.alphaF = 1-popt[2]
        self.dF = 1/popt[1]
        
        self.sendupdatecalSignal.emit(self.dF, self.alphaF)
       
    
    def import_file(self):
        
        #File Importation
        if self.fileformat == 0: # Importation procedure for Picasso hdf5 files.
            
            # Read H5 file
            f = h5.File(self.filename, "r")
            dataset = f['locs']
        
            # Load  input HDF5 file
            frame = dataset['frame']
            photon_raw = dataset['photons']
            bg = dataset['bg']          
            
            xdata = dataset['x'] 
            ydata = dataset['y'] 
        
        
        elif self.fileformat == 1: # Importation procedure for ThunderSTORM csv files.
            
            ## Read ThunderSTRORM csv file
            dataset = pd.read_csv(self.filename)
            # Extraxt headers names
            headers = dataset.columns.values
            
            # data from different columns           
            frame = dataset[headers[0]].values
            xdata = dataset[headers[1]].values 
            ydata = dataset[headers[2]].values
            photon_raw = dataset[headers[3]].values
            bg = dataset[headers[4]].values
            
        else: # Importation procedure for custom csv files.
        
        # TODO : import custom csv file 
            pass 
#            full_list = csvread(filename_wformat);
#            frame = full_list(:,1);
#            xloc = full_list(:,2);
#            yloc = full_list(:,3);
#            photon_raw = full_list(:,4);
#            if size(full_list,2)>4
#                bg = full_list(:,5);

        return xdata, ydata, frame, photon_raw, bg

    
    def filter_locs(self, x, y, frame, phot_corr, max_dist):
         
        # Build the output array
        listLocalizations = np.column_stack((x, y, frame, phot_corr))
             
        #REMOVING LOCS WITH NO LOCS IN i-1 AND i+1 FRAME

        # We keep a molecule if there is another one in the previous and next frame,
        # located at a distance < max_dist, where max_dist is introduced by user
        # (20 nm by default).

        min_div = 1000
        
        # We divide the list into sub-lists of 'min_div' locs to minimize memory usage

        Ntimes_min_div = int((listLocalizations[:,0].size/min_div))
        
        truefalse_sum_roi_acum = []
        
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
        
                       
            for i in range(min_range, max_range):
                for j in range(min_range, max_range):
                    daa[j-min_div_N*(N)] = ((x[i]-x[j])**2+(y[i]-y[j])**2)**(1/2)
                    frame_dif[j-min_div_N*(N)] = ((listLocalizations[i,2] - listLocalizations[j,2])**2)**(1/2)
                    if daa[(j-min_div_N*(N))] < max_dist and frame_dif[(j-min_div_N*(N))] == 1:
                        truefalse[(i-min_range),(j-min_range)] = 1
            
            truefalse_sum = truefalse.sum(axis=0)  
            # For each row (i.e. each molecule) we calculate the sum of every
            # column from the 'truefalse' matrix.
       
            truefalse_sum_roi_acum = np.append(truefalse_sum_roi_acum, truefalse_sum)
            
        # We choose the indexes of those rows whose columnwise sum is > or = to 2
        idx_filtered = np.where(truefalse_sum_roi_acum > 1)
        
        # APPLYING FILTER TO THE ORIGINAL LIST 

        x = listLocalizations[idx_filtered,0].T
        y = listLocalizations[idx_filtered,1].T
        photons = listLocalizations[idx_filtered,3].T
        framef = listLocalizations[idx_filtered,2].T
        
        x = x.flatten()
        y = y.flatten()
        
        return x,y,photons,framef
        
    def illum_correct(self, calibfilename, photon_raw, xdata, ydata):
        
        # To perform this correction, the linear dependency between local laser
        # power intensity and local background photons is used. Photons are
        # converted to the value they would have if the whole image was illuminated
        # with the maximum laser power intensity. This section is executed if the
        # user has chosen to perform correction due to non-flat illumination.
        
        datacalib = pd.read_csv(self.calibfilename)
        profiledata = pd.DataFrame(datacalib)
        profile = profiledata.values
        
        
        phot = photon_raw
        max_bg = np.percentile(profile, 97.5)
        phot_corr = np.zeros(photon_raw.size)
        
        # Correction loop
        profx = np.size(profile,0) 
        profy = np.size(profile,1) 
        
        for i in np.arange(len(phot)):
            if int((np.ceil(xdata[i]))) < profx and int((np.ceil(ydata[i]))) < profy:
                phot_corr[i] = phot[i]*(max_bg)/(profile[int(np.ceil(xdata[i])),int(np.ceil(ydata[i]))])
            elif int((np.ceil(xdata[i]))) > profx and int((np.ceil(ydata[i]))) < profy:
                phot_corr[i] = phot[i]*(max_bg)/(profile[int(np.floor(xdata[i])),int(np.ceil(ydata[i]))])
            elif int((np.ceil(xdata[i]))) < profx and int((np.ceil(ydata[i]))) > profy:
                phot_corr[i] = phot[i]*(max_bg)/(profile[int(np.ceil(xdata[i])),int(np.floor(ydata[i]))])
            elif int((np.ceil(xdata[i]))) > profx and int((np.ceil(ydata[i]))) > profy:
                phot_corr[i] = phot[i]*(max_bg)/(profile[int(np.floor(xdata[i])),int(np.floor(ydata[i]))])
        
        return phot_corr
                    
    
    @pyqtSlot()   
    def SIMPLER_function(self):
         
        xdata, ydata, frame, photon_raw, bg = self.import_file()
        
        # Convert x,y,sd values from 'camera subpixels' to nanometres
        x = xdata * self.pxsize
        y = ydata * self.pxsize
        
        # Correct photon counts if illum profile not uniform 
        if self.illum == True:
                        
            calibfilename = self.calibfilename
            phot_corr = self.illum_correct(calibfilename, photon_raw, xdata, ydata)
                    
        else:
            phot_corr = photon_raw
        
        # Filter localizations using max dist
        max_dist = self.maxdist # Value in nanometers
               
        self.x,self.y,photons,framef = self.filter_locs(x, y, frame, phot_corr, max_dist)

                   
        # SIMPLER z estimation
        z1 = (np.log(self.alphaF*self.N0)-np.log(photons-(1-self.alphaF)*self.N0))/(1/self.dF)
        z = np.real(z1)
        self.z = z.flatten()
        
        # Compute radial coordinate r from (x,y)   
        P = np.polyfit(self.x,self.y,1)
        
        def Poly_fun(x):
            y_polyfunc = P[0]*x + P[1]
            return y_polyfunc
        
        Origin_X = 0.999999*min(self.x)
        Origin_Y = Poly_fun(Origin_X)
        
        # Change from cartesian to polar coordinates
        tita = np.arctan(P[0])
        tita1 = np.arctan((self.y-Origin_Y)/(self.x-Origin_X))
        
        r = ((self.x-Origin_X)**2+(self.y-Origin_Y)**2)**(1/2)
        tita2 = [x - tita for x in tita1]
        self.r = np.cos(tita2)*r
        
                   
        simpler_output = np.column_stack((self.x, self.y, self.r, self.z, photons, framef))
        
        self.sendSIMPLERSignal.emit(simpler_output, frame)
    
    @pyqtSlot()
    def N0_calibration(self):
     
        #File Importation
        xdata, ydata, frame, photon_raw, bg = self.import_file() 
        
        # Convert x,y,sd values from 'camera subpixels' to nanometres
        x = xdata * self.pxsize
        y = ydata * self.pxsize
        
        # Correct photon counts if illum profile not uniform 
        if self.illum == True:
            
            calibfilename = self.calibfilename
            phot_corr = self.illum_correct(calibfilename, photon_raw, xdata, ydata)
                    
        else:
            phot_corr = photon_raw
            
        # Filter localizations using max dist
        max_dist = self.maxdist # Value in nanometers
              
        x,y,photons,framef = self.filter_locs(x, y, frame, phot_corr, max_dist)
        
        # For the "N0 Calibration" operation, there is no "Z calculation", 
        # because the aim of this procedure is to obtain N0 from a sample which 
        # is supposed to contain molecules located at z ~ 0.
        xl = np.array([np.amax(x), np.amin(x)]) 
        yl = np.array([np.amax(y), np.amin(y)]) 
        c = np.arange(0,np.size(x))
        
        hist, bin_edges = np.histogram(photons[c], bins = 20, density = True)
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
        N0m = coeff[1]
        sigmaN0 = coeff[2]
        
        
        
    def inc_angle_alpha(self):
        
        #File Importation
        xdata, ydata, frame, photon_raw, bg = self.import_file()   
        
        # Convert x,y,sd values from 'camera subpixels' to nanometres
        x = xdata * self.pxsize
        y = ydata * self.pxsize
        
    def obtain_bg(self):
        
        xdata, ydata, frame, photon_raw, bg = self.import_file()
        
        # Convert x,y,sd values from 'camera subpixels' to nanometres
        x = xdata * self.pxsize
        y = ydata * self.pxsize
        
        ## Excitation profile calculation and exportation:
        Img_bg = np.zeros((np.int(np.ceil(np.max(xdata))), np.int(np.ceil(np.max(ydata)))))
        # Empty matrix to be filled with background values pixel-wise.
             
        Count_molec_px = Img_bg # Empty matrix to be filled with the number of molecules
                                     # used to calcualte local background for each pixel.
             
        # If the list contains > 1,000,000 localizations, only 500,000 are used in
        # order to speed up the analysis and avoid redundancy.
        vector_indices = np.arange(0,np.size(xdata))
        c_random = np.random.permutation(np.size(vector_indices))
        length_c_random = np.int(np.min([1e+6, np.size(x)]))
        range_c = range(0,length_c_random)
        c = c_random[range_c]
        
        
        for i in range(0, np.size(x[c])):
            
            # The #molecules used to calculate local background in current pixel is updated.
            xind = np.int(np.ceil(xdata[c[i]]))-1
            yind = np.int(np.ceil(ydata[c[i]]))-1
            Count_molec_px[xind, yind] = Count_molec_px[xind, yind] + 1
            Count_i = Count_molec_px[xind, yind]
            
            # If the current pixel's background has already been estimated, then its 
            # value is updated through a weighted average:
            count_div = (Count_i-1)/Count_i
            if Count_i > 1:
                Img_bg[xind, yind] = count_div * (Img_bg[xind, yind]) + (1/Count_i) * bg[c[i]] 
            else:
                Img_bg[xind, yind] = bg[c[i]]
   
    @pyqtSlot()
    def render(self):
                       
        if self.rz_xyz == 0:
            
            lat = self.r
            ax = self.z
            B = self.gauss_render(lat,ax)

            
            
        elif self.rz_xyz == 1:
            
            ax = self.z
            lat1 = self.x
            B1 = self.gauss_render(lat1, ax)
            lat2 = self.y
            B2 = self.gauss_render(lat2, ax)
            B = ((B1,B2))
            B = np.asarray(B)   

        
        self.sendrenderSignal.emit(B)
        
    def gauss_render(self, lat, ax):
        
        # define px size in the SR image (in nm)
        self.pxsize_render = self.pxsize/self.mag
        sigma_latpx = self.sigma_lat/self.pxsize_render
        sigma_axpx = self.sigma_ax/self.pxsize_render
        
        # re define origin for lateral and axial coordinates
        self.r_ori = lat-min(lat) + self.sigma_lat
        self.z_ori = ax-min(ax) + self.sigma_ax
        
        # re define max
        max_r = (max(lat)-min(lat)) + self.sigma_lat
        max_z = (max(ax)-min(ax)) + self.sigma_ax
       

        # from nm to px
        SMr = self.r_ori/self.pxsize_render
        SMz = self.z_ori/self.pxsize_render
        
        # Definition of pixels affected by the list of SM (+- 5*sigma)
        A = np.zeros((np.int(np.ceil(max_r/self.pxsize_render)), np.int(np.ceil(max_z/self.pxsize_render))))

        for i in np.arange(len(SMr)):
            A[int(np.floor(SMr[i])), int(np.floor(SMz[i]))] = 1    
        
        sigma_width_nm = np.max([self.sigma_lat, self.sigma_ax]);
        sigma_width_px = sigma_width_nm/self.pxsize_render;
        sd = disk(int(np.round(5*sigma_width_px)));
        # This matrix contains 1 in +- 5 sigma units around the SM positions
        A_affected = dilation(A,sd);
        # r and z positions affected
        indaffected = np.where(A_affected==1)
        raffected = indaffected[0]
        zaffected = indaffected[1]
        
        #'PSF' is a function that calculates the value for a given position (r,z)
        # assuming a 2D Gaussian distribution centered at (SMr(k),SMz(k))
        def PSF(r, z, SMr, SMz, I):
            psf = (I/(2*np.pi*sigma_latpx*sigma_axpx))*np.exp(-(((r-SMr)**2)/(2*sigma_latpx**2) + ((z-SMz)**2)/(2*sigma_axpx**2)))
            return psf
        
        # B = empty matrix that will be filled with the values given by the
        # Gaussian blur of the points listed in (SMr, SMz)
        B = np.zeros((np.int(np.ceil(5*sigma_width_px+(max_r/self.pxsize_render))), 
              np.int(np.ceil(5*sigma_width_px+(max_z/self.pxsize_render)))))
        
        # For each molecule from the list
        for k in np.arange(len(SMr)):
        # For each pixel of the final image with a value different from zero
            for i in np.arange(len(raffected)): 
                B[raffected[i],zaffected[i]] = B[raffected[i],zaffected[i]] + PSF(raffected[i],zaffected[i],SMr[k],SMz[k],1)
                # Each 'affected' pixel (i) will take the value that had at the beggining
                # of the k-loop + the value given by the distance to the k-molecule.
        
        return B
    
    
    def fit_circle(self):
       
       pass

            
    def make_connection(self, frontend):
        
        frontend.paramSignal.connect(self.get_frontend_param)
        frontend.updatecalSignal.connect(self.getParameters_SIMPLER)
        frontend.N0calSignal.connect(self.N0_calibration)
        frontend.runSIMPLERSignal.connect(self.SIMPLER_function)
        frontend.renderSignal.connect(self.render)
        
        
        
                
        
if __name__ == '__main__':
    
    if not QtGui.QApplication.instance():
        app = QtGui.QApplication([])
    else:
        app = QtGui.QApplication.instance()

    
    icon_path = r'logo3.jpg'
    app.setWindowIcon(QtGui.QIcon(icon_path))
    
    
    worker = Backend()    
    gui = Frontend()
    
    gui.emit_param()
    worker.make_connection(gui)
    gui.make_connection(worker)
         
    

    gui.setWindowIcon(QtGui.QIcon(icon_path))
    gui.show() #Maximized()
    #gui.showFullScreen()
        
    # app.exec_()     
