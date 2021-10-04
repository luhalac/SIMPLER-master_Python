# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:05:18 2019

@author: Lucia Lopez

GUI for SIMPLER analysis 

conda command for converting QtDesigner file to .py:
pyuic5 -x SIMPLER_GUI_design.ui -o SIMPLER_GUI_design.py
    
"""

import os

os.chdir(r'C:\Users\Lucia\Documents\NanoFísica\SIMPLER\SIMPLER-master_Python')

import ctypes
from skimage import io
import h5py as h5
import pandas as pd
from tkinter import Tk, filedialog
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from circlefit import CircleFit



import pyqtgraph as pg
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import pyqtSignal, pyqtSlot

import SIMPLER_GUI_design
from matplotlib import cm



# see https://stackoverflow.com/questions/1551605
# /how-to-set-applications-taskbar-icon-in-windows-7/1552105#1552105
# to understand why you need the preceeding two lines
myappid = 'mycompany.myproduct.subproduct.version' # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

class Frontend(QtGui.QMainWindow):
    
    # define frontend signals
    paramSignal = pyqtSignal(dict)
    loadfileSignal = pyqtSignal()
    loadcalibfileSignal = pyqtSignal()
    updatecalSignal = pyqtSignal()
    finetuneSignal = pyqtSignal()
    finetunefitSignal = pyqtSignal()
    N0calSignal = pyqtSignal()
    obtainbgSignal = pyqtSignal()
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
        self.illumcorr = self.ui.checkBox_correction_2
        self.illumcorr.stateChanged.connect(self.emit_param)
        
        self.fitcircle1 = self.ui.checkBox_fitcircle
        self.illumcorr.stateChanged.connect(self.fine_tune)
        
        
        
        fileformat_list = ["Picasso hdf5", "ThunderStorm csv", "custom csv"]
        self.fileformat = self.ui.comboBox_fileformat
        self.fileformat.addItems(fileformat_list)
        self.fileformat.currentIndexChanged.connect(self.emit_param)
        
        
        self.N0fileformat = self.ui.comboBox_N0fileformat
        self.N0fileformat.addItems(fileformat_list)
        self.N0fileformat.currentIndexChanged.connect(self.emit_param)
        
        self.bgfileformat = self.ui.comboBox_bgfileformat
        self.bgfileformat.addItems(fileformat_list)
        self.bgfileformat.currentIndexChanged.connect(self.emit_param)
       
        
        NA_list = ["1.42", "1.45", "1.49"]
        self.NA = self.ui.comboBox_NA
        self.NA.addItems(NA_list)
        
        self.browsefile = self.ui.pushButton_browsefile
        self.browsefile.clicked.connect(self.select_file)
        
        self.browseN0file = self.ui.pushButton_N0browsefile
        self.browseN0file.clicked.connect(self.select_N0file)
        
        self.browsebgfile = self.ui.pushButton_browsefilebg
        self.browsebgfile.clicked.connect(self.select_bgfile)
        
        self.browsecalibfile = self.ui.pushButton_browsecalibfile
        self.browsecalibfile.clicked.connect(self.select_calibfile)
        
        self.browseN0calibfile = self.ui.pushButton_N0browsecalibfile
        self.browseN0calibfile.clicked.connect(self.select_N0calibfile)
        
        self.browsetunefile = self.ui.pushButton_browsefile_tune
        self.browsetunefile.clicked.connect(self.select_tunefile)
        
        self.updatecal = self.ui.pushButton_updatecal
        self.updatecal.clicked.connect(self.update_cal)
        
        self.N0cal = self.ui.pushButton_N0cal
        self.N0cal.clicked.connect(self.N0_cal)
        
        self.runSIMPLER = self.ui.pushButton_runSIMPLER
        self.runSIMPLER.clicked.connect(self.run_SIMPLER)
        
        self.finetune = self.ui.pushButton_tune
        self.finetune.clicked.connect(self.fine_tune)
        
        self.finetunefit = self.ui.pushButton_tune_fit
        self.finetunefit.clicked.connect(self.tunefit)
        
        self.bg = self.ui.pushButton_excprof
        self.bg.clicked.connect(self.obtainbg)
        
       
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
        
        # lateral range and binning for fine tunning rel z hist
        self.latmin = self.ui.lineEdit_latmin
        self.latmax = self.ui.lineEdit_latmax
        self.nbins = self.ui.lineEdit_bin
        
        self.latmin.textChanged.connect(self.latchange)
        self.latmax.textChanged.connect(self.latchange)
        self.nbins.textChanged.connect(self.latchange)
        
        self.lmin = None
        self.lmax = None
        self.bins = None
        
        # beam displacement and TIRF angle cal
        
        self.browsebeamfile = self.ui.pushButton_browsefilebeam
        self.browsebeamfile.clicked.connect(self.select_beamfile)
        
        self.showbeam = self.ui.pushButton_stack
        self.showbeam.clicked.connect(self.dispbeam)
       
        # define colormap
        
        cmap = cm.get_cmap('viridis', 100)
        colors = cmap.colors
        colors = np.delete(colors, np.s_[3], axis=1)
        col = 255*colors
        self.vir = col.astype(int)
        
       
        self.brush1 = pg.mkBrush(self.vir[20])
        self.brush2 = pg.mkBrush(self.vir[40])
        self.brush3 = pg.mkBrush(self.vir[70])
        self.pen1 = pg.mkPen(self.vir[20])
        self.pen2 = pg.mkPen(self.vir[40])
        self.pen3 = pg.mkPen(self.vir[70])
      


        
    def emit_param(self):
        
        params = dict()
        
        params['fileformat'] = int(self.fileformat.currentIndex())
        params['N0fileformat'] = int(self.N0fileformat.currentIndex())
        params['filename'] = self.ui.lineEdit_filename.text()
        params['N0filename'] = self.ui.lineEdit_N0filename.text()
        params['tunefilename'] = self.ui.lineEdit_filename_tune.text()
        params['calibfilename'] = self.ui.lineEdit_calibfilename.text()
        params['N0calibfilename'] = self.ui.lineEdit_N0calibfilename.text()
        params['bgfileformat'] = int(self.bgfileformat.currentIndex())
        params['bgfilename'] = self.ui.lineEdit_bgfilename.text()
        params['illumcorr'] = self.illumcorr.isChecked()
        params['rz_xyz'] = int(self.selectop.currentIndex())
        params['lambdaex'] = float(self.ui.lineEdit_lambdaex.text())
        params['lambdaem'] = float(self.ui.lineEdit_lambdaem.text())  
        params['angle'] = float(self.ui.lineEdit_angle.text())
        params['alpha'] = float(self.ui.lineEdit_alpha.text())
        params['angletune'] = float(self.ui.lineEdit_angletune.text())
        params['alphatune'] = float(self.ui.lineEdit_alphatune.text())
        params['rangeangle'] = float(self.ui.lineEdit_rangeang.text())
        params['stepsangle'] = float(self.ui.lineEdit_stepang.text())
        params['rangealpha'] = float(self.ui.lineEdit_rangealpha.text())
        params['stepsalpha'] = float(self.ui.lineEdit_stepalpha.text())
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
                                                      title = 'Select file')
            if root.filenamedata != '':
                self.ui.lineEdit_filename.setText(root.filenamedata)
                
        except OSError:
            pass
        
        if root.filenamedata == '':
            return
    
    def select_N0file(self):
        try:
            root = Tk()
            root.withdraw()
            root.filenameN0cal = filedialog.askopenfilename(initialdir=self.initialDir,
                                                      title = 'Select N0 calib file')
            if root.filenameN0cal != '':
                self.ui.lineEdit_N0filename.setText(root.filenameN0cal)
                
        except OSError:
            pass
        
        if root.filenameN0cal == '':
            return
        
    def select_bgfile(self):
        try:
            root = Tk()
            root.withdraw()
            root.filenamebg = filedialog.askopenfilename(initialdir=self.initialDir,
                                                      title = 'Select N0 calib file')
            if root.filenamebg != '':
                self.ui.lineEdit_bgfilename.setText(root.filenamebg)
                
        except OSError:
            pass
        
        if root.filenamebg == '':
            return
        
    def select_tunefile(self):
        try:
            root = Tk()
            root.withdraw()
            root.filenametune = filedialog.askopenfilename(initialdir=self.initialDir,
                                                      title = 'Select tunning file')
            if root.filenametune != '':
                self.ui.lineEdit_filename_tune.setText(root.filenametune)
                
        except OSError:
            pass
        
        if root.filenametune == '':
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
    
    def select_N0calibfile(self):    
        try:
            root = Tk()
            root.withdraw()
            root.filenameN0calib = filedialog.askopenfilename(initialdir=self.initialDir,
                                                      title = 'Select calibration file')
            if root.filenameN0calib != '':
                self.ui.lineEdit_N0calibfilename.setText(root.filenameN0calib)
                
        except OSError:
            pass
        
        if root.filenameN0calib == '':
            return
    
    def select_beamfile(self):    
        try:
            root = Tk()
            root.withdraw()
            root.filenamestack = filedialog.askopenfilename(initialdir=self.initialDir,
                                                      title = 'Select stack')
            if root.filenamestack != '':
                self.ui.lineEdit_stackfile.setText(root.filenamestack)
                
        except OSError:
            pass
        
        if root.filenamestack == '':
            return
    
    def update_cal(self):
        
        self.emit_param()
        self.updatecalSignal.emit()
        
    def N0_cal(self):
        
        self.emit_param()
        self.N0calSignal.emit()
        
    def obtainbg(self):
        
        self.emit_param()
        self.obtainbgSignal.emit()
        
    def fine_tune(self):
        
        self.emit_param()
        self.finetuneSignal.emit()
        
    def tunefit(self): 

        
        self.emit_param()
        self.finetunefitSignal.emit()
     
    
    def run_SIMPLER(self):
        
        self.emit_param()
        self.updatecalSignal.emit()
        self.runSIMPLERSignal.emit()
    
        
    def valuechange(self):
        
        self.pointsize = self.slider.value()
        self.scatterplot(self.simpler_output)
        self.updateROIPlot()
    
    def latchange(self):
        
        self.bins = int(self.nbins.text())
        self.lmin = float(self.latmin.text())
        self.lmax = float(self.latmax.text())
        
      
        self.dispfinetune(self.finetune_output)

    
    @pyqtSlot(np.ndarray, np.ndarray)    
    def scatterplot(self, simpler_output):  
    
        self.simpler_output = simpler_output

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
            
            self.largeROI.hide()                
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
            
            self.largeROI.hide()
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
            
            self.empty_layout(self.ui.scatterlayout_2)
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
            self.roi.addScaleHandle([1, 1], [0, 0])
            self.roi.addRotateHandle([0, 0], [1, 1])                             
            plotxylarge.addItem(self.roi)         
                                             

                
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
            self.selected = pg.ScatterPlotItem(xroi, yroi, pen = self.pen1,
                                               brush = None, size = self.pointsize)    
            
        if self.buttonxz.isChecked():
            self.selected = pg.ScatterPlotItem(xroi, zroi, pen=self.pen2,
                                               brush = None, size = self.pointsize)
            
        if self.buttonyz.isChecked():
            self.selected = pg.ScatterPlotItem(yroi, zroi, pen=self.pen3,
                                               brush = None, size = self.pointsize)
        
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
            hist.gradient.loadPreset('viridis')
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
            hist1.vb.setLimits(yMin=0, yMax=np.max(B[0])) 
            for tick in hist1.gradient.ticks:
                tick.hide()
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
            hist2.vb.setLimits(yMin=0, yMax=np.max(B[1])) 
            for tick in hist2.gradient.ticks:
                tick.hide()
            renderWidgetyz.addItem(hist2, row=0, col=1)
                
            self.empty_layout(self.ui.renderlayout_2)        
            self.ui.renderlayout_2.addWidget(renderWidgetyz)
            
        elif rz_xyz == 2:
            
            renderWidgetxy = pg.GraphicsLayoutWidget()
          
    
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
            renderWidgetxy.addItem(hist, row=0, col=1)
                
            self.empty_layout(self.ui.renderlayout)        
            self.ui.renderlayout.addWidget(renderWidgetxy)
     
    
    @pyqtSlot(np.ndarray)
    def dispfinetune(self, finetune_output):
        
        self.finetune_output = finetune_output
             
        photons = finetune_output[:,0]
        photons_median = finetune_output[:,1]
        lateral = finetune_output[:,2]
        lateral_median = finetune_output[:,3]
        axial = finetune_output[:,4]
        axial_median = finetune_output[:,5]
        
        lateralc = lateral - lateral_median
        axialc = axial - axial_median
        
        
        # # zmin and zmax establishes an axial range centered at the axial
        # # 'median' position, and spanning over an axial length 20% greater
        # # than the range given by (max-min)axial positions.
        # zmin = min(axialc) - 0.1*(max(axialc)-min(axialc))
        # zmax = max(axialc) + 0.1*(max(axialc)-min(axialc))
       
        self.fitcircle = self.fitcircle1.isChecked()

        if self.fitcircle == True:
            
            xy = np.vstack((lateralc, axialc)).T
            circle = CircleFit(xy)
            xc = circle[0]
            yc = circle[1]
            Rc = circle[2]
            dc = 2*Rc
            
            diameter = str(np.round(dc, decimals=2)) 
            self.ui.lineEdit_D.setText(diameter)
            
            theta_fit = np.linspace(-np.pi, np.pi, 180)

            xfit = xc + Rc*np.cos(theta_fit)
            yfit = yc + Rc*np.sin(theta_fit)
            
        else:
            
            dc = []
            xfit = []
            yfit = []
        
        # scatter plot of aligned known structures
        scattertuneWidget = pg.GraphicsLayoutWidget()
        plot = scattertuneWidget.addPlot(title="Combined structures")
        plot.setLabels(bottom=('r [nm]'), left=('z [nm]'))
        plot.setAspectLocked(True)
                             
        penfit = pg.mkPen(c='#8f9805', width = 2)   
        
      
        rz = pg.ScatterPlotItem((lateralc),(axialc),brush=None,pen=self.pen1,size=3)
        circlefit = pg.PlotCurveItem(xfit, yfit, pen = penfit)
        plot.addItem(rz)
        plot.addItem(circlefit)
                     
        self.empty_layout(self.ui.tunelayout)
        self.ui.tunelayout.addWidget(scattertuneWidget)
        
        # histogram of relative axial positions of known structures
        histzrelWidget = pg.GraphicsLayoutWidget()
        histz = histzrelWidget.addPlot(title="Relative z")
               
        
        if self.lmin != None:
            
            axialc = axialc[(lateralc>self.lmin) & (lateralc<self.lmax)]
            
        else:
            pass
        
        if self.bins != None:
            
            bins = self.bins
        else:
            bins = 20
        
        hist, bin_edges = np.histogram(axialc, bins=bins)
        widthz = np.mean(np.diff(bin_edges))
        bincenters = np.mean(np.vstack([bin_edges[0:-1],bin_edges[1:]]), axis=0)
        bargraph = pg.BarGraphItem(x0 = 0, y = bincenters, height = widthz, width = hist, brush = self.brush1)
        histz.addItem(bargraph)
        
        self.empty_layout(self.ui.tunelayouthist)
        self.ui.tunelayouthist.addWidget(histzrelWidget)
        
        # histogram of axial positions of known structures
        histzWidget = pg.GraphicsLayoutWidget()
        histabsz = histzWidget.addPlot(title="z Histogram")
        
        histz, bin_edgesz = np.histogram(axial, bins=20)
        widthzabs = np.mean(np.diff(bin_edgesz))
        bincentersz = np.mean(np.vstack([bin_edgesz[0:-1],bin_edgesz[1:]]), axis=0)
        bargraphz = pg.BarGraphItem(x = bincentersz, height = histz, width = widthzabs, brush = self.brush3)
        histabsz.addItem(bargraphz)
                
        self.empty_layout(self.ui.zlayouthist)
        self.ui.zlayouthist.addWidget(histzWidget)
        
        # fit for diff angles and alpha

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def fitfinetune(self, angles, alphas, D):
        
        # plot diameter for different parameters
        fitplotWidget = pg.GraphicsLayoutWidget()
        plot = fitplotWidget.addPlot(title="fine tuning")
        plot.setLabels(bottom=('Angle [º]'), left=('D [nm]'))
        # plot.setAspectLocked(True)
                             
                
        for i in np.arange(len(alphas)):
            Dvsang = pg.PlotCurveItem(angles, D[:,i], 
                pen = pg.mkPen(self.vir[15*(i+1)],width = 2),
                name = 'α ='+ str(alphas[i]))
            plot.addItem(Dvsang)
            plot.addLegend()
        
            
                     
        self.empty_layout(self.ui.tunefitlayout)
        self.ui.tunefitlayout.addWidget(fitplotWidget)          






    def dispbeam(self):  
        
        # Add image widget
        self.imv = pg.ImageView()
        
        
         # Configure slices slicer
        self.number_of_slices = self.get_number_of_slices()
        self.ui.slicesSlider.setMaximum(self.number_of_slices - 1)
        self.ui.slicesSlider.valueChanged.connect(self.imageView)

        self.imageView(self.ui.slicesSlider.value())
        
        self.empty_layout(self.ui.beam_layout)        
        self.ui.beam_layout.addWidget(self.imv)

    def get_number_of_slices(self):
        
        filenamebeam = self.ui.lineEdit_stackfile.text()
        self.Img_beamstack = io.imread(filenamebeam)    
        Nslices = np.size(self.Img_beamstack,0)
        
        return Nslices 


    def imageView(self, slice_number):

        imagedata = self.get_image(slice_number)
        self.imv.setImage(imagedata)
        
        

    def get_image(self, slice_number):
        
        filenamebeam = self.ui.lineEdit_stackfile.text()
        self.Img_beamstack = io.imread(filenamebeam)    
        data = self.Img_beamstack[slice_number, :, :]
    
        return data

    @pyqtSlot(np.ndarray)    
    def dispbg(self, Img_bg):  
        
        bgWidget = pg.GraphicsLayoutWidget()
          
    
        # image widget set-up and layout
        vb = bgWidget.addPlot(row=0, col=0)
    
        img = pg.ImageItem(Img_bg)
        vb.clear()
        vb.addItem(img)
        # self.vb.setAspectLocked(True)
            
        #set up histogram for the rendered image
        hist = pg.HistogramLUTItem(image=img)   #set up histogram for the liveview image
        hist.vb.setLimits(yMin=0, yMax=np.max(Img_bg))           
        for tick in hist.gradient.ticks:
            tick.hide()
        bgWidget.addItem(hist, row=0, col=1)
            
        self.empty_layout(self.ui.bgLayout)        
        self.ui.bgLayout.addWidget(bgWidget)
     
    
    @pyqtSlot(np.float, np.float)    
    def dispupdateparam(self, dF, alphaF):
        
           
        text = 'dF=' + str(np.round(dF, decimals=2)) +\
            ' alphaF=' + str(np.round(alphaF, decimals=2)) 
        self.ui.lineEdit_updatecal.setText(text)
        
    
    @pyqtSlot(np.float, np.float, np.ndarray, np.ndarray)    
    def dispN0(self, N0m, sigmaN0, photonsc, histfit):
        
        text = '<N0>=' + str(np.int(N0m)) +\
            ' sigmaN0=' + str(np.int(sigmaN0))
        
        
        self.ui.lineEdit_updateN0.setText(text)
        
        # histogram of N0 and gaussian fit
        histN0Widget = pg.GraphicsLayoutWidget()
        histN0 = histN0Widget.addPlot(title="N0 histogram and fit")
        histN0.setLabels(bottom=('N0 [photons]'), left=('Counts'))
        
        
        hist, bin_edges = np.histogram(photonsc, bins=40)
        N0c = np.arange(bin_edges[0], bin_edges[-1], 100)
     
        fithist = pg.PlotCurveItem(N0c, histfit, pen = self.pen3)
        
        width = np.mean(np.diff(bin_edges))
        bincenters = np.mean(np.vstack([bin_edges[0:-1],bin_edges[1:]]), axis=0)
        bargraph = pg.BarGraphItem(x = bincenters, height = hist, width = width, brush = self.brush1)
        histN0.addItem(bargraph)
        histN0.addItem(fithist)
        
        self.empty_layout(self.ui.layout_N0)
        self.ui.layout_N0.addWidget(histN0Widget)
        
        
    
    @pyqtSlot(np.ndarray, np.ndarray)    
    def dispframef(self, simpler_output, frame):
        
        framef = simpler_output[:,5]
            
        text = '# of localizations from raw data = ' + str(int(len(frame))) + '\n' +\
            ' # of valid localizations (after filter) = ' + str(int(len(framef))) 
        self.ui.textEdit_framef.setText(text)
        
        
        
    def make_connection(self, backend):
        
        backend.sendupdatecalSignal.connect(self.dispupdateparam)
        backend.sendupdateN0Signal.connect(self.dispN0)
        backend.sendSIMPLERSignal.connect(self.scatterplot)
        backend.sendSIMPLERSignal.connect(self.dispframef)
        backend.sendrenderSignal.connect(self.disprender)
        backend.sendtuneSignal.connect(self.dispfinetune)
        backend.sendtunefitSignal.connect(self.fitfinetune)
        backend.sendbgSignal.connect(self.dispbg)

        
                



class Backend(QtCore.QObject):

    paramSignal = pyqtSignal(dict)
    sendupdatecalSignal = pyqtSignal(np.float, np.float)
    sendupdateN0Signal = pyqtSignal(np.float, np.float, np.ndarray, np.ndarray)
    sendSIMPLERSignal = pyqtSignal(np.ndarray, np.ndarray)
    sendrenderSignal = pyqtSignal(np.ndarray)
    sendtuneSignal = pyqtSignal(np.ndarray)
    sendtunefitSignal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    sendbgSignal = pyqtSignal(np.ndarray)
   
        
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
    
        
    @pyqtSlot(dict)
    def get_frontend_param(self, params):
        
        # updates parameters according to what is input in the GUI
        self.fileformat = params['fileformat']
        self.filename = params['filename']
        self.N0fileformat = params['N0fileformat']
        self.N0filename = params['N0filename']
        self.tunefilename = params['tunefilename']
        self.calibfilename = params['calibfilename']
        self.N0calibfilename = params['N0calibfilename']
        self.bgfileformat = params['bgfileformat']
        self.bgfilename = params['bgfilename']
        self.rz_xyz = params['rz_xyz']
        self.illum = params['illumcorr']
        self.lambdaex = params['lambdaex']
        self.lambdaem = params['lambdaem'] 
        self.angle = params['angle']
        self.alpha = params['alpha']
        self.angletune = params['angletune']
        self.alphatune = params['alphatune']  
        self.rangeangle = params['rangeangle']
        self.stepsangle = params['stepsangle']
        self.rangealpha = params['rangealpha']
        self.stepsalpha = params['stepsalpha']
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
        
        return self.dF, self.alphaF
       
    
    def import_file(self,filename, fileformat):
        
        #File Importation
        if fileformat == 0: # Importation procedure for Picasso hdf5 files.
            
            # Read H5 file
            f = h5.File(filename, "r")
            dataset = f['locs']
        
            # Load  input HDF5 file
            frame = dataset['frame']
            photon_raw = dataset['photons']
            bg = dataset['bg']          
            
            xdata = dataset['x'] 
            ydata = dataset['y'] 
            
            # Convert x,y values from 'camera subpixels' to nanometres
            xdata = xdata * self.pxsize
            ydata = ydata * self.pxsize
            
            
        
        elif fileformat == 1: # Importation procedure for ThunderSTORM csv files.
            
            ## Read ThunderSTRORM csv file
            dataset = pd.read_csv(filename)
            # Extraxt headers names
            headers = dataset.columns.values
            
            # data from different columns           
            frame = dataset[headers[0]].values
            xdata = dataset[headers[1]].values 
            ydata = dataset[headers[2]].values
            photon_raw = dataset[headers[3]].values
            bg = dataset[headers[4]].values
            
        else: # Importation procedure for custom csv files.

            # Read custom csv file
            dataset = pd.read_csv(filename)
            # Extraxt headers names
            headers = dataset.columns.values
             
            # data from different columns           
            frame = dataset[headers[0]].values
            xdata = dataset[headers[1]].values 
            ydata = dataset[headers[2]].values
            photon_raw = dataset[headers[3]].values
            bg = dataset[headers[4]].values
            


        return xdata, ydata, frame, photon_raw, bg

    
    def filter_locs(self, x, y, frame, phot_corr, max_dist):
         
        # Build the output array
        listLocalizations = np.column_stack((x, y, frame, phot_corr))
        #REMOVING LOCS WITH NO LOCS IN i-1 AND i+1 FRAME

        # We keep a molecule if there is another one in the previous and next frame,
        # located at a distance < max_dist, where max_dist is introduced by user
        # (20 nm by default).

        min_div = 10
        
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
        
        datacalib = pd.read_csv(calibfilename, header=None)
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
         
        
        filename = self.filename
        fileformat = self.fileformat
        xdata, ydata, frame, photon_raw, bg = self.import_file(filename, fileformat)
        
        x = xdata
        y = ydata 
        
        # Correct photon counts if illum profile not uniform 
        if self.illum == True:
                        
            calibfilename = self.calibfilename
            phot_corr = self.illum_correct(calibfilename, photon_raw, xdata, ydata)
                    
        else:
            phot_corr = photon_raw
        print(0)
        # Filter localizations using max dist
        max_dist = self.maxdist # Value in nanometers
        print(1)
        self.x,self.y,photons,framef = self.filter_locs(x, y, frame, phot_corr, max_dist)
        print(2)
                   
        # SIMPLER z estimation
        z1 = (np.log(self.alphaF*self.N0)-np.log(photons-(1-self.alphaF)*self.N0))/(1/self.dF)
        z = np.real(z1)
        self.z = z.flatten()
        print(3)
        # Compute radial coordinate r from (x,y)   
        P = np.polyfit(self.x,self.y,1)
        print(4)
        def Poly_fun(x):
            y_polyfunc = P[0]*x + P[1]
            return y_polyfunc
        print(5)
        Origin_X = 0.999999*min(self.x)
        Origin_Y = Poly_fun(Origin_X)
        print(6)
        # Change from cartesian to polar coordinates
        tita = np.arctan(P[0])
        tita1 = np.arctan((self.y-Origin_Y)/(self.x-Origin_X))
        print(7)
        r = ((self.x-Origin_X)**2+(self.y-Origin_Y)**2)**(1/2)
        tita2 = [x - tita for x in tita1]
        self.r = np.cos(tita2)*r
        print(4)
                   
        simpler_output = np.column_stack((self.x, self.y, self.r, self.z, photons, framef))
        print(5)
        self.sendSIMPLERSignal.emit(simpler_output, frame)
    
    @pyqtSlot()
    def N0_calibration(self):
        
        filename = self.N0filename
        fileformat = self.N0fileformat


        #File Importation
        xdata, ydata, frame, photon_raw, bg = self.import_file(filename, fileformat) 
        
        # Convert x,y,sd values from 'camera subpixels' to nanometres
        x = xdata * self.pxsize
        y = ydata * self.pxsize

        
        # Correct photon counts if illum profile not uniform 
        if self.illum == True:
                        
            calibfilename = self.N0calibfilename
            phot_corr = self.illum_correct(calibfilename, photon_raw, ydata, xdata)            
     
        else:
            phot_corr = photon_raw
        

        # Filter localizations using max dist

        max_dist = self.maxdist # Value in nanometers
          
        x,y,photons,framef = self.filter_locs(x, y, frame, phot_corr, max_dist)
        
        # For the "N0 Calibration" operation, there is no "Z calculation", 
        # because the aim of this procedure is to obtain N0 from a sample which 
        # is supposed to contain molecules located at z ~ 0.
        c = np.arange(0,np.size(x))
        
        photonsc = photons[c]
        hist, bin_edges = np.histogram(photonsc, bins = 40, density = False)
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
        
        N0c = np.arange(bin_edges[0], bin_edges[-1], 100)
        histfit = gauss(N0c, *coeff)
        
        self.N0m = np.round(coeff[1])
        self.sigmaN0 = np.round(coeff[2])
        
        
        
        self.sendupdateN0Signal.emit(self.N0m, self.sigmaN0, photons, histfit)
        

        
        
    @pyqtSlot()    
    def fine_tune(self):
           
        ## Read csv file
        filename = self.tunefilename
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
        
        [dF_ori, alphaF_ori] = self.getParameters_SIMPLER()
        
        self.angle = self.angletune
        self.alpha = self.alphatune    
        [dF, alphaF] = self.getParameters_SIMPLER()
       
           
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
            photons_matrix[:,i] = self.N0*(alphaF_ori*np.exp(-axial_matrix[:,i]/dF_ori)
            + 1-alphaF_ori)
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
        axial_median = (np.log(alphaF*self.N0)-np.log(photons_median
                        -(1-alphaF)*self.N0))/(1/dF)
        
        # Some elements from the axial vector are zero because the 'data' matrix
        # contains structures with different number of localizations and thus
        # there are columns filled with zeros. Now, we remove those elements. 
            
        c = np.where(axial == 0)
        
        photonsd = np.delete(photons,c) 
        photons_mediand = np.delete(photons_median,c)
        lateral_mediand = np.delete(lateral_median,c)
        laterald = np.delete(lateral,c)
        axial_mediand = np.delete(axial_median,c)
        axiald = np.delete(axial,c)
        axiald = dF*(np.log(alphaF)-np.log(photonsd/self.N0-(1-alphaF)))
        
          
        
        finetune_output = np.column_stack((photonsd, photons_mediand, laterald, lateral_mediand, axiald, axial_mediand))
        self.sendtuneSignal.emit(finetune_output)
     
    def finetunefit(self):
        
        filename = self.tunefilename
        dataset = pd.read_csv(filename, header=None)
        
        [dF_ori, alphaF_ori] = self.getParameters_SIMPLER()
        
        # Lateral positions are obtained from odd columns
        lateral_matrix = dataset.values[:, ::2]
        lateral_matrix[np.where(np.isnan(lateral_matrix))]=0
        # Axial positions are obtained from even columns
        axial_matrix = dataset.values[:, 1::2]
        axial_matrix[np.where(np.isnan(axial_matrix))]=0
            
        # reshape into vectors    
        lateral = lateral_matrix.flatten() 
        axial = axial_matrix.flatten()
        
        angi = self.angletune - self.rangeangle/2
        angf = self.angletune + self.rangeangle/2
        
        
        alphai = self.alphatune - self.rangealpha/2
        alphaf = self.alphatune + self.rangealpha/2

        
        angles = np.round(np.arange(angi, angf, self.stepsangle), decimals = 1)
        alphas = np.round(np.arange(alphai, alphaf, self.stepsalpha), decimals = 1)

        Nang = len(angles)
        Nalp = len(alphas)
        
        print(Nalp)
        D = np.zeros((Nang, Nalp))

        
        for k in np.arange(Nang):
            
            
            for l in np.arange(Nalp):
                # The values used for the axial positions calculation of the known
                # structures through SIMPLER are obtained from the 'run_SIMPLER'
                # interface. It is important to set them correctly before running the
                # operation.
                self.angle = angles[k]
                self.alpha = alphas[l]  
                
                
                [dF, alphaF] = self.getParameters_SIMPLER()
                
                   
                # # The number of photons for each localization are retrieved from the 
                # # axial position and the dF, alphaF and N0 values obtained in the
                # # above step.
                
                photons_matrix = np.zeros(np.shape(axial_matrix))
                photons_median_matrix = np.zeros(np.shape(axial_matrix))
                lateral_median_matrix = np.zeros(np.shape(axial_matrix))
                
                # # The next function allows to obtain a custom 'median' value, which is
                # # calculated as the mean value between the p-centile 10% and p-centile
                # # 90% from a given distribution. We use this function in order to
                # # re-center the localizations from the known structures around [0,0]
                
                median_perc90_10_center = lambda x: np.mean([np.percentile(x,90), np.percentile(x,10)])
                
                # # To calculate the 'median' values, we use valid localizations, 
                # # i.e. those with axial positions different from 0;
                # # there are elements filled with z = 0 and lateral = 0 in the 'data' matrix,
                # # because not every known structures have the same number of localizations
                
                for i in np.arange(np.shape(lateral_matrix)[1]):
                    c = np.where(axial_matrix[:,i]!=0) 
                    photons_matrix[:,i] = self.N0*(alphaF_ori*np.exp(-axial_matrix[:,i]/dF_ori)
                    + 1-alphaF_ori)
                    photons_median_matrix[:,i] = (np.ones((np.shape(photons_matrix)[0]))
                                                *median_perc90_10_center(photons_matrix[c,i]))
                    lateral_median_matrix[:,i] = (np.ones((np.shape(lateral_matrix)[0]))
                                                *median_perc90_10_center(lateral_matrix[c,i]))
                
                # # Number of photons for each lozalization    
                photons = photons_matrix.flatten()
                # # Median value of the number of photons for the structure to which
                # # each localization belongs
                photons_median = photons_median_matrix.flatten()
                # # Lateral positions median value for the structure to which
                # # each localization belongs                                                
                lateral_median = lateral_median_matrix.flatten()
                # # Median value of the axial position for the structure to which 
                # # each localization belongs                                            
                axial_median = (np.log(alphaF*self.N0)-np.log(photons_median
                                 -(1-alphaF)*self.N0))/(1/dF)
                
                # # Some elements from the axial vector are zero because the 'data' matrix
                # # contains structures with different number of localizations and thus
                # # there are columns filled with zeros. Now, we remove those elements. 
                    
                c = np.where(axial == 0)
       
                photonsd = np.delete(photons,c) 
                lateral_mediand = np.delete(lateral_median,c)
                laterald = np.delete(lateral,c)
                axial_mediand = np.delete(axial_median,c)
                axiald = np.delete(axial,c)
                axiald = dF*(np.log(alphaF)-np.log(photonsd/self.N0-(1-alphaF)))
        
                lateralc = laterald - lateral_mediand
                axialc = axiald - axial_mediand
                    
                xy = np.vstack((lateralc, axialc)).T
                circle = CircleFit(xy)
                xc = circle[0]
                yc = circle[1]
                Rc = circle[2]
                dc = 2*Rc    
                D[k,l] = dc
        
        
        self.sendtunefitSignal.emit(angles, alphas, D)
              
        
        
       
    @pyqtSlot()    
    def obtain_bg(self):
        
        ## Read file
        filename = self.bgfilename
        fileformat = self.bgfileformat
        
        xdata, ydata, frame, photon_raw, bg = self.import_file(filename, fileformat)
        
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
   
        self.sendbgSignal.emit(Img_bg)

   
    
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
            
        elif self.rz_xyz == 2:
            
            lat = self.x
            ax = self.y
            B = self.gauss_render(lat,ax)
            

        
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
        frontend.finetuneSignal.connect(self.fine_tune)
        frontend.finetunefitSignal.connect(self.finetunefit)
        frontend.N0calSignal.connect(self.N0_calibration)
        frontend.obtainbgSignal.connect(self.obtain_bg)
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
        
    app.exec_()     
