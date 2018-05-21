#SpecTracer is a program desinged to reduce echelle spectra. 
#It is able to trace ordersusing manual input from the user.
#It uses bias and overscan to eliminate sources of noise inherent to the detector
#It uses dark files to remove stray light from the detector.
#The program also accounts for sensitivity variations from pixel to pixel
#The program is able to relate wavelength and pixel coordianted in order
#To identify wavelength of observed spectral features

#The code was designed by: Oscar Fernando Romero Matamala
#For: Dr. Veronique Petit at Florida Institute of Technology

#Version 2.0.0

#New Functionality:
#orderTracer() method modified
#A Semi-Automatic trace mechanism has been introduced in this version.
#It requires minimal user input and allows for a more precise and
#faster tracing of the orders.
#There were significant variations in the order positions from night to
#night which encouraged the introduction of this routine.




#Importing all of the needed libraries
import sys
sys.ps1 = 'SOMETHING'
from astropy.io import fits 
from numpy import *
import numpy as np
import numpy.ma as ma
import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import Tkinter as tk 
from Tkinter import *
import tkMessageBox
import tkSimpleDialog
from tkFileDialog import askdirectory
from tkFileDialog import askopenfilename
from tkFileDialog import asksaveasfilename
from tkFileDialog import askopenfilenames
import tkFont
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
import signal
import threading
import multiprocessing
from multiprocessing import *
from multiprocessing.sharedctypes import Value, Array
import os
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize
import scipy.linalg
import scipy.ndimage as ndimage
from scipy.signal import argrelextrema
from scipy.stats import chisquare 
import pylab



#Identifies the platform in which the code is running
def getOS ():
	global divider
	if (sys.platform == 'win32'):
		divider = '\\'
	else:
		divider = '/'

#File Operations

def getFileName (instruction):
	global file_Name, entry, fileBox, isFiles, name


	if (not isFiles):
		file_Name = ''
		isFiles = True

	#Tkinter object initialize
	fileBox = Tk()

	fileBox.title ("SpecTracer")
	helv20 = tkFont.Font(family ='Helvetica', size = 10, weight = 'bold')
	
	#Window Components initialization
	primary_label = Label(fileBox, text = instruction, font = helv20)
	browse_button = Button(fileBox, text = "Browse", command = browseFile, font = helv20)
	ok_button = Button(fileBox, text="OK", command = closeFile, font = helv20)
	entry = Entry(fileBox, width = 50)

	#Component Location
	primary_label.grid (row = 0, column = 0,columnspan = 10, sticky = N)
	browse_button.grid (row = 1, column =1, sticky = E, padx = 5, pady = 5)
	ok_button.grid (row = 1, column =2, sticky = E)
	entry.grid(row = 1, column = 0, pady = 5, sticky = S)

	fileBox.mainloop()
	
	return name

def readSpectrumFits(instruction):
	spectrum_FileName=getFileName(instruction)
	hdulist = fits.open(spectrum_FileName)
	length = hdulist[1].header['naxis2'] #Reads FITS file, and searches for term 'naxis2'
	columns = hdulist[1].header['tfields'] #Reads FITS file and searches for 'tfields'
	spectrum = np.ndarray(shape=(columns, length))
	tbdata = hdulist[1].data 

	i = 0
	while (i< columns):
		spectrum[i] = tbdata.field(i)
		i = i+1

	hdulist.close()

	return spectrum	

def saveToFile (name, toWrite):
	path = primary_dir + divider + name
	extension = ''
	if (name == 'flatfield'):
		toWrite.dump(path)
	elif (name == 'bias'):
		toWrite.dump(path)
	elif (name == 'dark'):
		toWrite.dump(path)
	else:
		extension = '.npy'
		np.save(path, toWrite)

	message = 'File: ' + name + extension+ ' Created successfully!'
	dialog(message, 'i')

def saveToSecondaryFile (name, toWrite):
	path = secondary_dir + divider + name
	np.save(path, toWrite)

	message = 'File: ' + name + '.npy Created successfully!'
	dialog (message, 'i')

def saveToNormalizeFile (name, toWrite):
	path = normalize_dir + divider + name
	np.save(path, toWrite)

	message = 'File: ' + name + '.npy Created successfully!'
	dialog (message, 'i')


#Interactive Browse Modules

def browseDir ():
	global instrument

	Tk().withdraw()
	instrument = askdirectory()

	entry.delete(0, END)
	entry.insert(0, instrument)

def browseFile ():
	global file_Name

	Tk().withdraw()
	file_Name = askopenfilename()
	entry.delete(0, END)
	entry.insert(0, file_Name)


#Closing Tkinter instances modules
#Due to the lack of an OOP, closing a particular Tkinter window
#Causes the interpreter to be unrecoverable and the window can't be recalled
#To avoid this a window for each plot was created, each with its closing method
#All of this methods just quit and destroy the window, and get necessary input
#information

#This methods extract the path to the directory/file after quitting and destroying
def closeDir ():
	if (instrument==''):
		message = 'Please select a folder!'
		dialog(message, 'i')
	else:
		folderDialog.quit()
		folderDialog.destroy()

		getDir (instrument)

def closeBiasDir ():
	if (instrument==''):
		message = 'Please select a folder!'
		dialog(message, 'i')
	else:
		bias.quit()
		bias.destroy()

		getDir (instrument)

def closeDarkDir ():
	if (instrument==''):
		message = 'Please select a folder!'
		dialog(message, 'i')
	else:
		dark.quit()
		dark.destroy()

		getDir (instrument)

def closeSecondaryDir ():
	if (instrument==''):
		message = 'Please select a folder!'
		dialog(message, 'i')
	else:
		second.quit()
		second.destroy()

		getDir (instrument)

def closeFile ():
	if (file_Name=='' or not(((file_Name[-3::] == 'fit') or (file_Name[-3::]=='txt')) or (file_Name[-4::]=='fits'))):
		message = 'Please select a valid file!'
		dialog(message, 'i')
	else:
		fileBox.quit()
		fileBox.destroy()

	getName (file_Name)
	isFiles = False

def closeMultiFile ():
	if (file_Name=='' or not((file_Name[-3::] == 'fit') or (file_Name[-4::]=='fits'))):
		message = 'Please select a valid file!'
		dialog(message, 'i')
	else:
		fileBox.quit()
		fileBox.destroy()

		getListName (file_List)
		isFiles = False


#This modules just quit and destroy the window.
def close():
	root.quit()
	root.destroy()

def closeOrderCorrectWin ():
	orderCorrectWin.quit()
	orderCorrectWin.destroy()

def closeOverScanWin ():
	overScanWin.quit()
	overScanWin.destroy()

def closeSpectrumWin ():
	spectrumWin.quit()
	spectrumWin.destroy()

def closeFlatWin ():
	flatWin.quit()
	flatWin.destroy()

def closeReducedSpectWin ():
	reducedSpectWin.quit()
	reducedSpectWin.destroy()

def closeGaussWin ():
	gaussianWin.quit()
	gaussianWin.destroy()

def closeCorrectWin ():
	correctWin.quit()
	correctWin.destroy()

def closeAppendWin ():
	appendWin.quit()
	appendWin.destroy()

def closeOverScanFitWin ():
	overScanFitWin.quit()
	overScanFitWin.destroy()

def closeSensitivityWin ():
	sensitivityWin.quit()
	sensitivityWin.destroy()

def closeFlatFieldWin():
	flatFieldWin.quit()
	flatFieldWin.destroy()

def closeMaster ():
	master.quit()
	master.destroy()
	sys.exit()

def closeOrderCalibWin():
	orderCalibrationWin.quit()
	orderCalibrationWin.destroy()
		
def closeTraceWind ():
	choice = tkMessageBox.askyesno("SpecTracer", "Are you sure you want to quit?")
	if (choice):
		orderTracerWindow.quit()
		orderTracerWindow.destroy()
 
def doneTraceWind ():
 	orderTracerWindow.quit()
	orderTracerWindow.destroy()

def closeCalibWind ():
	choice = tkMessageBox.askyesno("SpecTracer", "Are you sure you want to quit?")
	if (choice):
		calibrationWindow.quit()
		calibrationWindow.destroy()
		fig13.canvas.mpl_disconnect(cid)
		fig14.canvas.mpl_disconnect(rid)

def closeNormalizeWin ():
	choice = tkMessageBox.askyesno("SpecTracer", "Are you sure you want to quit?")
	if (choice):
		normalizeSpectWin.quit()
		normalizeSpectWin.destroy()

def done ():
	orderCalibrationWin.quit()
	orderCalibrationWin.destroy()

def doneAppend ():
	canvas.mpl_disconnect(cid)
	appendWin.quit()
	appendWin.destroy()


#This two modules extract the input from the window to obtain the path
#to the file or directory needed

def getName (file_Name):
	global name
	name = file_Name

def getListName (file_Name):
	global name
	name = list(file_List)

def getDir (instrument):
	global primary_dir, secondary_dir, bias_dir, dark_dir, secondary_dir, normalize_dir
	if (is_Primary):
		primary_dir = instrument
	elif (isBias):
		bias_dir = instrument
	elif (isDark):
		dark_dir = instrument
	elif (isSec):
		secondary_dir = instrument
	elif (isNrm):
		normalize_dir = instrument
	isFiles = False


#This module obtains the primary directory where all the core files
#will be stored.

def folderCreation (prim, bia, dar, sec, nrm, message):
	global instrument, entry, folderDialog, isFiles
	global is_Primary, isBias, isDark, isSec, isNrm
	is_Primary = prim
	isBias = bia
	isDark = dar
	isSec=sec
	isNrm=nrm
	if (not isFiles):
		instrument = ''
		isFiles = True

	folderDialog = Tk()

	folderDialog.title ("SpecTracer")
	helv20 = tkFont.Font(family ='Helvetica', size = 10, weight = 'bold')
	
	primary_label = Label(folderDialog, text = message, font = helv20)
	browse_button = Button(folderDialog, text = "Browse", command = browseDir, font = helv20)
	ok_button = Button(folderDialog, text="OK", command = closeDir, font = helv20)
	entry = Entry(folderDialog, width = 50)

	primary_label.grid (row = 0, column = 0, sticky = N)
	browse_button.grid (row = 1, column =1, sticky = E, padx = 5, pady = 5)

	ok_button.grid (row = 1, column =2, sticky = E)
	entry.grid(row = 1, column = 0, pady = 5, sticky = S)

	folderDialog.mainloop()

#This Module opens .npy type files
	
def openNumpyFile (fileName):
	path = primary_dir+divider+fileName
	return np.load(path)

def openNormalizationFile (fileName):
	path = normalize_dir+divider+fileName
	return np.load(path)

#Saves Reduced Spectrum into FITS file

def saveSpectrum (spectrum):
	#Obtain user determined File Name
	star_File = asksaveasfilename( defaultextension = ".fits")

	i = 0

	#Initialize file parts
	label = 'Order_' + str(i)

	#Creation of Table for FITS file
	col  = fits.Column(name = label, format = 'D' ,array = spectrum[i])
	column  = [col]
	hdu = fits.PrimaryHDU(spectrum)
	i=i+1
	
	#Stores each order in a column of the FITS file
	while (i < len(spectrum)):
		label = 'Order_' + str(i)
		column.extend([fits.Column(name = label, format = 'D' ,array = spectrum[i])])
		i = i+1

	cols = fits.ColDefs(column)

	tbhdu = fits.BinTableHDU.from_columns(cols)
	thdulist = fits.HDUList([hdu, tbhdu])

	#writes to file
	thdulist.writeto(star_File)
	message = 'File at: ' + star_File + ' created succesfully!'
	dialog(message, 'i')

#Reads master.txt
def calibrationFileRead (fileName):
	calibrationMaster = np.loadtxt(fileName, delimiter = ',')
	return calibrationMaster

#Gets the correlated pixel-coordinated and wavelength files.
def getScale (order):
	x_Range = np.linspace(0, g_length, g_length)
	try:
		z = np.load(secondary_dir + divider + str(order)+'.npy')
	except (IOError):
		scale = x_Range
		isCalibrated = False
	else:
		f = np.poly1d(z)
		scale = f(x_Range)
		isCalibrated = True

	return scale, isCalibrated

#Image Operations

#Gets size of image
def getDimensions (name):
	hdu = fits.getheader(name)
	length = hdu['naxis1']
	width = hdu ['naxis2']

	if (length > width):
		flag = True
	else:
		flag = False
		temp = width
		width = length
		length = temp

	return flag, length, width


#Ensures the image is landscape oriented
def forceLandscape(data, isLandscape):
	if (not isLandscape):
		data = zip(*data[::-1])
	return data	

#Fits Gaussian to the data points of flat lamp for weighted sum
def gaussian (x,a,x0,sigma):

	return a*np.exp(-(x-x0)**2/(2*sigma**2))

#Updated plot by replotting fit lines after modifications are done
def fitGrapher (x_new,axis, figure):
	global g_data
	i = 0

	#For each order
	while (i< len(g_orderLoc)):
		#Trace order (usual 2nd order)
		t=f[i]
		t.c[0]=g_multiplier[i]*f[i][2]
		t.c[1]=f[i][1]-g_drift[i]
		y_new = np.copy(t(x_new))+np.copy(g_orderLoc[i])+np.copy(v_pan[i])
		plt.gca().plot(x_new, y_new, 'r-')
		message = 'Order: ' + str(i)
		plt.gca().text(5, (g_orderLoc[i]), message, style = 'italic', 
					bbox ={'facecolor':'white', 'alpha':0.5, 'pad':3})
		f[i][2] = f[i][2]/g_multiplier[i]
		f[i][1]=f[i][1]+g_drift[i]
		i = i+1
	plt.gca().imshow(g_data, origin = 'lower')
	plt.gcf().canvas.draw()

#Stores the coordinates of the points being scattered
def scatterArray (order):
	i = 0
	x_list = []
	y_list = []
	l_list = []
	while (i<len(calibrationMaster)):
		if (order == int(calibrationMaster[i,0])):
			x_list.append((calibrationMaster[i,1]))
			y_list.append((spectrum_In[int(order), int(calibrationMaster[i,1])]))
			l_list.append((calibrationMaster[i,2]))
		i = i+1
	x_Position = np.array(x_list)
	y_Position = np.array(y_list)
	labels_loc = np.array(l_list)

	return x_Position, y_Position, labels_loc

#Update Tkinter plots after modifications are done (i.e. changing order)
def updatePlotShow (order):
	global fig11
	title = "Order: " + str(order)
	scale, isCalibrated = getScale(order)

	#Labels for x-axis. In case there is no information for the order it becomes pixel
	if isCalibrated:
		label = 'Wavelength'
	else:
		label = 'Pixel'

	#Clear axes for replotting
	ax11.cla()
	ax11.plot (scale[overScan_location[0]:overScan_location[1]], spectrum_In[order, overScan_location[0]:overScan_location[1]],'-k')
	ax11.set_title(title, fontsize=16)
	ax11.set_xlabel(label, fontsize=14)
	ax11.set_ylabel("Flux", fontsize = 14)
	ax11.tick_params(axis='both', which = 'major', labelsize=12)
	fig11.tight_layout()

	fig11.canvas.draw()

def updatePlotGauss (order):
	global fig8, flat_gauss, weightArray
	title = "Gaussian Fit. Order: " + str(order)
	plt.gca()
	plt.cla()
	
	ax8.plot(gauss_step, weightArray[order], 'k--', label= 'Fit')
	ax8.plot(gauss_step,flat_gauss[order].flatten(), 'k-', label = 'Data')
	legend = ax8.legend(loc = 'lower center', shadow = False)

	plt.title(title, fontsize=16)
	plt.xlabel("Thick", fontsize=14)
	plt.ylabel("Flux", fontsize = 14)
	
	fig8.canvas.draw()

def updatePlotSens (order):
	global fig5, ax5
	title = "Order: " + str(order)
	plt.gcf()
	plt.clf()

	#Plots a 3D surface
	ax5 = fig5.gca(projection = '3d')
	ax5.scatter3D(xPoints2[::20],yPoints2[::20],
						flat_flux[order].flatten()[::20], c = 'r',s = 50)
	ax5.plot_surface (X2, Y2, surface_val2[order], rstride = 50, cstride = 50, alpha = 0.5)
	ax5.set_xlabel('Pixel')
	ax5.set_ylabel('Pixel')
	ax5.set_ylim(np.min(original_Y[order]), np.max(original_Y[order]))
	ax5.set_zlabel('Flux')
	ax5.set_title('Flat Field' + title)
	
	ax5.mouse_init(rotate_btn=1, zoom_btn=3)
	ax5.view_init(elev=20, azim=45)
	plt.gcf().canvas.draw()

def updatePlot (order):
	global fig13, pointsUp, isUpSelected, crossRelatedPoints, labels, cid, isLine
	crossRelatedPoints =np.array([])
	isUpSelected = False
	
	high = argrelextrema(spectrum_In[order], np.greater, order = 10)
	title = "Order: " + str(order)
	ax15.cla()
	ax13.cla()
	ax13.plot (spectrum_In[order], '-k')
	x_Position, y_Position, labels = scatterArray(order)
	amount = len(x_Position)
	pointsUp = ax13.scatter(x_Position,y_Position, color = ["blue"]*amount,s = [75]*amount, alpha = 0.5, picker = 5)
	i = 0
	while (i<len(x_Position)):
		ax13.annotate(labels[i], xy = (x_Position[i], y_Position[i]),
					xytext = (10,10),
					textcoords = 'offset points', ha = 'right', va = 'bottom',
					bbox = dict(boxstyle = 'round, pad = 0.2', fc = 'white', alpha = 0.2))
		i = i+1
	plt.gcf()
	ax13.set_title(title, fontsize=16)
	ax13.set_xlabel("Pixel", fontsize=14)
	ax13.set_ylabel("Flux", fontsize = 14)
	ax13.tick_params(axis='both', which = 'major', labelsize=12)
	
	cid = fig13.canvas.mpl_connect('pick_event', clickScatterUp)
	fig13.tight_layout()
	

	fig13.canvas.draw()

def updateLowPlot (order):
	global fig14, pointsLow, x_Coord, rid,isLine
	isLine = False
	high = argrelextrema(spectrum_II[order], np.greater, order = 10)
	title = "Order: " + str(order)
	ax15.cla()
	ax14.cla()
	ax14.plot (spectrum_II[order],'-k')

	amount = len(high[0])
	x_Position= high[0]
	x_Coord = np.array(x_Position)
	y_Position =spectrum_II[order, high[0]]
	pointsLow = ax14.scatter(x_Coord,y_Position, color=["blue"]*amount, 
			s = [75]*amount, alpha = 0.5, picker = 5)

	ax14.set_title(title, fontsize=16)
	ax14.set_xlabel("Pixel", fontsize=14)
	ax14.set_ylabel("Flux", fontsize = 14)
	ax14.tick_params(axis='both', which = 'major', labelsize=12)
	ax15.set_title('Correlation '+title)
	ax15.set_xlabel('Pixel')
	ax15.set_ylabel('Wavelength')
	rid = fig14.canvas.mpl_connect('pick_event', clickScatterLow)
	fig14.tight_layout()
	fig15.tight_layout()
	fig15.canvas.draw()
	fig14.canvas.draw()

def firstPassNormalization (order):
	global firstPass, normalization_function, spectrumNormalized


	maxPoints = argrelextrema(spectrum[order,0,overScan_location[0]:overScan_location[1]], np.greater, order = 100)
	x_Coord=maxPoints[0]
	y_Coord=np.copy(spectrum[order,0, x_Coord])
	normalization = np.polyfit (x_Coord, y_Coord, 2)
	normalization_function = np.poly1d(normalization)

	spectrumNormalized[order]=np.copy(spectrumNormalized[order])/normalization_function(x_Dimension)

	return normalization

def updateNormalizePlot (order):
	global figNormalizedSpectrum, figFittedSpectrum, axFittedSpectrum, axNormalizedSpectrum, isNormalized, normalization_function
	global x_Coord_Masked,y_Coord_Up_Masked, y_Coord_Low_Masked, order_g
	global fittedLine, normalizedLine, spectrumOrig, coloredSegmentX, coloredSegmentY
	global maskedSegmentX, maskedSegmentY, jump_Array, normalizationOrder_Array, spectrumNormalized, comparisonLine

	order_g=order
	axFittedSpectrum.cla()
	axNormalizedSpectrum.cla()

	spectrumOrig = axFittedSpectrum.plot(spectrum[order_g,0, overScan_location[0]:overScan_location[1]], 'k-')

	if(not isNormalized[order] and not (isCleared[order_g])):
		normalization_Array[order_g]=np.array(firstPassNormalization(order_g))
		isNormalized[order_g] = True
	else:
		normalization_function=np.poly1d(normalization_Array[order])
		spectrumNormalized[order]=np.copy(spectrum[order,0])/normalization_function(x_Dimension)

	if (not normalizedFileExists) or (isCleared[order_g]):
		maxPoints = argrelextrema(spectrumNormalized[order_g, overScan_location[0]:overScan_location[1]], np.greater, order = 1)
		amount = len(maxPoints[0])
		x_Coord=np.array(maxPoints[0])
		y_Coord_Up=np.copy(spectrum[order_g, 0, maxPoints[0]])
		x_Coord_Masked = ma.array(x_Coord)
		y_Coord_Up_Masked =ma.array(y_Coord_Up)

	comparisonLine= np.ones(len(spectrum[order_g,0]))

	title = "Order: " + str(order_g) + " (Non-Normalized)"
	title_Low = "Order: " + str(order_g) + " (Normalized)"
	axFittedSpectrum.set_title(title, fontsize=16)
	axFittedSpectrum.set_xlabel("Pixel", fontsize=14)
	axFittedSpectrum.set_ylabel("Flux", fontsize=14)
	axFittedSpectrum.tick_params(axis='both', which= 'major', labelsize=12)
	axNormalizedSpectrum.set_title(title_Low, fontsize=16)
	axNormalizedSpectrum.set_xlabel("Pixel", fontsize=14)
	axNormalizedSpectrum.set_ylabel("Flux", fontsize=14)

	fittedLine=axFittedSpectrum.plot(normalization_function(x_Dimension), 'b-')
	normalizedLine=axNormalizedSpectrum.plot(spectrumNormalized[order_g, overScan_location[0]:overScan_location[1]], 'k-')
	if (not coloredSegmentX[order]==[]):
		i=0
		while(i<len(coloredSegmentX[order])):
			axFittedSpectrum.plot(np.array(coloredSegmentX[order][i]),
				np.array(coloredSegmentY[order][i]),'y-')
			i = i+1

	axNormalizedSpectrum.plot(comparisonLine, 'r--')

	figFittedSpectrum.tight_layout()
	figNormalizedSpectrum.tight_layout()
	figFittedSpectrum.canvas.draw()
	figNormalizedSpectrum.canvas.draw()

	toggleSelector.RS=RectangleSelector(axNormalizedSpectrum, selectNormalized, 
		drawtype = 'box', useblit=True, button=[1, 3], minspanx=1, minspany=1, 
		spancoords='pixels')

def updateOrderPlot ():
	global maximum_left, maximum_right, maximum_vertex, fig19, fig20, ax, ax20, columns
	global points_right, points_left, points_vertex, lowerVal

	points_left = ax[0].scatter(maximum_left[::], columns[0,maximum_left[::]],
									color=['blue']*len(maximum_left[::]),
									s =[20]*len(maximum_left[::]))
	ax[0].plot(columns[0])

	points_vertex = ax[1].scatter(maximum_vertex, columns[1,maximum_vertex],
									color=['blue']*len(maximum_vertex),
									s =[20]*len(maximum_vertex))
	ax[1].plot(columns[1])

	points_right = ax[2].scatter(maximum_right[::], columns[2,maximum_right[::]],
									color=['blue']*len(maximum_right[::]),
									s =[20]*len(maximum_right[::]))
	ax[2].plot(columns[2])

	for label,x,y in zip(labels_left, maximum_left[::],columns[0,maximum_left]):
		ax[0].annotate(label, xy=(x,y), xytext=(0,10),style = 'italic',
			 textcoords='offset points',
			bbox={'facecolor':'white', 'alpha':0.5, 'pad':3})

	for label,x,y in zip(labels_vertex , maximum_vertex, columns[1,maximum_vertex]):
		ax[1].annotate(label, xy=(x,y), xytext=(0,10),style = 'italic',
			 textcoords='offset points',
			bbox={'facecolor':'white', 'alpha':0.5, 'pad':3})

	for label,x,y in zip(labels_right, maximum_right[::], columns[2,maximum_right]):
		ax[2].annotate(label, xy=(x,y), xytext=(0,10),style = 'italic',
			 textcoords='offset points',
			bbox={'facecolor':'white', 'alpha':0.5, 'pad':3})

	for label,x,y in zip(labels_left, 
		np.repeat(overScanLeft, len(amount_left)), maximum_left):
		ax20.annotate(label, xy=(x,y), xytext=(0,0),style = 'italic',
			 textcoords='offset points',
			bbox={'facecolor':'white', 'alpha':0.5, 'pad':3})

	for label,x,y in zip(labels_vertex, 
		np.repeat(vertex_location, len(maximum_vertex)), 
		maximum_vertex):
		ax20.annotate(label, xy=(x,y), xytext=(0,0),style = 'italic',
			textcoords='offset points',
			bbox={'facecolor':'white', 'alpha':0.5, 'pad':3})

	for label,x,y in zip(labels_right, 
		np.repeat(overScanRight, len(maximum_right)), maximum_right):
		ax20.annotate(label, xy=(x,y), xytext=(0,0),style = 'italic',
			 textcoords='offset points',
			bbox={'facecolor':'white', 'alpha':0.5, 'pad':3})

	ax[0].set_title ('Left')
	ax[1].set_title('Center')
	ax[2].set_title('Right')
	
	fig19.tight_layout()
	fig19.canvas.draw()
	fig20.canvas.draw()

	toggleSelector.RS=RectangleSelector(ax[1], selectPeaks, 
		drawtype = 'box', useblit=True, button=[1, 3], minspanx=1, minspany=1, 
		spancoords='pixels')


#ADjusts the median of the gaussian curve as this can shift by ~1 px
def gaussAdjust(weights, point):
	i = (np.abs(weights-point)).argmin()
	return weights[i]

#Event Driven
def onclick (event): 
	global rid, fitX,fitY
	#Checks the type of event
	if (event.dblclick):
		#checks click was inside plot
		if (event.xdata != None) and (event.ydata != None):
			xCoord,yCoord = event.xdata, event.ydata
			message =  "x = " + str(xCoord) + " y = " + str(yCoord)	
			dialog(message, 'i')
			#Adds the selected pixel coordinate to the array
			fitX= np.append(fitX, xCoord)
			fitY = np.append(fitY, yCoord)
			orderCalibInstructions()

			#In case the tracing is done, proceed to order selection
			if len (fitX) == 3:
				
				orderCanvas.mpl_disconnect(cid)
				rid = orderCanvas.mpl_connect('button_press_event', orderPositions)
				
		else:
			message = 'Please click inside the plot!'
			dialog(message, 'i')
	
#Obtains the location of the rest of the orders
def orderPositions (event):
	global g_proper_orderLoc
	if event.dblclick:
		if (event.button == 1):
			yCoord = event.ydata
			message = " y = " + str(yCoord)	
			dialog(message, 'i')
			g_proper_orderLoc=np.append(g_proper_orderLoc, yCoord)
		else:
			orderCanvas.mpl_disconnect(rid)
			orderCalibrationWin.quit()
			orderCalibrationWin.destroy()

def orderLocator (event):
	global z
	if event.dblclick:
		if (event.button == 1):
			yCoord = event.ydata
			message = "y = " + str(yCoord)
			dialog(message, 'i')
			g_orderLoc.append(yCoord)
			z=np.insert(z,(len(z[0])-1),z[:,(len(z[0])-1)],1)
		else:
			fig16.canvas.mpl_disconnect(cid)
			plt.close()


def selectPeaks (eclick,erelease):
	global x_Initial, x_End
	x_Initial = int(eclick.xdata)
	x_End = int(erelease.xdata)
	if(x_Initial<0):
		x_Initial=0
	if(x_End>=len(columns[0])):
		x_End=len(columns[0]-1)
	updateSegmentPeaks(x_Initial, x_End)

def toggleSelector():
	return

def updateSegmentPeaks(start,end):
	global location_vertex, location_left, location_right, points_left, points_vertex, points_right
	location_vertex = peakLocations(start,end)
	location_left = np.copy(location_vertex)
	location_right = np.copy(location_vertex)
	points_left._facecolors[location_left,:] = (0,1,0,1)
	points_vertex._facecolors[location_vertex,:] = (0,1,0,1)
	points_right._facecolors[location_right,:] = (0,1,0,1)
	fig19.canvas.draw()

def autoFitter ():
	global f, z, location_vertex,g_data,overScanRight,overScanLeft,overScanRight,thick

	i=overScanLeft+1
	while (i<overScanRight):
		maximum=argrelextrema(g_data[int(lowerVal[i-overScanLeft-1]-thick/2)::,i],np.greater_equal,order=thick/2)[0]+int(lowerVal[i-overScanLeft-1]-thick/2)
		j=0
		toDelete=np.array([])
		while (j<len(maximum)-1):
			if (maximum[j]==(maximum[j+1]-1)):
				toDelete= np.append(toDelete, [j])
			j=j+1

		maximum = np.delete(maximum, toDelete)

		if (i==overScanLeft+1):
			y_values_order=np.array([maximum[location_vertex]])
		else:
			y_values_order=np.append(y_values_order,[maximum[location_vertex]],axis=0)

		i=i+1
	x_values_order = np.arange(overScanLeft+1,overScanRight,1)
	z=np.polyfit(x_values_order, y_values_order, 2)
	
	doneTraceWind()

def fitReader (z):
	f =[]
	i=0
	while (i < len(z[0])):
		f.append(np.poly1d(z[:,i]))
		i=i+1

	return f

def peakLocations (start, end):
	condition = (maximum_vertex>start)&(maximum_vertex<end)
	locations = np.where(condition)
	return locations


def checkVertex(max_row, length):
	i = 0
	vertex = length/2
	vertex_max=0
	difference = length
	for i in range(len(max_row)):
		if (abs(max_row[i]-vertex)<difference):
			vertex_max = i
			difference = abs(max_row[i]-vertex)
	return vertex_max

def maximumDiscriminator (maxima, length,overscan_left,overscan_right):
	corrector_left = 0
	corrector_right = 0
	vertex = len(g_data[0])/2
	lowerVal=np.zeros(overscan_right-overscan_left)
	lowerVal[vertex-overscan_left]=maxima[0]
	isFoundLeft = False
	isFoundRight = False
	i=vertex
	print overscan_right
	while (i>overscan_left):
		if(not isFoundLeft):
			maximum_vertex = argrelextrema(g_data[::,i], np.greater_equal, order=thick/2)[0]
			if abs(maximum_vertex[0]-lowerVal[i-overscan_left])<=thick/2:
				lowerVal[i-overscan_left-1]=maximum_vertex[0]
			else:
				isFoundLeft=True
		if(isFoundLeft):
			maximum_vertex = argrelextrema(g_data[int(lowerVal[i-overscan_left]-thick/2):,i], np.greater_equal, order=thick/2)[0]
			lowerVal[i-overscan_left-1]=maximum_vertex[0]+lowerVal[i-overscan_left]-thick/2
		print lowerVal[i-overscan_left-1]
		i=i-1

	i=vertex
	
	while (i<overscan_right):
		if(not isFoundRight):
			maximum_vertex = argrelextrema(g_data[::,i], np.greater_equal, order=thick/2)[0]
			if abs(maximum_vertex[0]-lowerVal[i-overscan_left-1])<=thick/2:
				lowerVal[i-overscan_left]=maximum_vertex[0]
			else:
				isFoundRight=True

		if(isFoundRight):
			maximum_vertex = argrelextrema(g_data[int(lowerVal[i-overscan_left-1]-thick/2):,i], np.greater_equal, order=thick/2)[0]
			lowerVal[i-overscan_left]=maximum_vertex[0]+lowerVal[i-overscan_left-1]-thick/2

		print lowerVal[i-overscan_left]
		i=i+1

	print lowerVal
	return lowerVal

#Wavelength Calibration submodules

#Obtains the information from the clicked point on upper plot
#Changes color to green
def clickScatterUp (event):
	global isUpSelected, upperInd
	upperInd = event.ind
	pointsUp._facecolors[upperInd,:] = (0,1,0,0.5)
	fig13.canvas.draw()
	isUpSelected = True

#Obtains the information from the clicked point on lower plot
#Changes color to green
def clickScatterLow (event):
	global isUpSelected
	if isUpSelected:
		lowerInd = event.ind
		pointsLow._facecolors[lowerInd,:] = (0,1,0,0.5)
		fig14.canvas.draw()
		isUpSelected = False
		calibrationFunction (lowerInd)
	else:
		message =  'Please click the upper plot first'
		dialog(message,'i')

def selectNormalized (eclick,erelease):
	global x_Initial, x_End
	x_Initial = int(eclick.xdata)
	x_End = int(erelease.xdata)
	if(x_Initial<0):
		x_Initial=0
	if(x_End>=len(spectrum[0,0])):
		x_End=len(spectrum[0,0]-1)
	removeSegmentNormalization(x_Initial, x_End)

def toggleSelector (event):
	return
	
#Resets plot and stores related points into array, Scatters points being related
#Changes color to green.
def calibrationFunction (lowerInd):
	global fig15, x_Position_Low, crossRelatedPoints, isLine
	
	plt.figure(3)

	crossRelatedPoints =  np.append(crossRelatedPoints,[x_Coord[lowerInd],labels[upperInd]])
	crossRelatedPoints = np.reshape(crossRelatedPoints, ((len(crossRelatedPoints)/2),2))
	pointRelated = fig15.gca().scatter([x_Coord[lowerInd]], [labels[upperInd]],
		c = 'b')
	if (isLine):
		line.pop(0).remove()
		isLine = False
	crossRelatedScatter.append(pointRelated)
	plt.tight_layout()
	fig15.canvas.draw()

def removeSegmentNormalization(x_Initial, x_End):
	coloredSegmentX[order].extend([np.arange(x_Initial,x_End+1,1).tolist()])
	coloredSegmentY[order].extend([np.array(spectrum[order,0,x_Initial:x_End+1]).tolist()])
	maskedSegmentX[order].extend([x_Coord_Masked[x_Initial:x_End+1].tolist()])
	maskedSegmentY[order].extend([y_Coord_Up_Masked[x_Initial:x_End+1].tolist()])

	x_Coord_Masked[x_Initial:x_End+1] = ma.masked
	y_Coord_Up_Masked[x_Initial:x_End+1]=ma.masked

	axFittedSpectrum.plot(np.arange(x_Initial,x_End+1,1),np.array(spectrum[order,0,x_Initial:x_End+1]),'y-')
	

	figFittedSpectrum.canvas.draw()

def validateSlice():
	jump = int(entry_slice.get())

	if (jump > 0):
		isValid = True
		jump_Array[order]=jump
	else:
		isValid = False

	return jump,isValid
def validateOrder ():
	orderNrm = int(entry_nrm.get())

	if (orderNrm>0):
		isValid=True
		normalizationOrder_Array[order]=orderNrm
	else:
		isValid = False

	return orderNrm,isValid

def updateDegree():
	global fittedLine, normalizedLine, spectrumOrig, spectrumNormalized, jump, normalizationOrder, jump_Array, normalizationOrder_Array


	jump,isValidJump = validateSlice()
	normalizationOrder,isValidOrder =validateOrder()

	if(isValidJump and isValidOrder):
		fittedLine.pop(0).remove()
		normalizedLine.pop(0).remove()

		normalization_Array[order_g] =np.array(np.polyfit(x_Coord_Masked.compressed()[::jump_Array[order]], y_Coord_Up_Masked.compressed()[::jump_Array[order]], normalizationOrder_Array[order]))
		normalization_function = np.poly1d(normalization_Array[order_g])


		fittedLine = axFittedSpectrum.plot(normalization_function(x_Dimension))
		spectrumNormalized[order] = np.copy(spectrum[order,0])/normalization_function(x_Dimension)
		normalizedLine=axNormalizedSpectrum.plot(spectrumNormalized[order, overScan_location[0]:overScan_location[1]],'k-')
		axNormalizedSpectrum.plot(comparisonLine, 'r--')

		figFittedSpectrum.canvas.draw()
		figNormalizedSpectrum.canvas.draw()

	else:
		if(not isValidJump):
			message = "Slice value invalid. \nPlease type an integer greater than 1"
			dialog(message,'i')
		if (not isValidOrder):
			message = "Order of fit invalid. \nPlease type a numebr greater than 1"
			dialog(message, 'i')
	jump_Array[order] = jump
	normalizationOrder_Array[order]=normalizationOrder

def clearRemovedPoints():
	global coloredSegmentX, coloredSegmentY, maskedSegmentX, maskedSegmentY, x_Coord_Masked, y_Coord_Up_Masked, isNormalized, isCleared

	coloredSegmentX[order]=[]
	coloredSegmentY[order]=[]
	maskedSegmentX[order]=[]
	maskedSegmentY[order]=[]

	x_Coord_Masked.mask = False
	y_Coord_Up_Masked.mask = False

	isNormalized[order] = False

	isCleared[order]=True

	updateNormalizePlot(order)

def clearSelectedPoints():
	ax[0].cla()
	ax[1].cla()
	ax[2].cla()
	updateOrderPlot()


#Navigation functions for windows with multiple orders

def advanceNormalized ():
	global order
	if ((order+1) < len(weightArray)):
		order_loc = order + 1
		updateNormalizePlot(order_loc)
		order = order_loc
	else:
		message = 'End of orders'
		dialog(message,'i')

def previousNormalized ():
	global order
	if ((order-1) >= 0):
		order_loc = order-1
		updateNormalizePlot(order_loc)
		order = order_loc
	else:
		message = 'End of orders'
		(message,'i')

def goNormalizedTo ():
	global order
	order = int(entry.get())
	if ((order >= 0) and (order < len(weightArray))):
		updateNormalizePlot(order)
	else:
		message = 'Order out of bounds!'
		dialog(message, 'i')

def advanceGaussOrder ():
	global order
	if ((order+1) < len(weightArray)):
		order_loc = order + 1
		updatePlotGauss(order_loc)
		order = order_loc
	else:
		message = 'End of orders'
		dialog(message,'i')

def previousGaussOrder ():
	global order
	if ((order-1) >= 0):
		order_loc = order-1
		updatePlotGauss(order_loc)
		order = order_loc
	else:
		message = 'End of orders'
		(message,'i')

def goGaussTo ():
	global order
	order = int(entry.get())
	if ((order >= 0) and (order < len(weightArray))):
		updatePlotGauss(order)
	else:
		message = 'Order out of bounds!'
		dialog(message, 'i')

def advanceSensOrder ():
	global order
	if ((order+1) < len(surface_val2)):
		order_loc = order + 1
		updatePlotSens(order_loc)
		order = order_loc
	else:
		message = 'End of orders'
		dialog(message,'i')

def previousSensOrder ():
	global order
	if ((order-1) >= 0):
		order_loc = order-1
		updatePlotSens(order_loc)
		order = order_loc
	else:
		message = 'End of orders'
		dialog(message,'i')

def goSensTo ():
	global order
	order = int(entry.get())
	if ((order >= 0) and (order < len(surface_val2))):
		updatePlotSens(order)
	else:
		message = 'Order out of bounds!'
		dialog(message, 'i')

def advanceOrder ():
	global order
	if ((order+1) < len(spectrum_In)):
		order_loc = order + 1
		updatePlotShow(order_loc)
		order = order_loc
	else:
		message = 'End of orders'
		dialog(message,'i')

def previousOrder ():
	global order
	if ((order-1) >= 0):
		order_loc = order-1
		updatePlotShow(order_loc)
		order = order_loc
	else:
		message = 'End of orders'
		dialog(message,'i')

def goTo ():
	global order
	order = int(entry.get())
	if ((order >= 0) and (order < len(spectrum_In))):
		updatePlotShow(order)
	else:
		message = 'Order out of bounds!'
		dialog(message, 'i')

def advanceUpOrder ():
	global order
	if ((order+1) < len(spectrum_In)):
		order_loc = order + 1
		updatePlot(order_loc)
		order = order_loc
	else:
		message = 'End of orders'
		dialog(message,'i')

def previousUpOrder ():
	global order
	if ((order-1) >= 0):
		order_loc = order-1
		updatePlot(order_loc)
		order = order_loc
	else:
		message = 'End of orders'
		dialog(message,'i')

def goUpTo ():
	global order
	order = int(entry_up.get())
	if ((order >= 0) and (order < len(spectrum_In))):
		updatePlot(order)
	else:
		message = 'Order out of bounds!'
		dialog(message, 'i')

def advanceDwnOrder ():
	global orderLow
	if ((orderLow+1) < len(spectrum_In)):
		order_loc = orderLow + 1
		updateLowPlot(order_loc)
		orderLow = order_loc
	else:
		message = 'End of orders'
		dialog(message,'i')

def previousDwnOrder ():
	global orderLow
	if ((orderLow-1) >= 0):
		order_loc = orderLow-1
		updateLowPlot(order_loc)
		orderLow = order_loc
	else:
		message = 'End of orders'
		dialog(message,'i')

def goDwnTo ():
	global orderLow
	orderLow = int(entry_dwn.get())
	if ((orderLow >= 0) and (orderLow < len(spectrum_In))):
		updateLowPlot(orderLow)
	else:
		message = 'Order out of bounds!'
		dialog(message, 'i')


#Generates the regression from the correlated points
#Of wavelength and pixel coordinate
def regression():
	global fig3,z, isLine, line, isLabel, chi_Label
	x_val = crossRelatedPoints[:,0]
	y_val = crossRelatedPoints[:,1]

	z_Arr = []
	f_Arr = []
	chi_Arr = []
	points = len(x_val)

	if (isLine):
		line.pop(0).remove()
		isLine = False



	i = 0

	#Checks for the best fit line, smallest Chi-Squared
	#Uses odd functions to avoid repetition of wavlength values for 2 different coordinates
	while ((2*i+1)< points-1):
		z = np.polyfit(x_val,y_val, (2*i+1))
		f = np.poly1d(z)
		y_Expected = f(x_val)

		chi_Squared,_= chisquare (y_val, y_Expected, ddof = points-1)

		z_Arr.append(z)
		f_Arr.append(f)
		chi_Arr.append(chi_Squared)

		i = i+1


	bestFit = np.argmin(chi_Arr)



	x_Range = np.linspace(0, g_length, g_length)

	z = z_Arr[bestFit]
	f = f_Arr[bestFit]

	plt.figure(3)
	line = ax15.plot(f(x_Range))
	chi_Label = ax15.annotate("Chi-Squared: "+str(chi_Squared), xy=(0.2,0.8), xycoords="axes fraction",
									va="center", ha="center", bbox=dict(boxstyle="square", fc="w"))
	fig15.canvas.draw()
	isLine = True
	isLabel = True

#Stores the desired regreesion into the folder with the date.
def saveRegression():
	global z, orderLow
	
	name_File = str(orderLow)
	saveToSecondaryFile(name_File, z)

#Sets the degree of the fit determined by the user
#Raises warings for even functions and for degrees smaller than the numebr of
#data points
def regressionSpecific():
	global fig3,z, isLine, line, isLabel, chi_Label

	x_val = crossRelatedPoints[:,0]
	y_val = crossRelatedPoints[:,1]

	points = len(x_val)
	degree = int(entry_rt.get())
	if (isLine):
		line.pop(0).remove()
		isLine = False
	if (isLabel):
		chi_Label.remove()
		isLabel = False
	i = 0

	#Warning for degree higher than number of data
	if (degree >= points-1):
		message = '''Warning: The degree of the polynomial to be fit, is higher or
	equal than the number of points - 1. This means, that there can
	be a perfect fit to the points.
	'''
		dialog(message,'i')

	#Warning for even functions
	elif (degree%2 == 0):
		message = '''Warning: Fitting an even function could cause two different
	pixel coordinates to be related to the same wavelength'''
		dialog (message,'i')

	z = np.polyfit(x_val,y_val, degree)
	f = np.poly1d(z)
	y_Expected = f(x_val)

	chi_Squared,_= chisquare (y_val, y_Expected, ddof = points-1)
	
	i = i+1



	x_Range = np.linspace(0, g_length, g_length)

	plt.figure(3)
	line = ax15.plot(f(x_Range))
	chi_Label = ax15.annotate("Chi-Squared: "+str(chi_Squared), xy=(0.2,0.8), xycoords="axes fraction", va="center",
		ha="center", bbox=dict(boxstyle="square", fc="w"))
	fig15.canvas.draw()
	isLabel = True
	isLine = True

#Receives a message and a type of dialog box to be shown
def dialog (message, kind):
	#Information
	if (kind == 'i'):
		tkMessageBox.showinfo("SpecTracer", message)
	#Integer entry
	elif (kind == 'e'):
		number = tkSimpleDialog.askinteger("SpecTracer", message)
		return number
	#Float entry
	elif (kind == 'f'):
		number = tkSimpleDialog.askfloat("SpecTracer", message)
		return number
	#Yes No question
	elif (kind =='q'):
		answer = tkMessageBox.askyesno("SpecTracer", message)
		return answer

#Sends instructions for the order tracing
def orderCalibInstructions():
	global fitX, thick
	if len(fitX)==1:
		message = "Click on the vertex of the order"
		dialog(message,'i')

	elif len(fitX)==2:
		message =  "Click on the coordinate of the order at x = " + str(g_length)
		dialog(message,'i')

	elif len(fitX)==3:
		message = "Please input the thickness of the order"
		thick = dialog(message, 'e')
		message = "Double Click in the order positions.\nIt is recommended to leave a 2px padding.\nDouble right click when finished or click the Done button"
		dialog(message, 'i')
#Main Modules
def orderTracer ():
	instruction = 'Please type the name of the file from where the orders \nwill be read. (Flat Lamp)'
	
	nameFile_orderCalib = getFileName(instruction)

	global g_length, g_width, ax2, fig2, ax1, fig1, g_les, g_drift, ax, fig19, fig20, ax20,ax4
	global fitX, fitY, g_proper_orderLoc, g_multiplier, correction, v_pan, h_pan, calibrationWindow, orderTracerWindow
	global counter, thick, orderCalibrationWin,orderCorrectWin, orderCanvas, g_orderLoc, cid, medianBias_val, meanOverScan_val
	global corrector_left, corrector_right, amount_left, amount_vertex, amount_right, labels_left, labels_right
	global labels_vertex, maximum_left, maximum_vertex, maximum_right, z, maxima_Row_Fit, maxima_Row_Function, columns
	global left_location, vertex_location, right_location, g_data,f, entry_thick, overScanLeft,overScanRight,lowerVal

	medianBias_val = openNumpyFile('bias')
	meanOverScan_val = openNumpyFile('overscan.npy')
	overScan_location = openNumpyFile('overscanloc.npy')

	data = fits.getdata(nameFile_orderCalib)
	isLandscape, g_length, g_width= getDimensions (nameFile_orderCalib)
	g_data = forceLandscape(data, isLandscape)

	g_data = g_data-medianBias_val-meanOverScan_val
	overScanLeft=overScan_location[0]
	overScanRight=overScan_location[1]

	toDelete_left=np.array([])
	toDelete_vertex=np.array([])
	toDelete_right=np.array([])

	
	vertex_location=np.array([len(g_data[0])/2])
	

	columns = np.ones(shape = (3, len(g_data)))
	
	columns[1] = g_data[::, len(g_data[0])/2]
	

	message = "The image of the Flat Lamp is shown.\nIt can be used for reference for the following steps,"
	dialog(message, 'i')

	orderCalibrationWin = Tk()
	orderCalibrationWin.title('SpecTracer')

	helv20 = tkFont.Font(family ='Helvetica', size = 10, weight = 'bold')

	fig1 = plt.figure(figsize= (10,6))
	ax1 = fig1.add_subplot(111)

	ax1.imshow(g_data, origin = 'lower')
	ax1.set_xlabel('Pixel')
	ax1.set_ylabel('Pixel')
	ax1.set_title('CCD')
	
	toolbar_frame=Frame(orderCalibrationWin)

	done_button = Button(orderCalibrationWin, text = 'Done', command = doneThick)
	orderCanvas = FigureCanvasTkAgg(fig1, master=orderCalibrationWin)
	label_thick = Label(orderCalibrationWin,text='Thickness: ',font=helv20)
	entry_thick = Entry(orderCalibrationWin)
	
	done_button.grid(row=3, column = 2, sticky = W)

	toolbar_frame.grid(row=0, column=0, columnspan =2, sticky = W)

	label_thick.grid(row=3,column=0,columnspan=1,sticky=E)
	entry_thick.grid(row=3,column=1,columnspan=1,sticky=W)
	toolbar = NavigationToolbar2TkAgg(orderCanvas, toolbar_frame)
	orderCanvas.get_tk_widget().grid(row=1, column = 0, columnspan = 3,rowspan=1)
	
	fig1.canvas.draw()
	
	message = "Please input the thickness of the order"
	dialog(message, 'i')

	
	orderCalibrationWin.protocol("WM_DELETE_WINDOW", closeOrderCalibWin)
	orderCalibrationWin.mainloop()


	maximum_vertex = argrelextrema(columns[1], np.greater_equal, order=thick/2)[0]
	lowerVal = maximumDiscriminator(maximum_vertex, g_length, overScanLeft,overScanRight)

	print lowerVal


	columns[0] = g_data[::,overScanLeft+1]
	columns[2] = g_data[::, overScanRight-1]

	maximum_left= argrelextrema(columns[0,int(lowerVal[0]-thick/2):], np.greater_equal, order=thick/2)[0]+int(lowerVal[0]-thick/2)
	maximum_right = argrelextrema(columns[2,int(lowerVal[-1]-thick/2):], np.greater_equal, order=thick/2)[0]+int(lowerVal[-1]-thick/2)

	i=0

	while (i<len(maximum_left)-1):
		if (maximum_left[i]==(maximum_left[i+1]-1)):
			toDelete_left = np.append(toDelete_left, [i])
		i=i+1

	i=0	
	while (i<len(maximum_vertex)-1):
		if (maximum_vertex[i]==(maximum_vertex[i+1]-1)):
			toDelete_vertex = np.append(toDelete_vertex, [i])
		i=i+1

	i=0
	while (i<len(maximum_right)-1):
		if (maximum_right[i]==(maximum_right[i+1]-1)):
			toDelete_right = np.append(toDelete_right, [i])
		i = i+1


	maximum_left = np.delete(maximum_left, toDelete_left)
	maximum_vertex = np.delete(maximum_vertex, toDelete_vertex)
	maximum_right = np.delete(maximum_right, toDelete_right)
	'''
	row = g_data[maximum_vertex[0]-thick/4,overScanLeft:overScanRight]
	x_Coord=np.arange(0,len(row), 1)
	maxima_Row_Fit = np.polyfit (x_Coord, row, 20)
	maxima_Row_Function = np.poly1d(maxima_Row_Fit)

	max_row = argrelextrema(maxima_Row_Function(x_Coord), np.greater_equal, order =thick/2)[0]

	vertex_Row_Index = checkVertex(max_row, g_length)
	'''

	

	amount_left=np.arange(0,len(maximum_left),1)
	amount_vertex=np.arange(0,len(maximum_vertex),1)
	amount_right=np.arange(0,len(maximum_right),1)

	labels_left = ['{0}'.format(i) for i in range(len(maximum_left))]
	labels_vertex = ['{0}'.format(i) for i in range(len(maximum_vertex))]
	labels_right = ['{0}'.format(i) for i in range(len(maximum_right))]

	
	orderTracerWindow = Tk()

	orderTracerWindow.title ("SpecTracer")
	fig19, ax = plt.subplots(3, sharex=True)

	toolbar_frame=Frame(orderTracerWindow)
	toolbar_frame_dwn=Frame(orderTracerWindow)

	fig20 = plt.figure (figsize =(8,4))
	ax20 = fig20.add_subplot (111)
	ax20.imshow(g_data, origin = 'lower')
	
	helv20 = tkFont.Font(family ='Helvetica', size = 10, weight = 'bold')

	done_button = Button(orderTracerWindow, text="Done",
					command = autoFitter, font = helv20)
	clear_button = Button(orderTracerWindow, text="Clear",
					command = clearSelectedPoints, font = helv20)
	close = Button(orderTracerWindow, text = "Close",
					command = closeTraceWind, font = helv20)

	canvas_up = FigureCanvasTkAgg(fig20, master=orderTracerWindow)

	done_button.grid(row=3, column = 0, sticky = W)
	clear_button.grid(row=3, column = 1, sticky = W)
	close.grid(row=3, column = 2, sticky = W)
	canvas_dwn = FigureCanvasTkAgg(fig19, master=orderTracerWindow)


	toolbar_frame.grid(row=0, column=0, columnspan =2, sticky = W)
	toolbar_frame_dwn.grid(row=0, column=4, columnspan =4, sticky = W)
	toolbar = NavigationToolbar2TkAgg(canvas_up, toolbar_frame)
	toolbar_dwn = NavigationToolbar2TkAgg(canvas_dwn, toolbar_frame_dwn)
	canvas_up.get_tk_widget().grid(row=2, column = 0, columnspan = 4)
	canvas_dwn.get_tk_widget().grid(row=2, column = 4, columnspan = 4, padx=3)
	
	
	fig19.tight_layout()
	fig20.canvas.draw()

	message = """A slice perpendicular to the orders has been taken to identify them
In the plot corresponding to the vertex, please drag a box
That includes all the orders of interest. Even though the orders
may look clearly defined on this plot, it is important to consider
the slices at the edges. Only choose those that are clearly identified
in all three of the plots. Other orders can be appended later."""
	dialog(message, 'i')

	updateOrderPlot()

	orderTracerWindow.protocol("WM_DELETE_WINDOW", closeTraceWind) 	 	
	orderTracerWindow.mainloop()
	
	g_multiplier = np.ones(len(z[0]))
	g_drift = np.zeros(len(z[0]))
	v_pan = np.zeros(len(z[0]))
	h_pan = np.zeros(len(z[0]))
	g_orderLoc = np.array((z[2]))

	z[2]=0

	f= fitReader(z)
	saveToFile ('fit' ,z)
	saveToFile ('thick', thick)

	x_new = np.linspace(0,g_length,g_length)

	orderCorrectWin = Tk()
	orderCorrectWin.title('SpecTracer')
	fig2 = plt.figure(figsize= (20,10))
	ax2 = fig2.add_subplot(111)
	toolbar_frame=Frame(orderCorrectWin)
	orderCorrectCanvas = FigureCanvasTkAgg(fig2, master=orderCorrectWin)
	toolbar_frame.grid(row=0, column=0, columnspan =2, sticky = W)
	toolbar = NavigationToolbar2TkAgg(orderCorrectCanvas, toolbar_frame)
	orderCorrectCanvas.get_tk_widget().grid(row=2, column = 0, columnspan = 4)

	ax2.axis([0,g_length,0,g_width])
	ax2.set_xlabel('Pixel')
	ax2.set_ylabel('Pixel')
	fitGrapher(x_new, ax2, fig2)


				
	message = "Some of the orders may not be traced \nvery accurately (specially at the edges) \nIf the bottom order is order 0, please \ntype the number of the order you would \nlike to modify to create a better fit. \nIf no order needs to be changed anymore \nplease type a -1"
	choice = dialog(message, 'e')

	while (choice >=0):

		#Make corrections to each of the fits
		
		message = "Type a number between 0 (not inclusive) and 9999 \nRemember that if < 1, the fit will widen \nand if >1 the fit will get more narrow"
		correction = dialog(message, 'f')

		message = "Type a number to pan the graph upward or downward"
		pan = dialog(message, 'e')

		message = "	Type a number to pan the graph to the right or left	\n(positive is to the right, negative to the left)"
		dan = dialog(message, 'f')

		g_multiplier[choice] = g_multiplier[choice]*correction
		v_pan[choice]=v_pan[choice]+pan
		g_drift[choice]=g_drift[choice]+dan


		ax2.cla()
		fitGrapher (x_new, ax2, fig2)

		
		message = "Some of the orders may not be traced \nvery accurately (specially at the edges) \nIf the bottom order is order 0, please \ntype the number of the order you would \nlike to modify to create a better fit. \nIf no order needs to be changed anymore \nplease type a -1"
		choice = dialog(message, 'e')
	
	orderCorrectWin.protocol('WM_DELETE_WINDOW',closeOrderCorrectWin)
	orderCorrectWin.mainloop()

	saveToFile ('correction', g_multiplier)
	g_orderLoc = g_orderLoc+v_pan-thick/2
	saveToFile ('displacement', g_orderLoc)
	saveToFile ('drift', g_drift)
	
def doneOverscan():
	global left, right
	left=int(entry_start.get())
	right=int(entry_end.get())
	overScanWin.quit()
	overScanWin.destroy()

def doneThick():
	global thick
	thick=int(entry_thick.get())
	orderCalibrationWin.quit()
	orderCalibrationWin.destroy()


def biasDarkCalibration ():
	global isBias, isDark, is_Primary, bias_dir, dark_dir, entry, instrument, isFiles
	global bias, dark, overScanWin, overScanFitWin, overScanWin,entry_end,entry_start,left,right
	instruction = 'Please select file from where the orders \nwill be read. (Flat Lamp)'
	nameFile_flat = getFileName(instruction)
	flat_data = fits.getdata(nameFile_flat) 
	isLandscape, length, width = getDimensions(nameFile_flat)
	flat_data = forceLandscape(flat_data, isLandscape)

	message = "Is there an overscan?"
	choice = dialog(message, 'q')

	if (choice):
		helv20 = tkFont.Font(family ='Helvetica', size = 10, weight = 'bold')

		overScanWin = Tk()
		overScanWin.title('SpecTracer')
		fig3 = plt.figure(figsize= (10,5))
		ax3 = fig3.add_subplot(111)
		toolbar_frame=Frame(overScanWin)

		
		overScanCanvas = FigureCanvasTkAgg(fig3, master=overScanWin)

		toolbar_frame.grid(row=0, column=0, columnspan =2, sticky = W)

		done_button = Button(overScanWin,text='Done',command=doneOverscan)
		label_start = Label(overScanWin, text = 'Left Overscan: ', font = helv20)
		label_end = Label(overScanWin,text='Right Overscan: ',font=helv20)
		
		entry_start = Entry(overScanWin)
		entry_end = Entry(overScanWin)
		toolbar = NavigationToolbar2TkAgg(overScanCanvas, toolbar_frame)

		overScanCanvas.get_tk_widget().grid(row=2, column = 0, rowspan=4,columnspan = 1)
		label_start.grid(row=2,column =1,rowspan=1,columnspan=1,sticky=W)
		entry_start.grid(row=2,column =2,rowspan=1,columnspan=1,sticky=W)
		label_end.grid(row=3,column =1,rowspan=1,columnspan=1,sticky=W)
		entry_end.grid(row=3,column =2,rowspan=1,columnspan=1,sticky=W)
		done_button.grid(row=4,column=1)

		ax3.imshow(flat_data, origin = 'lower')
		ax3.set_xlabel('Pixel')
		ax3.set_ylabel('Pixel')
		ax3.set_title('CCD')
		fig3.canvas.draw()


		message = "Please input the coordinate where the left overscan ends and the right overscan starts"
		dialog(message,'i')

		overScanWin.protocol('WM_DELETE_WINDOW', closeOverScanWin)
		overScanWin.mainloop()

		left_Over = np.sum(flat_data[:,0:left], axis = 1)
		right_Over = np.sum(flat_data[:,right:length], axis = 1)

		overScan = left_Over+right_Over
		overScan = overScan/((left+1)+(length-right))

		x_new = np.linspace (0, len(overScan), len(overScan))

		z = np.polyfit(x_new, overScan, 0)
		f = np.poly1d(z)
		
		overScanFitWin = Tk()
		overScanFitWin.title('SpecTracer')

		toolbar_frame=Frame(overScanFitWin)
		fig4 = plt.figure(figsize= (10,5))
		ax4 = fig4.add_subplot(111)
		overScanFitCanvas = FigureCanvasTkAgg(fig4, master=overScanFitWin)
		toolbar_frame.grid(row=0, column=0, columnspan =2, sticky = W)
		toolbar = NavigationToolbar2TkAgg(overScanFitCanvas, toolbar_frame)
		overScanFitCanvas.get_tk_widget().grid(row=2, column = 0, columnspan = 4)
		
		plt.plot(f(x_new))
		plt.scatter (x_new, overScan)
		fig4.canvas.draw()

		overScanFitWin.protocol("WM_DELETE_WINDOW", closeOverScanFitWin)
		overScanFitWin.mainloop()

		meanOverScan_val = f(0)


	else:
		meanOverScan_val = 0
		left = 0
		right = length

	message = "Select the folder containing the bias files"
	folderCreation(False, True, False, False, False, message)

	biases_arr = np.array([])
	for root, dirs, filenames in os.walk(bias_dir):
		for filename in filenames:
			if filename.endswith('.fit') or filename.endswith('.fits'):
				biases_arr = np.append(biases_arr,[os.path.join(bias_dir, filename)])

	temp_data = fits.getdata(biases_arr[0])

	bias_data = np.ndarray(shape=(len(biases_arr),len(temp_data), len(temp_data[0])))
	

	i = 0	
	while i < len(biases_arr):
		bias_data[i] = fits.getdata(biases_arr[i])-meanOverScan_val
		bias_data[:,0:left] = ma.masked
		bias_data[:,right:length]=ma.masked

		i=i+1
	medianBias_val = ma.median(bias_data, 0)

	message = "Are Dark files available?"
	choice = dialog(message, 'q')
	if (choice):
		message = "Select the folder containg the dark files"
		folderCreation(False, False, True, False, False, message)

		darks_arr = np.array([])
		for root, dirnames, filenames in os.walk(dark_dir):
			for filename in filenames:
				if filename.endswith('.fit') or filename.endswith('.fits'):
					darks_arr = np.append(darks_arr,[os.path.join(dark_dir, filename)])

		temp_data = fits.getdata(darks_arr[0])
		dark_data = np.ndarray(shape=(len(darks_arr),len(temp_data), len(temp_data[0])))
	

		i = 0	
		while i < len(darks_arr):
			dark_data[i] = fits.getdata(darks_arr[i])-meanOverScan_val-medianBias_val
			dark_data[:,0:left] = ma.masked
			dark_data[:,right:length]=ma.masked
			i=i+1
		medianDark_val = ma.median(dark_data, 0)

	else:
		medianDark_val = np.ndarray(shape = (np.shape(medianBias_val)))
		medianDark_val[:]=0


	saveToFile('bias', medianBias_val)
	saveToFile('dark', medianDark_val)
	saveToFile('overscan', meanOverScan_val)
	saveToFile('overscanloc', [left, right])
	
def sensitivityCalibration (): 
	global order, fig5, ax5, entry, xPoints2, yPoints2, flat_flux,X2,Y2,surface_val2
	global sensitivityWin,flatFieldWin, original_Y
	instruction = "	Please select the file from where the orders\nwill be read. (Flat Lamp)"
	nameFile_flat = getFileName(instruction)

	flat_data = fits.getdata(nameFile_flat) 
	isLandscape, length, width = getDimensions(nameFile_flat)
	flat_data = forceLandscape(flat_data, isLandscape)
	medianBias_val = openNumpyFile('bias')
	medianDark_val = openNumpyFile('dark')
	meanOverScan_val = openNumpyFile('overscan.npy')
	overScan_location = openNumpyFile('overscanloc.npy')
	order_displace = openNumpyFile("displacement.npy")
	order_thick = openNumpyFile("thick.npy")
	order_multiply = openNumpyFile("correction.npy")
	order_drift = openNumpyFile("drift.npy")
	z = openNumpyFile("fit.npy")


	flat_data = flat_data-medianBias_val-meanOverScan_val

	
	g = fitReader(z)	

	xPoints = np.arange(0,length,1)
	xPoints = np.tile (xPoints, order_thick)
	xPoints.shape = (len(xPoints),1)
	yPoints = np.arange (0,order_thick,1)
	yPoints = np.repeat(yPoints, length)
	yPoints.shape = (len(yPoints),1)

	original_Y = np.ndarray(shape=(len(order_displace),order_thick,length))

	flat_unfold = ma.zeros((len(order_displace), order_thick, length))
	surf_data = ma.zeros ((len(order_displace), len(yPoints), 3))
	surface_val = np.ndarray(shape = (len(order_displace), width, length))
	surface_val2 = np.ndarray(shape = (len(order_displace), width, length))
	flat_flux = np.ndarray(shape = (len(order_displace), width, length))
	flat_unfold[:,:,0:overScan_location[0]] = ma.masked
	flat_unfold[:,:,overScan_location[1]:length]=ma.masked	

	#Pixel by pixel trace while storing in an array that has the shape of 
	#the image but just contains the order
	k = 0
	while (k<len(order_displace)):
		displace = order_displace[k]
		count = 0
		while (count < order_thick):
			i = 0
			while (i < length):
				t=g[k]
				t.c[0]=order_multiply[k]*g[k][2]
				t.c[1]=t.c[1]-order_drift[k]
				j = int(t(i))
				pan = int(j + displace+count)
				original_Y[k,count,i]=pan
				flat_unfold[k,count,i]= flat_data[pan,i]
				flat_flux[k,pan,i]= flat_data[pan,i]
				if (flat_unfold[k,count,i]<0):
					flat_unfold[k,count,i]=0
					flat_flux[k,pan,i] = 0
				g[k][2] = g[k][2]/order_multiply[k]
				g[k][1]=g[k][1]+order_drift[k]
				i = i+1
			count = count +1

		temp_data= flat_unfold[k].flatten()
		temp_data.shape = (len(temp_data), 1)
		temp_table = np.concatenate ((xPoints, yPoints, temp_data), axis = 1)

		surf_data[k]=temp_table

		k = k + 1

	flat_flux = ma.masked_equal(flat_flux,0)

	k = 0
	i = 0

	xPoints2 = np.arange(0,length,1)
	xPoints2 = np.tile (xPoints2, width)
	xPoints2.shape = (len(xPoints2),1)
	yPoints2 = np.arange (0,width,1)
	yPoints2 = np.repeat(yPoints2, length)
	yPoints2.shape = (len(yPoints2),1)

	X2, Y2 = np.meshgrid(np.linspace(0,length,length),
							np.linspace(0,width,width))

	while (i< len(order_displace)):
		sig = np.std(surface_val2[i])
		surface_val2[i]= ndimage.gaussian_filter(flat_flux[i], sigma = (0.05,75), order = 0)
		i = i+1
	

	sensitivityWin = Tk()

	sensitivityWin.title ("SpecTracer")
	fig5 = plt.figure(figsize= (10,5))
	ax5 = fig5.add_subplot(111)

	toolbar_frame=Frame(sensitivityWin)

	next_button = Button(sensitivityWin, text="Next Order", command = advanceSensOrder)
	prev_button = Button(sensitivityWin, text="Previous Order", command = previousSensOrder)
	goTo_button = Button(sensitivityWin, text = "Go!", command = goSensTo)
	entry = Entry(sensitivityWin)

	next_button.grid(row=4, column = 3, sticky = S)
	prev_button.grid(row=4, column = 0, sticky = W)
	goTo_button.grid(row=4, column = 2, sticky = S)
	entry.grid(row=4, column = 1, sticky = S)
	sensCanvas = FigureCanvasTkAgg(fig5, master=sensitivityWin)

	toolbar_frame.grid(row=0, column=0, columnspan =2, sticky = W)
	toolbar = NavigationToolbar2TkAgg(sensCanvas, toolbar_frame)
	sensCanvas.get_tk_widget().grid(row=2, column = 0, columnspan = 4)
	


	updatePlotSens(0)

	order = 0

	sensitivityWin.protocol("WM_DELETE_WINDOW", closeSensitivityWin) 	
	sensitivityWin.mainloop()

	flat_field2 = flat_flux/surface_val2


	flatFieldWin = Tk()
	flatFieldWin.title('SpecTracer')
	toolbar_frame=Frame(flatFieldWin)

	fig6 = plt.figure(figsize= (10,5))
	ax6 = fig6.add_subplot(111)
	flatFieldCanvas = FigureCanvasTkAgg(fig6, master=flatFieldWin)
	toolbar_frame.grid(row=0, column=0, columnspan =2, sticky = W)
	toolbar = NavigationToolbar2TkAgg(flatFieldCanvas, toolbar_frame)
	flatFieldCanvas.get_tk_widget().grid(row=2, column = 0, columnspan = 4)
	
	plt.imshow(flat_field2[0],origin='lower')
	plt.title('Flat Field')
	fig6.canvas.draw()

	

	flatFieldWin.protocol("WM_DELETE_WINDOW", closeFlatFieldWin)
	flatFieldWin.mainloop()
	
	saveToFile('flatfield', flat_field2)

def getMultiFileName (instruction):
	global file_List, entry, fileBox, isFiles, name


	if (not isFiles):
		file_Name = ''
		isFiles = True

	#Tkinter object initialize
	fileBox = Tk()

	fileBox.title ("SpecTracer")
	helv20 = tkFont.Font(family ='Helvetica', size = 10, weight = 'bold')
	
	#Window Components initialization
	primary_label = Label(fileBox, text = instruction, font = helv20)
	browse_button = Button(fileBox, text = "Browse", command = browseMultiFile, font = helv20)
	ok_button = Button(fileBox, text="OK", command = closeMultiFile, font = helv20)
	entry = Entry(fileBox, width = 50)

	#Component Location
	primary_label.grid (row = 0, column = 0,columnspan = 10, sticky = N)
	browse_button.grid (row = 1, column =1, sticky = E, padx = 5, pady = 5)
	ok_button.grid (row = 1, column =2, sticky = E)
	entry.grid(row = 1, column = 0, pady = 5, sticky = S)

	fileBox.mainloop()
	
	return name

def browseMultiFile ():
	global file_List

	Tk().withdraw()
	file_List = askopenfilenames()

	entry.delete(0, END)
	entry.insert(0, file_List)

def readMultiFile(name_list, length, width, bias, dark, overScan):
	i=0
	data_cube = np.ndarray(shape=(len(name_list),width, length))
	while (i<len(name_list)):
		data_cube[i]=fits.getdata(name_list[i])
		i =i+1

	data_cube = data_cube-bias
	data_cube = data_cube-dark
	data_cube = data_cube-overScan

	spectrum_data = np.median(data_cube,0)

	return spectrum_data


	
def specTracer ():
	global spectrum, isPlotted, spectrumWin, gaussianWin, flatWin, entry
	global weightArray, reducedSpectWin, fig8, fig9,ax8,ax9, gauss_step, flat_gauss, order, x_Dimension,normalization_Array
	global figFittedSpectrum, figNormalizedSpectrum, axFittedSpectrum, axNormalizedSpectrum, spectrumNormalized, isNormalized, normalizeSpectWin,rect
	global canvas_NormalizedSpectrum, entry_nrm, entry_slice, maskedSegmentX, maskedSegmentY, coloredSegmentY, coloredSegmentX, isCleared
	global jump_Array, normalizationOrder_Array, normalizedFileExists, overScan_location

	z = openNumpyFile ('fit.npy')
	displacement = openNumpyFile('displacement.npy')
	correct = openNumpyFile('correction.npy')
	thick = openNumpyFile('thick.npy')
	order_drift = openNumpyFile('drift.npy')
	bias = openNumpyFile('bias')
	dark = openNumpyFile('dark')
	overScan = openNumpyFile('overscan.npy')
	overScan_location = openNumpyFile('overscanloc.npy')
	flatF = openNumpyFile('flatfield')
	
	maske = ma.getmaskarray(flatF)


	instruction = 'Please type the name of the FITS file \ncontaining the Flat Lamp'
	flat_file = getFileName(instruction)
	flat_data = fits.getdata(flat_file) 

	isLandscape_flat, length_flat, width_flat = getDimensions(flat_file)
	flat_data = forceLandscape(flat_data, isLandscape_flat)
	flat_data = flat_data-bias-overScan

	instruction =  'Please type the name of the file containing the spectrum'
	spectrum_list = getMultiFileName(instruction)

	message = 'Is the file a calibration lamp?'
	choice = dialog(message, 'q')

	if (not (choice)):
		spect_data = readMultiFile(spectrum_list, length_flat, width_flat, bias, dark, overScan)
	else:
		spect_data=fits.getdata(spectrum_list[0])
		spect_data = spect_data-bias
		spect_data = spect_data-overScan


	isLandscape, length, width = getDimensions(spectrum_list[0])
	spect_data = forceLandscape(spect_data, isLandscape)

	isPlotted = False
	
	spect_data[:,0:overScan_location[0]] = ma.masked
	spect_data[:,overScan_location[1]:length]=ma.masked
	invalidFlux=np.where(spect_data<0.0)
	spect_data[invalidFlux[0],invalidFlux[1]]=0


	flat_data[:,0:overScan_location[0]] = ma.masked
	flat_data[:,overScan_location[1]:length_flat]=ma.masked


	spectrumWin = Tk()
	spectrumWin.title('SpecTracer')

	toolbar_frame=Frame(spectrumWin)

	fig7 = plt.figure(figsize= (10,5))
	ax7 = fig7.add_subplot(111)

	canvas = FigureCanvasTkAgg(fig7, master=spectrumWin)

	toolbar_frame.grid(row=0, column=0, columnspan =2, sticky = W)
	toolbar = NavigationToolbar2TkAgg(canvas, toolbar_frame)
	canvas.get_tk_widget().grid(row=2, column = 0, columnspan = 4)


	plt.imshow(spect_data, origin = 'lower')
	plt.xlabel('Pixel')
	plt.ylabel('Pixel')
	plt.title('CCD')
	fig7.canvas.draw()


	spectrumWin.protocol("WM_DELETE_WINDOW", closeSpectrumWin)
	spectrumWin.mainloop()

	np.set_printoptions(threshold=np.inf)
	

	temp_data = ma.zeros((len(displacement), width, length))
	temp_flat = ma.zeros((len(displacement), width, length))
	flat_gauss = ma.zeros((len(displacement),thick))
	weightArray = np.ndarray(shape = (len(displacement), thick))
	spectrum = ma.zeros((len(displacement), 1, length))
	spectrum_flat = ma.zeros((len(displacement), 1, length))

	g = fitReader(z)
	x_new = np.linspace(0,length,length)
	gauss_step = np.arange(0, thick, 1)


	i = 0
	count = 0
	k = 0

	while (k < len(displacement)):
		count = 0
		displace = displacement[k]
		while (count < thick):
			i = 0
			while (i < length):
				t=g[k]
				t.c[0]=correct[k]*g[k][2]
				t.c[1]=t.c[1]-order_drift[k]
				j = int(t(i))
				pan = int(j+displace+count)
				temp_data[k,pan, i] = spect_data[pan,i]
				temp_flat[k,pan,i]=flat_data[pan,i]
				g[k][2] = g[k][2]/correct[k]
				g[k][1]=g[k][1]+order_drift[k]
				i = i+1
			count = count + 1
		k = k +1


	#temp_data = ma.masked_less(temp_data, 0)
	#temp_flat = ma.masked_less(temp_flat,0)

	flatF_data = temp_data/flatF
	flatF_flat = temp_flat/flatF
	flatF_flat = flatF_flat.filled(0)
	flatF_data = flatF_data.filled(0)

	k = 0
	while (k < len(displacement)):
		count = 0
		displace = displacement[k]
		while (count < thick):
			i = 0
			while (i < length):
				t=g[k]
				t.c[0]=correct[k]*g[k][2]
				t.c[1]=t.c[1]-order_drift[k]
				j = int(t(i))
				pan = int(j+displace+count)
				flat_gauss [k,count] = flat_gauss[k,count]+flatF_flat[k, pan, i]
				g[k][2] = g[k][2]/correct[k]
				g[k][1]=g[k][1]+order_drift[k]
				i = i+1
			count = count + 1
		k = k +1
	

	k = 0
	spec = ma.sum(flatF_data, 1)

	while (k < len(displacement)):
		count = 0
		displace = displacement[k]
		
		mean = ma.sum(flat_gauss[k]*gauss_step)/ma.sum(flat_gauss[k])

		sig = ma.sum(flat_gauss[k]*(gauss_step-mean)**2)/ma.sum(flat_gauss[k])
		
		popt, pcov = scipy.optimize.curve_fit(gaussian,gauss_step, flat_gauss[k], 
												p0=[ma.max(flat_gauss[k]),mean, sig])
		weight = gaussian(gauss_step, *popt)
		
		sumWeight = np.sum(weight)
		while (count < thick):
			i = 0
			while (i < length):
				t=g[k]
				t.c[0]=correct[k]*g[k][2]
				t.c[1]=t.c[1]-order_drift[k]
				j = int(t(i))
				pan = int(j+displace+count)
				adjusted_Weight = gaussAdjust(weight, flatF_data[k,pan,i])
				spectrum[k,0, i] = spectrum[k,0,i]+((flatF_data[k,pan,i])*adjusted_Weight)/sumWeight
				adjusted_Weight = gaussAdjust(weight, flatF_flat[k,pan,i])
				spectrum_flat[k,0, i] = spectrum_flat[k,0,i]+((flatF_flat[k,pan,i])*adjusted_Weight)/sumWeight
				g[k][2] = g[k][2]/correct[k]
				g[k][1]=g[k][1]+order_drift[k]
				i = i+1
			count = count + 1
		weightArray[k] = weight
		k = k +1

	k = 0
	gauss_step_flat = np.arange(overScan_location[0],overScan_location[1],1)
	mean = length_flat/2.0
	sig = ((np.sum((gauss_step_flat-mean)**2))**0.5)/len(gauss_step_flat)

	while (k<len(spectrum_flat)):
		popt, pcov = scipy.optimize.curve_fit(gaussian,gauss_step_flat, np.copy(spectrum_flat[k,0,overScan_location[0]:overScan_location[1]]), 
												p0=[ma.max(spectrum_flat[k,0,overScan_location[0]:overScan_location[1]]),mean, sig])
		normalizeFlat = gaussian(gauss_step_flat, *popt)

		spectrum_flat[k,0,overScan_location[0]:overScan_location[1]]=np.copy(spectrum_flat[k,0,overScan_location[0]:overScan_location[1]])/normalizeFlat
		k=k+1

	
	gaussianWin = Tk()
	gaussianWin.title('SpecTracer')

	toolbar_frame=Frame(gaussianWin)

	fig8 = plt.figure(figsize= (10,5))
	ax8= fig8.add_subplot(111)
	helv20 = tkFont.Font(family ='Helvetica', size = 10, weight = 'bold')

	next_button = Button(gaussianWin, text="Next Order", command = advanceGaussOrder, font = helv20)
	prev_button = Button(gaussianWin, text="Previous Order", command = previousGaussOrder, font = helv20)
	goTo_button = Button(gaussianWin, text = "Go!", command = goGaussTo, font = helv20)
	entry = Entry(gaussianWin)

	next_button.grid(row=4, column = 3, sticky = S)
	prev_button.grid(row=4, column = 0, sticky = W)
	goTo_button.grid(row=4, column = 2, sticky = S)
	entry.grid(row=4, column = 1, sticky = S)
	
	canvas = FigureCanvasTkAgg(fig8, master=gaussianWin)
	toolbar_frame.grid(row=0, column=0, columnspan =2, sticky = W)
	toolbar = NavigationToolbar2TkAgg(canvas, toolbar_frame)
	canvas.get_tk_widget().grid(row=2, column = 0, columnspan = 4)
	

	updatePlotGauss(0)

	order = 0

	gaussianWin.protocol("WM_DELETE_WINDOW", closeGaussWin)
	gaussianWin.mainloop()	
	
	spectrum = ma.copy(spectrum)/ma.copy(spectrum_flat)

	spectrum_flattened = spectrum.flatten()

	spectrum_valid = spectrum.compressed()

	SS = spectrum_valid.flatten()

	xPoints = np.arange(0,length*len(displacement),1)

	XPF = xPoints.flatten()
	XPF_mask = ma.array(XPF, mask = ma.getmaskarray(spectrum))



	x_new = np.linspace(0, length, length)


	xPs = np.linspace(0, len(SS), len(SS))

	flatWin = Tk()
	flatWin.title('SpecTracer')

	toolbar_frame=Frame(flatWin)

	fig9 = plt.figure(figsize= (10,5))
	ax9 = fig9.add_subplot(111)

	canvas = FigureCanvasTkAgg(fig9, master=flatWin)
	toolbar_frame.grid(row=0, column=0, columnspan =2, sticky = W)
	toolbar = NavigationToolbar2TkAgg(canvas, toolbar_frame)
	canvas.get_tk_widget().grid(row=2, column = 0, columnspan = 4)
	

	plt.imshow(flatF_data[len(flatF_data)/2],origin='lower')
	plt.title ('Flat Fielded Data at order 15')
	plt.xlabel('Pixel')
	plt.ylabel('Pixel')
	fig9.canvas.draw()


	flatWin.protocol("WM_DELETE_WINDOW", closeFlatWin)
	flatWin.mainloop()

	spectrumNormalized = np.copy(spectrum[:,0])
	isCleared = np.zeros((len(displacement)), dtype=bool)
	jump_Array = np.ones((len(displacement)),dtype=int)*10
	normalizationOrder_Array = np.ones((len(displacement)))*2
	x_Dimension = np.arange(0,length,1)
	normalization_Array = np.ndarray(shape = (len(displacement)),dtype=object)
	isNormalized = np.zeros(len(displacement), dtype=bool)

	maskedSegmentX = [[]]
	maskedSegmentY = [[]]
	coloredSegmentX = [[]]
	coloredSegmentY = [[]]

	i=0

	for i in range(len(displacement)-1):
		maskedSegmentX.extend([[]])
		maskedSegmentY.extend([[]])
		coloredSegmentX.extend([[]])
		coloredSegmentY.extend([[]])

	if (not choice):
		normalizedFileExists = checkStarNormalization()
		spectrum.mask=False
		

		normalizeSpectWin = Tk()
		normalizeSpectWin.title('SpecTracer')

		toolbar_frame=Frame(normalizeSpectWin)
		toolbar_frame_dwn=Frame(normalizeSpectWin)

		figFittedSpectrum = plt.figure(figsize = (6,3))
		figNormalizedSpectrum = plt.figure(figsize = (6,3))
		axFittedSpectrum= figFittedSpectrum.add_subplot(111)
		axNormalizedSpectrum = figNormalizedSpectrum.add_subplot(111)

		next_button = Button(normalizeSpectWin, text="Next Order", command = advanceNormalized, font = helv20)
		prev_button = Button(normalizeSpectWin, text="Previous Order", command = previousNormalized, font = helv20)
		goTo_button = Button(normalizeSpectWin, text = "Go!", command = goNormalizedTo, font = helv20)
		update_button = Button(normalizeSpectWin, text = "Update", command = updateDegree, font = helv20)
		clear_button =Button(normalizeSpectWin, text="Clear", command=clearRemovedPoints, font=helv20)
		entry = Entry(normalizeSpectWin)
		entry_nrm=Entry(normalizeSpectWin)
		entry_slice = Entry(normalizeSpectWin)
		label_nrm = Label(normalizeSpectWin, text="Order of Fit", font = helv20)
		label_slice=Label(normalizeSpectWin, text="Slice Size", font=helv20)


		canvas_FittedSpectrum = FigureCanvasTkAgg(figFittedSpectrum, master = normalizeSpectWin)
		canvas_NormalizedSpectrum = FigureCanvasTkAgg(figNormalizedSpectrum, master = normalizeSpectWin)

		toolbar_frame.grid(row=0, column=0, columnspan =4, sticky = W)
		toolbar_frame_dwn.grid(row=5, column=0, columnspan =4, sticky = W)

		toolbar_FittedSpectrum = NavigationToolbar2TkAgg(canvas_FittedSpectrum, toolbar_frame)
		toolbar_NormalizedSpectrum = NavigationToolbar2TkAgg(canvas_NormalizedSpectrum, toolbar_frame_dwn)

		next_button.grid(row=7, column = 3, sticky = S)
		prev_button.grid(row=7, column = 0, sticky = W)
		goTo_button.grid(row=7, column = 2, sticky = S)
		clear_button.grid(row=1,column =5, sticky=W, columnspan=3, rowspan=2,padx=3 )
		entry.grid(row=7, column = 1, sticky = S)
		entry_nrm.grid(row=2, column = 6, sticky=W,columnspan=2,padx=3)
		label_nrm.grid(row=2, column = 4, sticky=W)
		entry_slice.grid(row=3,column = 6, sticky = W,columnspan=2,padx=3)
		label_slice.grid(row=3, column = 4, sticky = W)
		update_button.grid(row=4, column = 5, sticky = N, columnspan=3, pady=3)
		
		entry_nrm.insert(0, "2")
		entry_slice.insert(0,"10")

		canvas_FittedSpectrum.get_tk_widget().grid(row=1, column = 0, columnspan=4,
													rowspan=4)
		canvas_NormalizedSpectrum.get_tk_widget().grid(row =6, column = 0, 
													columnspan = 4,pady=2)
		

		updateNormalizePlot(0)

		normalizeSpectWin.protocol("WM_DELETE_WINDOW", closeNormalizeWin)
		normalizeSpectWin.mainloop()
		saveAuxiliaryFiles()


	saveSpectrum(spectrumNormalized)

def saveAuxiliaryFiles():
	global normalization_Array, coloredSegmentY, coloredSegmentX, maskedSegmentX, maskedSegmentY, jump_Array, normalizationOrder_Array, isNormalized

	clrd_Y=np.array(coloredSegmentY)
	clrd_X=np.array(coloredSegmentX)
	mskd_X=np.array(maskedSegmentX)
	mskd_Y=np.array(maskedSegmentY)

	saveToNormalizeFile('colored_X',clrd_X)
	saveToNormalizeFile('colored_Y',clrd_Y)
	saveToNormalizeFile('masked_X',mskd_X)
	saveToNormalizeFile('masked_Y',mskd_Y)
	saveToNormalizeFile('slice', jump_Array)
	saveToNormalizeFile('normalization_order', normalizationOrder_Array)
	saveToNormalizeFile('normalization', normalization_Array)
	saveToNormalizeFile('normalized_bool', isNormalized)

def readAuxiliaryFiles():
	global normalization_Array, coloredSegmentY, coloredSegmentX, maskedSegmentX, maskedSegmentY, jump_Array, normalizationOrder_Array, isNormalized
	clrd_X=openNormalizationFile('colored_X.npy')
	clrd_Y=openNormalizationFile('colored_Y.npy')
	mskd_X=openNormalizationFile('masked_X.npy')
	mskd_Y=openNormalizationFile('masked_Y.npy')
	jump_Array=openNormalizationFile('slice.npy')
	normalizationOrder_Array=openNormalizationFile('normalization_order.npy')
	normalization_Array=openNormalizationFile('normalization.npy')
	isNormalized = openNormalizationFile('normalized_bool.npy')

	coloredSegmentX = clrd_X.tolist()
	coloredSegmentY=clrd_Y.tolist()
	maskedSegmentX=mskd_X.tolist()
	maskedSegmentY=mskd_Y.tolist()
	

def checkStarNormalization():
	message="Has this star been normalized before?"
	isPreviouslyNormalized = dialog(message, 'q')

	if(isPreviouslyNormalized):
		message = "Please select the folder related to the Star"
		folderCreation(False,False,False,False,True, message)
		readAuxiliaryFiles()
	else:
		message = "Please create the folder for the normalization of this Star"
		folderCreation(False,False,False,False,True, message)

	return isPreviouslyNormalized

def closeGraph ():
	choice = tkMessageBox.askyesno("SpecTracer", "Are you sure you want to quit?")
	if (choice):
		graph.quit()
		graph.destroy()

def spectrumGraph ():
	global canvas, spectrum_In, order, entry, graph, fig11, ax11, g_length, overScan_location

	overScan_location=openNumpyFile('overscanloc.npy')

	instruction = "Please type the name of a file containing the image \ncaptured by the CCD (Flat Lamp)"
	nameFile_Instrument = getFileName(instruction)
	isLandscape, g_length, g_width = getDimensions (nameFile_Instrument)

	instruction = "Please type the name of the file, containing the reduced spectrum"
	spectrum_In = readSpectrumFits(instruction)

	message = "Please select the folder with the Calibrations."
	folderCreation(False, False, False, True, False, message)

	
	graph = Tk()
	toolbar_frame= Frame(graph)


	graph.title ("SpecTracer")
	fig11 = plt.figure(figsize= (10,5))
	ax11 = fig11.add_subplot(111)

	helv20 = tkFont.Font(family ='Helvetica', size = 10, weight = 'bold')

	next_button = Button(graph, text="Next Order", command = advanceOrder, font = helv20)
	prev_button = Button(graph, text="Previous Order", command = previousOrder, font = helv20)
	goTo_button = Button(graph, text = "Go!", command = goTo, font = helv20)
	entry = Entry(graph)

	next_button.grid(row=4, column = 3, sticky = S)
	prev_button.grid(row=4, column = 0, sticky = W)
	goTo_button.grid(row=4, column = 2, sticky = S)
	entry.grid(row=4, column = 1, sticky = S)
	canvas = FigureCanvasTkAgg(fig11, master=graph)
	toolbar_frame.grid(row=0, column=0, columnspan =2, sticky = W)

	toolbar = NavigationToolbar2TkAgg(canvas, toolbar_frame)
	canvas.get_tk_widget().grid(row=2, column = 0, columnspan = 4)
	


	updatePlotShow(0)

	order = 0

	graph.protocol("WM_DELETE_WINDOW", closeGraph) 	
	graph.mainloop()

def wavelengthFunctionGen():
	global canvas, spectrum_In, spectrum_II, order, orderLow, isLine
	global entry_up, entry_dwn, entry_rt, calibrationWindow, fig13, fig14, fig15, ax13,ax14,ax15, calibrationMaster
	global crossRelatedScatter, isLandscape, g_length, g_width, date

	instruction = "Please type the name of a file containing the image \ncaptured by the CCD (Flat Lamp)"
	nameFile_Instrument = getFileName(instruction)
	isLandscape, g_length, g_width = getDimensions (nameFile_Instrument)
	
	instruction = "Select the file of the reduced spectrum with known wavelengths"
	spectrum_In = readSpectrumFits(instruction)
	
	instruction = "Select the file with the reduced spectrum of the calibration lamp\nFor this specific measurement"
	spectrum_II = readSpectrumFits(instruction)

	instruction = "Select the master file with: wavelengths, coordinates, and order"
	master_Calibration_File = getFileName(instruction)
	calibrationMaster = calibrationFileRead(master_Calibration_File)
	

	instruction = "Please select the folder (or create one) with the date \n on which these measurements were taken. Format: 'DD-MM-YYYY"
	folderCreation(False, False, False, True, False, instruction)

	crossRelatedScatter = []

	calibrationWindow = Tk()

	calibrationWindow.title ("SpecTracer")
	fig13 = plt.figure(figsize= (6,3))
	ax13 = fig13.add_subplot(111)

	fig14 = plt.figure (figsize =(6,3))
	ax14 = fig14.add_subplot (111)

	fig15 = plt.figure(figsize = (6,3))
	ax15 = fig15.add_subplot (111)

	toolbar_frame=Frame(calibrationWindow)
	toolbar_frame_dwn =Frame(calibrationWindow)
	toolbar_frame_rt=Frame(calibrationWindow)

	
	helv20 = tkFont.Font(family ='Helvetica', size = 10, weight = 'bold')

	next_button_up = Button(calibrationWindow, text="Next Order", 
					command = advanceUpOrder, font = helv20)
	prev_button_up = Button(calibrationWindow, text="Previous Order", 
					command = previousUpOrder, font = helv20)
	goTo_button_up = Button(calibrationWindow, text = "Go!", 
					command = goUpTo, font = helv20)
	entry_up = Entry(calibrationWindow)

	next_button_dwn = Button(calibrationWindow, text="Next Order", 
					command = advanceDwnOrder, font = helv20)
	prev_button_dwn = Button(calibrationWindow, text="Previous Order", 
					command = previousDwnOrder, font = helv20)
	goTo_button_dwn = Button(calibrationWindow, text = "Go!", 
					command = goDwnTo, font = helv20)
	entry_dwn = Entry(calibrationWindow)

	regression_button = Button(calibrationWindow, text="Regression", 
					command = regression, font = helv20)
	save_button = Button(calibrationWindow, text="Save", 
					command = saveRegression, font = helv20)
	degree_button = Button(calibrationWindow, text = "Go!", 
					command = regressionSpecific, font = helv20)
	entry_rt = Entry(calibrationWindow)
	close = Button(calibrationWindow, text = "Close",
					command = closeCalibWind, font = helv20)

	next_button_up.grid(row=4, column = 3, sticky = S)
	prev_button_up.grid(row=4, column = 0, sticky = W)
	goTo_button_up.grid(row=4, column = 2, sticky = S)
	entry_up.grid(row=4, column = 1, sticky = S)
	canvas_up = FigureCanvasTkAgg(fig13, master=calibrationWindow)

	next_button_dwn.grid(row=7, column = 3, sticky = S)
	prev_button_dwn.grid(row=7, column = 0, sticky = W)
	goTo_button_dwn.grid(row=7, column = 2, sticky = S)
	entry_dwn.grid(row=7, column = 1, sticky = S)
	canvas_dwn = FigureCanvasTkAgg(fig14, master=calibrationWindow)

	regression_button.grid(row=4, column = 5, rowspan = 5 ,sticky = W)
	save_button.grid(row=4, column = 8, rowspan = 5, sticky = W)
	degree_button.grid(row=4, column = 7, rowspan = 5, sticky = W)
	entry_rt.grid(row=4, column = 6, rowspan = 5, sticky = W)
	canvas_rt= FigureCanvasTkAgg(fig15, master = calibrationWindow)

	toolbar_frame.grid(row=0, column=0, columnspan =2, sticky = W)
	toolbar_frame_dwn.grid(row=5, column=0, columnspan =4, sticky = W)
	toolbar_frame_rt.grid(row=0, column=5, columnspan =4, rowspan=5, sticky = W)
	toolbar = NavigationToolbar2TkAgg(canvas_up, toolbar_frame)
	toolbar_dwn = NavigationToolbar2TkAgg(canvas_dwn, toolbar_frame_dwn)
	toolbar_rt = NavigationToolbar2TkAgg(canvas_rt, toolbar_frame_rt)
	canvas_up.get_tk_widget().grid(row=2, column = 0, columnspan = 4)
	canvas_dwn.get_tk_widget().grid(row=6, column = 0, columnspan = 4)
	canvas_rt.get_tk_widget().grid(row=2, column = 5, columnspan = 4, rowspan =5)
	
	updatePlot (0)
	updateLowPlot(0)
	fig15.canvas.draw()

	order = 0
	orderLow = 0
	isLine = False

	calibrationWindow.protocol("WM_DELETE_WINDOW", closeCalibWind) 	 	
	calibrationWindow.mainloop()

#Order Operations
def orderExists ():
	try:
		validate = np.load(primary_dir+divider+ 'displacement.npy')
	except (IOError):
		exists = False
	else:
		exists = True

	return exists

def orderAppend ():
	global g_data, g_width, isLandscape, g_length, g_orderLoc, g_correction, g_drift, appendWin
	global fig16, g_multiplier, f, v_pan, h_pan, cid, ax1, z, top, g_displacement, g_slide, canvas, f ,z
	
	instruction = " Please type the name of the file from where the orders\nw ill be read. (Flat Lamp)"
	nameFile_orderCalib = getFileName(instruction)

	g_displacement = openNumpyFile('displacement.npy')
	z = openNumpyFile('fit.npy')
	f=fitReader(z)
	g_correction = openNumpyFile('correction.npy')
	g_slide = openNumpyFile('drift.npy')

	data = fits.getdata(nameFile_orderCalib)
	g_data = []
	g_orderLoc = []
	placeHold = np.zeros(len(g_displacement))

	isLandscape, g_length, g_width= getDimensions (nameFile_orderCalib)
	g_data = forceLandscape(data, isLandscape)

	appendWin = Tk()
	appendWin.title('Spectracer')

	fig16 = plt.figure(figsize= (10,5))
	ax16 = fig16.add_subplot(111)

	toolbar_frame=Frame(appendWin)
	done_button = Button(appendWin, text = 'Done', command = doneAppend)
	canvas = FigureCanvasTkAgg(fig16, master=appendWin)

	done_button.grid(row=3, column = 0,columnspan = 10, sticky = S)
	toolbar_frame.grid(row=0, column=0, columnspan =2, sticky = W)
	toolbar = NavigationToolbar2TkAgg(canvas, toolbar_frame)
	canvas.get_tk_widget().grid(row=2, column = 0, columnspan = 4)
	cid = canvas.mpl_connect('button_press_event', orderLocator)
	selectedOrders = np.ones(len(f))




	ax16.axis([0,g_length,0,g_width])
	ax16.set_aspect('auto')
	ax16.scatter(placeHold, g_displacement,100,'r' ,marker = 'o')
	plt.imshow(g_data)
	fig16.canvas.draw()
	message ="Please double click on the order location. \nIt is recommended to leave a 2 px padding. When finished, double right click in the plot \nOr click on the done button"
	dialog(message, 'i')
	

	appendWin.protocol("WM_DELETE_WINDOW", closeAppendWin)
	appendWin.mainloop()

	

	g_orderLoc = np.array(g_orderLoc)
	top = len(np.append(g_orderLoc, g_displacement))

	g_multiplier = np.ones(len(g_orderLoc))
	g_drift = np.zeros(len(g_orderLoc))
	orderCorrections()

	correction = np.append (g_correction, g_multiplier)
	orderLoc = g_orderLoc+v_pan
	displacement = np.append(g_displacement+selectedOrders, orderLoc+selectedOrders[-1])
	slide = np.append(g_slide, g_drift)
	ind = np.lexsort((correction, slide, displacement))
	sortedNew = np.array([(displacement[i],correction[i], slide[i]) for i in ind])
	displacement = sortedNew[:,0]
	correction = sortedNew[:,1]
	slide = sortedNew[:,2]
	
	saveToFile('displacement',displacement)
	saveToFile('correction', correction)
	saveToFile('drift', slide)
	saveToFile('fit', z)


	plt.ioff()
	plt.close()

def orderCorrections ():
	global f, v_pan,fig17,ax17, h_pan, g_data, correctWin
	f=fitReader(z[:,-len(g_orderLoc):])
	v_pan = np.zeros(len(g_orderLoc))
	h_pan = np.zeros (len(g_orderLoc))
	x_new = np.linspace(0,g_length,g_length)

	correctWin = Tk()
	correctWin.title('SpecTracer')
	fig17 = plt.figure(figsize= (20,10))
	ax17 = fig17.add_subplot(111)

	canvas = FigureCanvasTkAgg(fig17, master=correctWin)
	toolbar_frame=Frame(correctWin)

	toolbar_frame.grid(row=0, column=0, columnspan =2, sticky = W)
	toolbar = NavigationToolbar2TkAgg(canvas, toolbar_frame)
	canvas.get_tk_widget().grid(row=2, column = 0, columnspan = 4)
	

	ax17.axis([0,g_length,0,g_width])
	ax17.set_aspect('auto')
	ax17.set_xlabel('Pixel')
	ax17.set_ylabel('Pixel')
	ax17.set_title ('CCD Image')

	fitGrapher(x_new, ax17, fig17)
				
	message = "	Some of the orders may not be traced very accurately (specially at the edges)\nIf the bottom order is order 0, please type the number of the order you would\nlike to modify to create a better fit.\nIf all the orders need changing type a " + str(top) + "\nIf no order needs to be changed anymore please type a -1"

	choice = dialog(message, 'e')
	while (choice >=0):
		if (choice == top):
			message = "Type a positive number to pan the graph upward or\nnegative to pan the graph downward"
			pan =dialog(message, 'e')
			v_pan[:] = v_pan[:]+pan
			
			message = "Type a positive number to pan the graph right or\nnegative to pan the graph left"
			dan = dialog(message, 'f')
			g_drift[:] = h_pan[:]+dan

			fitGrapher(x_new, ax17, fig17)

		else:
			message = "Type a number between 0 (not inclusive) and 9999\nRemember that if < 1, the fit will widen and \nif >1 the fit will get more narrow"
			correction = dialog(message, 'f')
			
			message = "Type a positive number to pan the graph upward or\nnegative to pan the graph downward"
			pan =dialog(message, 'e')
			

			message = "Type a positive number to pan the graph right or\nnegative to pan the graph left"
			dan = dialog(message, 'f')
			

			g_multiplier[choice] = g_multiplier[choice]*correction
			v_pan[choice]=v_pan[choice]+pan
			g_drift[choice] = g_drift[choice]+dan

		ax17.cla()
		fitGrapher (x_new, ax17, fig17)
		
		message = "Please type the number of the order you would like to modify to create a better fit.\nIf no order needs to be changed anymore please type a -1\nIf all the orders need changing (pan) type a " +str(top)
		choice = dialog(message, 'e')

	correctWin.protocol("WM_DELETE_WINDOW", closeCorrectWin)
	correctWin.mainloop()

def orderModify ():
	global g_data, g_orderLoc, g_correction, g_multiplier, v_pan, z, g_correction
	global g_length, g_width, isLandscape,f, g_displacement, top, g_drift, g_slide
	global isFiles

	isFiles = False
	
	instruction = " Please type the name of the file from where the orders\nwill be read. (Flat Lamp)"
	nameFile_orderCalib = getFileName(instruction)

	data = fits.getdata(nameFile_orderCalib)
	isLandscape, g_length, g_width= getDimensions (nameFile_orderCalib)
	g_data = forceLandscape(data, isLandscape)
	z = openNumpyFile ('fit.npy')
	g_correction = openNumpyFile('correction.npy')
	g_displacement = openNumpyFile('displacement.npy')
	g_slide = openNumpyFile('drift.npy')

	g_orderLoc = g_displacement
	g_multiplier=g_correction
	g_drift=g_slide

	top = len(g_orderLoc)
	orderCorrections()

	correction = g_multiplier
	orderLoc = g_orderLoc+v_pan
	displacement = orderLoc
	slide = g_drift
	ind = np.lexsort((correction, slide, displacement))
	sortedNew = np.array([(displacement[i],correction[i], slide[i]) for i in ind])
	
	displacement = sortedNew[:,0]
	correction = sortedNew[:,1]
	slide = sortedNew[:,2]

	saveToFile('displacement',displacement)
	saveToFile('correction', correction)
	saveToFile('drift', slide)

def orderActions ():
	global root

	if (orderExists()):
		root = Tk()
		root.title('SpecTracer')
		helv20 = tkFont.Font(family ='Helvetica', size = 10, weight = 'bold')

		primary_label = Label(root, text = "Please select an option", font = helv20)
		append_button = Button(root, text = "Append Order", 
				command = orderAppend, font = helv20)
		modify_button = Button(root, text = "Modify Orders", 
				command = orderModify, font = helv20)
		exit_button = Button(root, text = "Exit", 
				command = close, font = helv20)

		primary_label.grid(row =0, column = 0, columnspan = 3, sticky = N)
		append_button.grid(row=1, column = 0, columnspan = 3, 
				sticky = N, padx = 5, pady=5)
		modify_button.grid(row=2, column = 0, columnspan = 3,
				sticky = N, padx = 5, pady=5)
		exit_button.grid(row=3, column = 0, columnspan = 3,
				sticky = N, padx = 5, pady = 5)

		root.mainloop()

	else:
		orderTracer()

def welcome ():
	welcomeDialog = Tk()
	welcomeDialog.title('SpecTracer')
	message = 'Welcome to SpecTracer'
	dialog(message, 'i')
	welcomeDialog.quit()
	welcomeDialog.destroy()
	welcomeDialog.mainloop()

def main ():
	if __name__ == "__main__": 
		
		getOS()
		global isFiles, primary_dir, master
		isFiles = False 

		welcome()
		
		message = "Select the folder corresponding to the telescope used"

		folderCreation (True, False, False,False,False, message)

		
		isFiles = False
		master = Tk()
		master.title ("SpecTracer")
		helv20 = tkFont.Font(family ='Helvetica', size = 10, weight = 'bold')

		order_button = Button(master, text="Order Modify", 
				command = orderActions, font = helv20)
		noise_button = Button(master, text="Noise Calibration", 
				command = biasDarkCalibration, font = helv20)
		sensitivity_button = Button(master, text = "Sensitivity Calibration", 
				command = sensitivityCalibration, font = helv20)
		reduce_button = Button(master, text = "Reduce Spectrum", 
				command = specTracer, font = helv20)
		wavelength_button = Button(master, text = "Wavelength Calibration", 
				command = wavelengthFunctionGen, font = helv20)
		spectrum_button = Button(master, text = "Show Spectrum", 
				command = spectrumGraph, font = helv20)
		exit_button = Button(master, text = "Exit", 
				command = closeMaster, font = helv20)

		order_button.grid(row=2, column = 0, columnspan = 3, 
				padx = 20, pady=5, sticky = N+S)
		noise_button.grid(row=1, column = 0, columnspan = 3, 
				padx = 20, pady=5, sticky = N+S)
		sensitivity_button.grid(row=3, column = 0, columnspan = 3, 
				padx = 20, pady=5, sticky = N+S)
		reduce_button.grid(row=4, column = 0, columnspan = 3, 
				padx = 20, pady=5, sticky = N+S)
		wavelength_button.grid(row=5, column = 0, columnspan = 3, 
				padx = 20, pady=5, sticky = N+S)
		spectrum_button.grid(row=6, column = 0, columnspan = 3, 
				padx = 20, pady=5, sticky = N+S)
		exit_button.grid(row=7, column = 0, columnspan = 3, 
				padx = 20, pady=5, sticky = N+S)
	
		master.mainloop()
		plt.ioff()

main ()
