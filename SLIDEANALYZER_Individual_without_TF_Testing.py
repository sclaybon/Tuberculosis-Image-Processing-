import cv2
import numpy as np
import time
from skimage.feature import hessian_matrix_eigvals
from matplotlib import pyplot as plt
from scipy.ndimage.filters import convolve
import scipy
import scipy.misc
from scipy import ndimage
from imutils import perspective 
from imutils import contours
import numpy as np
import imutils
import cv2
import numpy as np
from scipy.spatial import distance as dist
import scipy.misc
import os
from tempfile import TemporaryFile
import re
from natsort import natsort
import pandas as pd

#midpoint function
def midpoint(ptA,ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
	
#read in the data from the bacilli location groundtruth ['P/N','Name',Y,X]
bacilliLocations = np.load("bacilliLocations.npy")

#initialize iteration number then read and sort files
k = 1
files = os.listdir()
files = natsort(files)

#initialize variables for precision calculation
totalNumberOfTruePositives = 0
totalNumberOfFalsePositives = 0
groundTruth = 0

#initialize empty array for excel outputs
accuracyFile = np.empty((0,4))




##############################################################

#################### BEGIN READING IMAGE #####################

##############################################################





#INPUT TEST IMAGE HERE
fn = "P (2).tif"
image = cv2.imread(fn)

imageActual = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

#real image used to draw circles and put text on later on
real = image.copy()

#convert the image to a 5MP image instead of whatever it is to make the program work properly
[height,width,depth] = image.shape
oldHeight = height
oldWidth = width
newHeight = int(height/1)
newWidth = int(width/1)
image = scipy.misc.imresize(image,(newHeight,newWidth,depth))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image = cv2.bilateralFilter(image,11,17,17)

print("bilateral filter done")

######## Begin Vessel Filtering ##########

#data type conversion to allow for numbers larger than 255 in the convolution step
image = image.astype(np.float64)

#create a 19x19 Prewitt gradient filter
sigma = 1
x = np.linspace(-3*np.round(sigma),3*np.round(sigma),(3*sigma*2+1))
xv,yv = np.meshgrid(x,x)

#use xv and yv as the prewitt derivative kernels for the gaussian kernel base
DGaussxx = 1/(2*np.pi*np.power(sigma,4)) * (np.square(xv)/np.square(sigma)-1) * np.exp(-(np.square(xv)+np.square(yv))/(2*np.square(sigma)))
DGaussxy = 1/(2*np.pi*np.power(sigma,6)) * (xv*yv) *  np.exp(-(np.square(xv)+np.square(yv))/(2*np.square(sigma)))
DGaussyy = np.transpose(DGaussxx)

xCoordinate = -3
yCoordinate = -3
DGaussx = 1/(2*np.pi*np.power(sigma,4)) * (np.square(xCoordinate)/np.square(sigma)-1) * np.exp(-(np.square(xCoordinate)+np.square(yCoordinate))/(2*np.square(sigma)))

#create second derivative image matrices / convolution with zeros padding the outer edges
Dxx = convolve(image,DGaussxx, mode='constant', cval=0.0)
Dxy = convolve(image,DGaussxy, mode='constant', cval=0.0)
Dyy = convolve(image,DGaussyy, mode='constant', cval=0.0)

#print(np.shape(np.array(Dxx)))
Dxx = (sigma ** 2)*Dxx
Dxy = (sigma ** 2)*Dxy
Dyy = (sigma ** 2)*Dyy

#plots of various second derivative image outputs
'''
plt.imshow(Dxx,interpolation = "none")
plt.colorbar()
plt.show()

plt.imshow(Dyy,interpolation = "none")
plt.colorbar()
plt.show()


plt.imshow(Dxy,interpolation = "none")
plt.colorbar()
plt.show()



output = Dxx
dFactor = np.max(np.abs(output))/255
output = output/dFactor
output = np.abs(output)
output = output.astype('uint8')
cv2.imwrite("Dxx.jpg",output)

		
output = Dyy
dFactor = np.max(np.abs(output))/255
output = output/dFactor
output = np.abs(output)
output = output.astype('uint8')
cv2.imwrite("Dyy.jpg",output)

		
output = Dxy
dFactor = np.max(np.abs(output))/255
output = output/dFactor
output = np.abs(output)
output = output.astype('uint8')
cv2.imwrite("Dxy.jpg",output)
		
print("done")
'''

#Calculate (abs sorted) eigenvalues and vectors
hessEigvals = hessian_matrix_eigvals(Dxx, Dxy, Dyy)
hessEigvals = np.array(hessEigvals)		

depth,heightY,widthX = hessEigvals.shape
hessEigvals = np.array([np.ravel(hessEigvals[0]),np.ravel(hessEigvals[1])])
hessEigvals = np.transpose(hessEigvals)
ind = np.argsort(np.abs(hessEigvals), axis=1)
height = hessEigvals.shape[0]
length = np.arange(0,height)[:,np.newaxis]
hessEigvals = hessEigvals[length,ind[0:height]]

#print(hessEigvals[:,0].shape)
lambda2 = np.reshape(hessEigvals[:,0],(heightY,widthX))
lambda1 = np.reshape(hessEigvals[:,1],(heightY,widthX))

#create beta one and beta two values for use in the 2D vesselnes functio
beta1 = 0.5
beta2 = 15
beta1 = 2 * beta1 ** 2
beta2 = 2 * beta2 ** 2

#Compute some similarity measures
lambda1[lambda1 == 0] = 1e-10
rb = (lambda2 / lambda1) ** 2
s2 = lambda1 ** 2 + lambda2 ** 2

#Compute the output image
filtered = np.exp(-rb / beta1) * (np.ones(np.shape(image)) - np.exp(-s2 / beta2))
filtered[lambda1 > 0] = 0

###### Convert Image to grayscale##########		
filtered = filtered*255
image_orig = filtered.astype(np.uint8)
		
#Convert image to Binary
thresh,binaryImage = cv2.threshold(image_orig,0,255,cv2.THRESH_BINARY)

print("starting contours")

#Remove outer border to avoid detection by vessel filter due to camera distortion
height,width = binaryImage.shape
binaryImage[:,0:20] = 0
binaryImage[:,width-20:width] = 0
binaryImage[0:20,:] = 0
binaryImage[height-20:height,:] = 0

#Find contours
cnts = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

#if no contours are found, exit the loop
if len(cnts) != 0:

	#sort the contours from left-to-right and initialize the 
	# 'pixels per metric' calibration variable
	(cnts, _) = contours.sort_contours(cnts)

	#convert pixel dimensions into actual size dimensions in meters
	pixelsPerMetric = 15
	
	#initialize counts for testing against bacilli locations
	cluster = 0
	clusterCount = 0
	num = 0
	truePositives = 0
			
	#loop through contours
	for c in cnts:
		
		#if the contour is not sufficiently large, ignore it
		if cv2.contourArea(c) < 1:
			continue
		
		#compute the rotated bounding box of the contour
		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box,dtype="int")

		#order the poiints in the contour such that they appear 
		#in top-left, top-right, bottom-right, and bottom-left order
		box = perspective.order_points(box)
		
		#draw bounding box if needed
		#cv2.drawContours(real,[box.astype("int")], -1,(0,255,0),1)
		
		#unpack the ordered bounding box, then compute the midpoint
		#between the top-left and top-right coordinates, followed by
		#the midpoint between bottom-left and bottom-right coordinates
		(tl,tr,br,bl) = box
		(tltrX,tltrY) = midpoint(tl,tr)
		(blbrX,blbrY) = midpoint(bl,br)
		
		#compute the midpoint between the top-left and top-right points,
		#followed by the midpoint between the top-right and bottom-right
		(tlblX,tlblY) = midpoint(tl,bl)
		(trbrX,trbrY) = midpoint(tr,br)
						
		#compute the Euclidean distance between the midpoints
		dA = dist.euclidean((tltrX,tltrY), (blbrX,blbrY))
		dB = dist.euclidean((tlblX,tlblY), (trbrX,trbrY))
						
		#compute the size of the object
		dimA = dA / pixelsPerMetric
		dimB = dB / pixelsPerMetric
		
		#parameters for minimum enclosing circle
		(x,y), radius = cv2.minEnclosingCircle(c)
		center = (int(x),int(y))
		radius = int(radius)
			
		#dimensions of bacilli			
		if dimA < .5 and dimB > 2.75*dimA and dimB < 6.75*dimA and dimB > .8:	
			
			#get a crop of the potential bacilli
			squareCrop = imageActual[int(y-12):int(y+12),int(x-12):int(x+12)]

			#foreground confirmation means
			totalCropMean = np.mean(squareCrop)
			centerMean = np.mean(imageActual[int(y-3):int(y+3),int(x-3):int(x+3)])

			if ( (centerMean - totalCropMean) > (centerMean/2)):					
			
				#Used to extract the bacillus for explanation of the algorithm purposes!
				#cv2.imwrite("bacillus.jpg", imageActual[int(y-3):int(y+3),int(x-3):int(x+3)].astype('uint8'))
				
				#Get the square patch of pixels around the center of each potential bacillus
				#plt.imshow(imageActual[int(y-3):int(y+3),int(x-3):int(x+3)].astype('uint8'),cmap='gray')
				#plt.colorbar()
				#plt.show()
										
				#add one for each bacilli found
				num = num+1
				
				#circle bacilli with a green color
				cv2.circle(real,center,30,(255,100,0),8)
				
				#dimension text for each bacilli
				#cv2.putText(real, "{:.1f} width".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
				#cv2.putText(real, "{:.1f} height".format(dimB), (int(trbrX + 30), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)

				#get the center locations of the bacilli that I find
				#centerPoint = np.array([int(y),int(x)])
		else:
			
			#IN THE FUTURE: check circularity for clusters
			perimeter =  cv2.arcLength(c,True)
			area = cv2.contourArea(c)
			circularity = (perimeter*perimeter) / (4*np.pi*area)
			
#save computer circled image
name = fn+'_AI.jpg'
cv2.imwrite(name,real)	
			
#output data
print("      ")
print(name)
print("single Bacilli = ",num)

