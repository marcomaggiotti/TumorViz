import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.signal
import argparse
from PIL import Image, ImageDraw
from PIL import ImagePath
from matplotlib.collections import PolyCollection
import matplotlib.patches as pat
import sklearn.feature_extraction.image
import matplotlib.cm
import torch
from skimage.external.tifffile import imread
from torchvision.models import DenseNet
from tqdm.autonotebook import tqdm
from  skimage.color import rgb2gray
from skimage import measure
import skimage.color as color
import skimage
import openslide
import PIL.Image

size = 896, 385

fname=r"/home/sudopizzai/Documents/data/brca1/mib1/BRCA03_MIB1.mrxs"
openSlideReader  = openslide.OpenSlide(fname)

#add mask creation which skips parts of image
mask_level = 4
img_ground_Openslide = openSlideReader.read_region((0, 0), mask_level, openSlideReader.level_dimensions[mask_level])
img_ground_Openslide_array = np.asarray(img_ground_Openslide)[:, :, 0:3]
plt.imshow(img_ground_Openslide)
plt.show()
imgRegions = imread('/home/sudopizzai/Documents/Lab/deepLearning/brac/images/03.tif', key = 0)
hsv=cv2.cvtColor(imgRegions,cv2.COLOR_BGR2YCrCb)

darkGreen_lo=np.array([60,60,60])
darkGreen_hi=np.array([180,180,180])

mask=cv2.inRange(hsv,darkGreen_lo,darkGreen_hi)
imgRegions[mask>0]=(0,0,0)
im = Image.fromarray(imgRegions)
im.save("./orig_3.jpg")
imgg = rgb2gray(imgRegions)
im = Image.fromarray(imgg)
contours = measure.find_contours(im, 0.1)

img_ground_Openslide_array = np.asanyarray(img_ground_Openslide_array)
im1 = Image.fromarray(img_ground_Openslide_array)
im1.save("./tumor_3.jpg")

def drawShape(img, coordinates, color):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgage = skimage.color.gray2rgb(gray)
    # Make sure the coordinates are expressed as integers
    coordinates = coordinates.astype(int)
    imgage[coordinates[:, 0], coordinates[:, 1]] = color
    return imgage

for contour in contours:
    imgFinal = drawShape(imgRegions, contour, [255, 255, 0])
# Grayscale

gray = cv2.cvtColor(imgRegions, cv2.COLOR_BGR2GRAY)

# Find Canny edges
edged = cv2.Canny(gray, 30, 200)
print(gray.shape)

cv2.waitKey(0)
contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print('Original Dimensions : ',edged.shape)

scale_percent = 40 # percent of original size
width = int(edged.shape[1] * scale_percent / 100)
height = int(edged.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(edged, dim, interpolation = cv2.INTER_AREA)

contoursGroundArray = cv2.drawContours(img_ground_Openslide_array, contours, -1, (0, 255, 0), 3)
print(contoursGroundArray)

new = cv2.imread('/home/sudopizzai/Documents/data/images/tumor_3.jpg')
new = cv2.resize(new, imgRegions.shape[1::-1])
dst = cv2.addWeighted(new, 0.9, imgRegions, 0.4, 0.0)

def nothing(x):
    pass

# Create a black image, a window
img = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('dst')

# create trackbars for color change
cv2.createTrackbar('Original','dst',1,100,nothing)
cv2.createTrackbar('Tumor','dst',1,100,nothing)
cv2.createTrackbar('Immune','dst',0,255,nothing)

dst = cv2.resize(dst, dim, interpolation = cv2.INTER_AREA)

while(1):
    cv2.imshow('image',dst)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    first = cv2.getTrackbarPos('Original','dst')
    second = cv2.getTrackbarPos('Tumor','dst')

    b = cv2.getTrackbarPos('B','image')

    first = first/50
    second = second/50
    dst = cv2.addWeighted(new, first, imgRegions, second, 0.0)
    dst = cv2.resize(dst, dim, interpolation = cv2.INTER_AREA)

cv2.destroyAllWindows()
