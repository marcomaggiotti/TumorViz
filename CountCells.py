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
from tqdm.autonotebook import tqdm
from  skimage.color import rgb2gray
from skimage import measure
import skimage.color as color
import skimage
import openslide
import PIL.Image

from torchviz import make_dot
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator

osh  = openslide.OpenSlide("/home/sudopizzai/Documents/data/TCGA/TCGA-A8-A090-01Z-00-DX1.01574070-D65E-486F-B69D-0F8E3816D057.svs")
cmap= matplotlib.cm.tab10
level = 1
mask_level = 1
size_slide = osh.level_dimensions[level]
slide_origin = (0 , 0)

img_resized = osh.read_region(slide_origin, mask_level, size_slide )

plt.imshow(img_resized)
plt.show()

print("image size ", img_resized.size)
shape = img_resized.size #osh.level_dimensions[level]

img = img_resized #np.asarray(new_tile9)[:, :, 0:3] #osh.read_region((0, 0), mask_level, osh.level_dimensions[mask_level]) #
imgArray = np.array(img)

gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(gray) # blue on green well defined dots
plt.show()
##### first
mask = cv2.threshold(gray, 255, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
stats = cv2.connectedComponentsWithStats(mask, 8)[2]
label_area = stats[1:, cv2.CC_STAT_AREA]

min_area, max_area = 50, 350  # min/max for a single circle
singular_mask = (min_area < label_area) & (label_area <= max_area)
circle_area = np.mean(label_area[singular_mask])

n_circles = int(np.sum(np.round(label_area / circle_area)))

print('Total circles:', n_circles)

##### second findContours

ret,thresh = cv2.threshold(gray,127,255,1)
contours,h = cv2.findContours(thresh,1,2)

imCopy = imgArray.copy()
ret,thresh = cv2.threshold(imgArray,127,255,0)

##image, contours, hierarchy =  cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(imCopy,contours,-1,(0,255,0), 3)
imCopy = cv2.resize(imCopy, (3000, 2000))
cv2.imshow('draw contours',imCopy)
cv2.waitKey(0)

"""
for cnt in contours[0:100]:
    cv2.drawContours(imgArray,[cnt],-1,(0,255,0),10)
    imgArray = cv2.resize(imgArray, (960, 540))                    # Resize image
    cv2.imshow("output", imgArray)                            # Show image
    cv2.waitKey(0)
"""
print("mask")

plt.imshow(mask) # blue on yellow well defined dots
plt.show()

plt.imshow(thresh) #
plt.show()

#image_file = Image.open(np.array(gray))

cv2_im = cv2.cvtColor(gray,cv2.COLOR_BGR2RGB) # now in black and white
pil_im = Image.fromarray(cv2_im)
plt.imshow(pil_im)
plt.show()

#cnts = cv2.findContours(pil_im, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
#print(cnts)

#xcnts = []
"""
for cnt in cnts:
    if 0 < cv2.contourArea(cnt) < 20000:
        xcnts.append(cnt)

imgg=rgb2gray(np.array(img))
#imgg = img.convert('LA')
th, threshed = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

print("\nDots number: {}".format(len(xcnts)))

plt.imshow(threshed)
plt.show()

mask = np.bitwise_and(imgg >0 ,imgg <200/255)
plt.imshow(mask)
plt.show()
"""
