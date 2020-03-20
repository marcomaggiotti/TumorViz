import os
import numpy as np

from skimage.external.tifffile import imread
from  skimage.color import rgb2gray
from skimage import measure, io

import matplotlib.pyplot as plt

import cv2

from PIL import Image, ImageDraw
from PIL import ImagePath

img = imread('/home/sudopizzai/Documents/Lab/deepLearning/brac/images/03.tif', key = 0)

#trasform in gray
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#transform image in YCrCb
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)

plt.imshow(hsv, cmap='gray')
plt.show()

#define lower and upper range of colors
lower_range = np.array([0,50,0])
upper_range = np.array([255,100,255])

mask = cv2.inRange(hsv, lower_range, upper_range)

cv2.imshow('image', img)
cv2.imshow('mask', mask)

contours,h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print(' Lenght 1 contours ', len(contours))

ret,thresh = cv2.threshold(imgray,0,255,0)
plt.imshow(thresh, cmap='gray')
plt.show()
contours,h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print(' Lenght contours ', len(contours))

plt.imshow(imgRegions, cmap='gray')
plt.show()

while(True):
   k = cv2.waitKey(5) & 0xFF
   if k == 27:
      break

cv2.destroyAllWindows()
