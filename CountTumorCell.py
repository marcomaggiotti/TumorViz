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

from skimage import morphology

import openslide
import PIL.Image

from torchviz import make_dot
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
from xml.dom import minidom
import xml.etree.ElementTree as ET


#slide file svs or mrxs
#slideName = "/home/sudopizzai/Documents/data/TCGA/TCGA-A8-A090-01Z-00-DX1.01574070-D65E-486F-B69D-0F8E3816D057.svs"
slideName = "/home/sudopizzai/Documents/data/brca1/mib1/BRCA03_MIB1.mrxs"

#root xml element
xml = ET.Element('?xml version="1.0" encoding="UTF-8" standalone="no"?')

#first xml node -> Annotations
annotations = ET.SubElement(xml, 'Annotations')
annotations.set("MicronsPerPixel", "0.25")

#2 Node Annotation
annotation1 = ET.SubElement(annotations, 'Annotation')
annotation1.set('Id','1')
annotation1.set("LineColor", "16762880")

#3 node regions
regions = ET.SubElement(annotation1, 'Regions')

#Reader from Openslide
openSlideReader  = openslide.OpenSlide(slideName)
cmap= matplotlib.cm.tab10
level = 4
mask_level = 4

size_slide = openSlideReader.level_dimensions[level]
slide_origin = (0 , 0)

img_resized = openSlideReader.read_region(slide_origin, mask_level, size_slide )

print("image size ", openSlideReader.level_dimensions[0])

shape = img_resized.size #openSlideReader.level_dimensions[level]
print("shape of resized slide", shape)

img = img_resized #np.asarray(new_tile9)[:, :, 0:3] #openSlideReader.read_region((0, 0), mask_level, openSlideReader.level_dimensions[mask_level]) #
imgArray = np.array(img)

gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#plt.imshow(gray) # blue on green well defined dots
#plt.show()
##### first
mask = cv2.threshold(gray, 255, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
stats = cv2.connectedComponentsWithStats(mask, 8)[2]
label_area = stats[1:, cv2.CC_STAT_AREA]

min_area, max_area = 150, 200 # min/max for a single circle
singular_mask = (min_area < label_area) & (label_area <= max_area)
circle_area = np.mean(label_area[singular_mask])

#this is should be the number of circles retreived by the treshold but I am not sure about
n_circles = int(np.sum(np.round(label_area / circle_area)))

print('Total circles:', n_circles)

##### second findContours
blur = cv2.GaussianBlur(gray, (7, 7), 2)
ret,thresh = cv2.threshold(gray,127,255,1)
contours,h = cv2.findContours(thresh,1,2)

print(' Lenght contours ', len(contours) )
#imCopy = imgArray.copy()
#ret,thresh = cv2.threshold(imgArray,127,255,0)from skimage import morphology

##image, contours, hierarchy =  cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#cv2.drawContours(imCopy,contours,-1,(0,255,0), 3)
#imCopy = cv2.resize(imCopy, (2000, 1500))
#cv2.imshow('draw contours',imCopy)
#cv2.waitKey(0)
index = 1

#empty mask with the same size of image
mask = np.zeros(imgArray.shape[:2], dtype=imgArray.dtype)

for cnt in contours[40000:60000]:

    area = cv2.contourArea(cnt)

    if area > 2 and area < 200:

        region1 = ET.SubElement(regions, 'Region')
        region1.set('Id',str(index))
        vertices = ET.SubElement(region1, 'Vertices')

        print(' area of the contour ', area)



        #we draw one by on the contour to the mask
        cv2.drawContours(mask, [cnt], 0, (255), -1)
        #print( area," ", cnt )
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

        n = approx.ravel()
        i = 0

        for j in n :
            if(i % 2 == 0):
                x = (n[i] * 16) - 13120
                y = (n[i + 1] * 16 ) - 93770
                vertex1 = ET.SubElement(vertices, 'Vertex')
                vertex1.set('X', str(x) )
                vertex1.set('Y',str(y))

            i = i + 1
        index = index + 1

    #cv2.drawContours( imgArray, [cnt], -1, (0,255,0), 3 )
    #imgArray = cv2.resize(imgArray, (960, 540)) #Resize image

mask = cv2.bitwise_not(mask)
mask = cv2.resize(mask, (2000, 1500))
cv2.imshow("output", mask)  # Show image
cv2.waitKey(0)

cv2.drawContours( imgArray, contours, -1, (0,255,0), 3 )

ResizedImgArray = cv2.resize(imgArray, (2000, 1500))
cv2.imshow("output", ResizedImgArray)  # Show image
cv2.waitKey(0)

mydata = ET.tostring(xml)
myfile = open("items2.xml", "wb")
myfile.write(mydata)

infile = "items2.xml"
outfile = "items3.xml"

delete_list = ["</?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>"]
fin = open(infile)
fout = open(outfile, "w+")
for line in fin:
    for word in delete_list:
        line = line.replace(word, "")
    fout.write(line)
fin.close()
fout.close()
