import os
import numpy as np

from skimage.external.tifffile import imread
from  skimage.color import rgb2gray
from skimage import measure, io

import matplotlib.pyplot as plt

import cv2

from PIL import Image, ImageDraw
from PIL import ImagePath

import xml.etree.ElementTree as ET

img = imread('/home/sudopizzai/Documents/Lab/deepLearning/brac/images/03.tif', key = 0)

#################################################################################################
xml = ET.Element('?xml version="1.0" encoding="UTF-8" standalone="no"?')

#first xml node -> Annotations
annotations = ET.SubElement(xml, 'Annotations')
annotations.set("MicronsPerPixel", "0.25")

#2 Node Annotation
annotation1 = ET.SubElement(annotations, 'Annotation')
annotation1.set('Id','1')
annotation1.set("LineColor", "16762880")
#################################################################################################
#3 node regions
regions = ET.SubElement(annotation1, 'Regions')

#trasform in gray
#imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#transform image in YCrCb
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)

#only when is hsv
plt.imshow(hsv, cmap='gray')
plt.show()

#define lower and upper range of colors
# B( Blue) beetween 50,150 give more borders
lower_range = np.array([50,20,100])
upper_range = np.array([150,150,230])

mask = cv2.inRange(hsv, lower_range, upper_range)

plt.imshow(mask, cmap='gray')
plt.show()

cv2.imshow('image', img)
cv2.imshow('mask', mask)


plt.show()

contours,h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

my_list = []

index = 1

offsetX = 13120
offsetY = 93750

#Counting the area of the contours find in the mask
for cnt in contours[0:20000]:

    area = cv2.contourArea(cnt)

    if area > 0.0 and area < 1000000.0 :

        region1 = ET.SubElement(regions, 'Region')
        region1.set('Id',str(index))
        vertices = ET.SubElement(region1, 'Vertices')

        my_list.append(area)
        #print(' area of the contour ', area)

        approx = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True)

        n = approx.ravel()
        i = 0

        for j in n :

            if(i % 2 == 0):

                x = (n[i] * 64) - offsetX
                y = (n[i + 1] * 64 ) - offsetY

                vertex1 = ET.SubElement( vertices, 'Vertex' )
                vertex1.set('X', str(x) )
                vertex1.set('Y',str(y))

            i = i + 1
        index = index + 1

print(' Lenght 1 contours ', len(contours))
#print(my_list)

plt.hist(my_list, bins=30)

plt.ylabel('Areas');
plt.show()

mydata = ET.tostring(xml)
myfile = open("items4.xml", "wb")
myfile.write(mydata)

infile = "items4.xml"
outfile = "items5.xml"

delete_list = ["</?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>"]

fin = open(infile)
fout = open(outfile, "w+")

for line in fin:
    for word in delete_list:
        line = line.replace(word, "")
    fout.write(line)
fin.close()
fout.close()

while(True):
   k = cv2.waitKey(5) & 0xFF
   if k == 27:
      break

cv2.destroyAllWindows()
