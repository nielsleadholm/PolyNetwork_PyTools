#!/usr/bin/env python3

import numpy as np
from PIL import Image

#The following code randomly generates images of 32x32 pixel size with arbitrary combinations of simple features (oriented lines)


#Generate the 256 unique arrangements of the edges (4^4 possibilities)

gen_array = np.zeros((2, 2, 256)) #Initialize array for holding the values indicating which feature is located in which quadrant

#Each of the four bars are uniquely identified by integers 0-3; note repetition is allowed
#Each of the four corners are indicated by their position in the 4D array, beginning in the top left-hand corner and moving clock-wise

#Counter is used to select each of the 256 unique objects
counter = 0
#At each quadrant, choose one of the four possible stimuli (vertical, horizontal, or one of the two diagonal bars); note repetition is allowed
for ii in range(0, 4): #Fourth corner feature (lower left-hand)
  for jj in range(0, 4): #Third corner feature (lower right-hand)
    for kk in range(0, 4): #Second corner feature (upper right-hand)
      for ll in range(0, 4): #First corner feature (upper left-hand)
        gen_array[:, :, counter] = [(ll, kk), (jj, ii)]
        counter += 1

print(counter)



#Generate an array of size 32x32 containing the oriented edges; each edge should be of the same size (4x2 pixels)

#gen_array_temp = gen_array[:, :, 100]

image_array = np.ones((32, 32, 256)) #Initialize array for storing the values that will be converted into grey-scale pixels

#Horizontal and vertical bars are each 8x2 or 2x8 pixels in size; they are thus, within their 16x16 quadrant, offset from either end by either 7 or 4 pixels

#Iterate through each of the 256 unique images
for ii in range(0, 256):
  #Iterate through each corner, inputting the appropriate feature
  for jj in range(0, 2): #Index of left or right side
    for kk in range(0, 2): #Index of top or bottom 
      if gen_array[kk, jj, ii] == 0: #Vertical bar
        image_array[(kk*16)+4:(kk*16)+12, (jj*16)+7:(jj*16)+9, ii] = 0
      elif gen_array[kk, jj, ii] == 1: #Horizontal bar
        image_array[(kk*16)+7:(kk*16)+9, (jj*16)+4:(jj*16)+12, ii] = 0
      elif gen_array[kk, jj, ii] == 2: #Downhill bar = \
        for mm in range(0, 8):
          image_array[(kk*16)+4+mm:(kk*16)+6+mm, (jj*16)+4+mm:(jj*16)+5+mm, ii] = 0
      elif gen_array[kk, jj, ii] == 3: #Uphill bar = /
        for mm in range(0, 8):
          image_array[(kk*16)+4+mm:(kk*16)+6+mm, ((jj+1)*16)-5-mm:((jj+1)*16)-4-mm, ii] = 0

#np.set_printoptions(threshold=np.nan)
#print(image_array)

image_array.astype(float)



#Create a subset of image-array

sub_image_array = image_array[:, :, 0]
#print(sub_image_array)

#Flatten array (C-style, i..e row-major order)
sub_image_array.flatten(order='C')


#Output sub_image_array as a data file for use of firing rates

sub_file = open("sub_file.gbo", "wb") #'wb' indicates that the file should be written in binary mode

sub_image_array.tofile("sub_file.gbo")