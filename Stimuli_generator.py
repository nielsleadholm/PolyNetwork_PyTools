#!/usr/bin/env python3

import numpy as np
from PIL import Image
np.set_printoptions(threshold=np.inf, linewidth=120)


### The following code randomly generates images of 32x32 pixel size with arbitrary combinations of simple features (oriented lines) ###


###Generate the 256 unique arrangements of four oriented bars (4^4 possibilities)###

gen_array = np.zeros((2, 2, 256)) #Initialize array for holding the values indicating which feature is located in which quadrant

#Each of the four bars are uniquely identified by integers 0-3
#Each of the four corners are indicated by their position in the 4D array, beginning in the top left-hand corner and moving clock-wise

stim_counter = 0 #Counter is used to select each of the 256 unique objects
#At each quadrant, choose one of the four possible stimuli (vertical, horizontal, or one of the two diagonal bars); note repetition is allowed
for ii in range(0, 4): #Fourth corner feature (lower left-hand)
  for jj in range(0, 4): #Third corner feature (lower right-hand)
    for kk in range(0, 4): #Second corner feature (upper right-hand)
      for ll in range(0, 4): #First corner feature (upper left-hand)
        gen_array[:, :, stim_counter] = [(ll, kk), (jj, ii)]
        stim_counter += 1



### Generate 32x32 arrays that will contain the pixel forms of the stimuli ###

#Initialize array for storing the values that will be converted into grey-scale pixels; each edge will hav the same pixel area (4x2 pixels total)
image_array = np.ones((32, 32, 256))

#Iterate through each of the 256 unique images
for ii in range(0, 256):
  #Iterate through each corner, inputting the appropriate feature based on gen_array
  for jj in range(0, 2): #Index of left or right side
    for kk in range(0, 2): #Index of top or bottom 
      if gen_array[kk, jj, ii] == 0: #Vertical bar
        image_array[(kk*16)+4:(kk*16)+12, (jj*16)+7:(jj*16)+9, ii] = 0
      elif gen_array[kk, jj, ii] == 1: #Horizontal bar
        image_array[(kk*16)+7:(kk*16)+9, (jj*16)+4:(jj*16)+12, ii] = 0
      elif gen_array[kk, jj, ii] == 2: #Downhill bar i.e. \
        for mm in range(0, 8):
          image_array[(kk*16)+4+mm:(kk*16)+6+mm, (jj*16)+4+mm:(jj*16)+5+mm, ii] = 0
      elif gen_array[kk, jj, ii] == 3: #Uphill bar i.e. /
        for mm in range(0, 8):
          image_array[(kk*16)+4+mm:(kk*16)+6+mm, ((jj+1)*16)-5-mm:((jj+1)*16)-4-mm, ii] = 0

image_array = image_array.astype(np.float32) #Cast the image_array as the appropriate Numpy type
#NB that float 32 is 32 bits; 8 bits = 1 byte, therefore float32 is 4 bytes (the same as a C++ float; an unspecified 
  # numpy float is 8 bytes)


### Create and output a sub-set of the total stimlus set ###

#Identify indeces of interest; these are indexed from 0, but all image files and analysis will be labeled indexing from 1
sub_image_indeces = [0, 10, 56, 255]
num_sub_images = len(sub_image_indeces)
sub_image_array = image_array[:, :, sub_image_indeces]

#Initialize flat array to store all firing rates; this will be the firing rate file provided to the Spike simulator
total_firing_rates = 32*32*num_sub_images
sub_image_array_flat = np.ones((total_firing_rates))

#Insert firing rate values into the flat array
#Note that the flat array can be read as each 32x32 image following the next; within each image, all the columns of a row are read before moving on to the next
flatten_counter = 0
for image_num in range(0, num_sub_images):
  ###Indexing over transforms (when eventually included) should be performed here ###
  for row in range(0,32):
    for column in range(0, 32):
      sub_image_array_flat[flatten_counter] = sub_image_array[row, column, image_num]
      flatten_counter += 1

#Output sub_image_array as a data file for use of firing rates
sub_file = open("sub_file.gbo", "wb") #'wb' indicates that the file should be written in binary mode
sub_image_array_flat = sub_image_array_flat.astype(np.float32)
sub_image_array_flat.tofile("sub_file.gbo")

#Unit test to check binary reading and writing is functioning as expected
test_bytes = open("sub_file.gbo", mode="rb")
test_output = np.fromfile("sub_file.gbo", dtype = np.float32)

if np.array_equal(test_output, sub_image_array_flat) == 0:
  #Throw an error if the arrays are not the same when read
  print("Error in writing to and/or reading from binary file")

#Output sub_image_array as .png files
sub_image_array = (sub_image_array* 255).astype(np.uint8)
for index in range(0, num_sub_images):
  img = Image.fromarray(sub_image_array[:, :, index], 'L')
  img.save('stim' + str(sub_image_indeces[index] + 1) + '.png')



#### Output All Generated Images ###

#Output image_array as a data file for use of firing rates


#Convert array into grey-scale .png image for visualization of the data and creation of figures

"""
image_array = (image_array* 255).astype(np.uint8) #Convert the array into grey-scale format

for index in range(0, 256):
  img = Image.fromarray(image_array[:, :, index], 'L')
  img.save('stim' + str(index + 1) + '.png')

"""
