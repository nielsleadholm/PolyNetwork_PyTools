#!/usr/bin/env python3

import numpy as np
#import pandas as pd #Pandas has multiple functions, including providing 'data_frame' objects that can be used for visualizing and analyzing data
#from matplotlib import pyplot as plt
#from PIL import Image
import os


#######
# *** Note that this code is based on the premise that the Gabor filters are held 
# contiguously in the firing rate array ***

# *** Make sure synaptic delays unit is correct *** 
#######


x_dim = 32
y_dim = 32
num_Gabor_filters = 2
total_num_first_layer_neurons = x_dim * y_dim
#total_num_inputs = x_dim * y_dim * num_Gabor_filters

num_synapses_per_pair = 3

#Ranges for values from which to sample uniformally
min_delay, max_delay = 0, 10 #Units in ms
min_weight, max_weight = 0, 1 


f_PreIDs = open("PresynapticIDs_Niels.txt", "w+")
f_PostIDs = open("PostsynapticIDs_Niels.txt", "w+")
f_Weights = open("SynapticWeights_Niels.txt", "w+")
f_Delays = open("SynapticDelays_Niels.txt", "w+")

#num_stimuli = 1
#num_transforms_per_image = 1


#Iterate through each post synaptic neuron
for ii in range(0, total_num_first_layer_neurons):
	#Iterate through each filter
	for jj in range(0, num_Gabor_filters):
		#Iterate through eah additional synapse for a given connection pair
		for kk in range(0, num_synapses_per_pair):
			f_PreIDs.write("-%d" % (ii+1))
			f_PostIDs.write("%d\n" % ii)
			f_Weights.write("%f\n" % np.random.uniform(min_weight, max_weight))
			f_Delays.write("%d\n" % np.random.randint(min_delay, max_delay))


f_PostIDs.close()
f_Weights.close()
f_Delays.close()









