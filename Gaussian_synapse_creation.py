#!/usr/bin/env python3

import numpy as np
np.set_printoptions(threshold=np.inf)
#import pandas as pd #Pandas has multiple functions, including providing 'data_frame' objects that can be used for visualizing and analyzing data
#from matplotlib import pyplot as plt
#from PIL import Image
import os


#######
# Note that this code is only for excitatory feed-forward connectivity

# Note that this code assumes the two layers are of the same dimension

# *** Make sure synaptic delays unit is correct *** 
#######


x_dim = 12
y_dim = 12
total_num_first_layer_neurons = x_dim * y_dim
#total_num_inputs = x_dim * y_dim * num_Gabor_filters

Gaussian_SD = 1.0
num_unique_presynaptic_neurons = 3 #The number of pre-synaptic neurons that will synapse on to a post-synaptic neuron
num_synapses_per_pair = 4 #Assuming a pre-and post-synaptic neuron are connected, the number of synapses total they will share

#Ranges for values from which to sample uniformally
min_delay, max_delay = 0, 10 #Units in ms
min_weight, max_weight = 0, 1 


f_PreIDs = open("PresynapticIDs_Niels.txt", "w+")
f_PostIDs = open("PostsynapticIDs_Niels.txt", "w+")
f_Weights = open("SynapticWeights_Niels.txt", "w+")
f_Delays = open("SynapticDelays_Niels.txt", "w+")

#num_stimuli = 1
#num_transforms_per_image = 1

visual_array = np.zeros([x_dim, y_dim])

post_neuron_counter = 0

magic_index = 77
outside_array_counter = 0

#Iterate through each post synaptic neuron
for ii in range(0, x_dim):
	for jj in range(0, y_dim):
		for kk in range(0, num_unique_presynaptic_neurons):
			x_Gaussian = np.random.normal(0, Gaussian_SD)
			y_Gaussian = np.random.normal(0, Gaussian_SD)
			x_pre = int(np.round(ii + x_Gaussian))
			y_pre = int(np.round(jj + y_Gaussian))
			#print("Neuron post index is %d" % post_neuron_counter)
			#Checks if randomly selected pre-synaptic neuron is outside of the possible indeces; if passes, then creates synapse
			if  0 <= x_pre < x_dim and 0 <= y_pre < y_dim:
				pre_index = x_pre * 32 + y_pre + 1
				#Iterate through eah additional synapse for a given connection pair
				for mm in range(0, num_synapses_per_pair):
					#f_PreIDs.write("-%d" % (ii+1))
					if post_neuron_counter == magic_index:
						visual_array[x_pre, y_pre] += 1
					f_PostIDs.write("%d\n" % post_neuron_counter)
					f_Weights.write("%f\n" % np.random.uniform(min_weight, max_weight))
					f_Delays.write("%d\n" % np.random.randint(min_delay, max_delay))
			else:
				outside_array_counter += 1
				#print("Indexed outside of array") #If outside of indeces, the particular synapse is skipped over. Note therefore that neurons at the edge of the layer will have on average fewer input synapses
		
		post_neuron_counter += 1
		if post_neuron_counter == 10000:
			print("Breaking")
			break
	if post_neuron_counter == 10000:
		print("Breaking")
		break

print("\nThe number of occasions indexing outside of the array was %d\n" % outside_array_counter)
print(visual_array)		

f_PostIDs.close()
f_Weights.close()
f_Delays.close()









