#!/usr/bin/env python3

import numpy as np
import math

### The following tool sorts the output spikes from the Spike simulator (1D arrays) into 2D .csv files suitable for analysis by the Russo algorithm ###

#For details on the Russo algorithm, please see the publication "Cell assemblies at multiple time scales with arbitrary lag constellations", Russo and Durstewitz, 2017
#Spike is a simulator written in C++/CUDA for spiking neural networks, more information on which can be found here: https://sites.google.com/view/spike-simulator/home
#This code assumes the user is analyzing neural networks wherein each layer is a 2D lattice, and multiple layers are present (see for example the network used in the paper 
#"The Emergence of Polychronization and Feature Binding in a Spiking Neural Network Model of the Primate Ventral Visual System", Eguchi et al, 2018)

#The Russo algorithm requires a 2D array where columns represent neuron IDs, and each neurons first to nth spike are indicated by the column
#Each array entry specifies the time (in seconds) at which the spike occurred; empty entries are required to be filled with a NaN identifier
#As the Russo algorithm does not accept neurons that never spike (i.e. empty rows), the following code adds a single spike at a random time-point for such neurons

#The user must provide below parameters that were used in generating the Spike neural network simulation
#Specify the number of neurons in each excitatory layer and each inhibitory layer, the number of layers, and the number of stimuli (the latter two are both counted from 1)
excit_dim = 32*32
inhib_dim = 12*12
num_layers = 3
extracted_layer = 3 #Identify the neuron layer of interest (e.g. 3rd layer)
num_stimuli = 2


#Load .txt files containing neuron id's and spikes
for jj in range(0, num_stimuli):
	stim_ids = np.genfromtxt('output_spikes_posttraining_stim' + str(jj+1) +'SpikeIDs.txt')
	stim_times = np.genfromtxt('output_spikes_posttraining_stim1SpikeTimes.txt')

	#Extract the neurons of interest
	extracted_layer = 3 #Identify the neuron layer of interest (e.g. 3rd layer)
	extracted_mask = np.where((excit_dim*(extracted_layer-1) < stim_ids) & (stim_ids <= extracted_layer*excit_dim)) #Returns an array of indeces for the neurons in the layer of interest

	extracted_ids = np.take(stim_ids, extracted_mask) #Returns an array of spike IDs, restricted to the layer of interest
	extracted_times = np.take(stim_times, extracted_mask) #Returns an array of spike times, restricted to the layer of interest

	extracted_ids = np.reshape(extracted_ids, len(extracted_ids[0])) #Re-shape the array into a column array so that it can be used by np.bincount later
	extracted_ids = extracted_ids.astype(int) #Convert from spike output (float)

	#Identify the neuron that spikes the max number of times, and return the number of times it spikes, used later
	max_spikes = np.max(np.bincount(extracted_ids))

	#Initialize a NaN array with rows = number of unique neurons in the layer, and columns = number of spikes of the maximally active neuron
	Russo_array = np.zeros([excit_dim, max_spikes])
	Russo_array[:, :] = np.nan

	# print(max_spikes)
	# print(np.max(extracted_times))

	#Iterate through each neuron of interest, inserting its spikes into the Russo-suitable array; if a neuron never spikes, insert a single random spike
	for ii in range(0, len(Russo_array[:, 0])):
	  #Extract a binary mask containing the indeces of when the neuron of interest has fired
	  temp_mask = np.where(extracted_ids == (excit_dim*(extracted_layer - 1) + ii + 1))
	  #Use the mask to identify all the spike times associated with that neuron, and assign it to the 'Russo_array'
	  Russo_array[ii, 0:(np.size(np.take(extracted_times, temp_mask)))] = np.take(extracted_times, temp_mask)
	  if math.isnan(Russo_array[ii, 0]) == 1: #Checks if the first element is NaN, in which case the entire row is (i.e. the neuron never spiked)
	    Russo_array[ii, 0] = np.random.random()*np.max(extracted_times) #Assigns the neuron a single spike, time of which is sampled from a continuous uniform distribution

	#print(Russo_array[0:20, 0:5])
	#print(np.shape(Russo_array))

	#Output file as CSV
	np.savetxt("posttraining_stim" + str(jj+1) + "_Russo.csv", Russo_array, delimiter=',')








