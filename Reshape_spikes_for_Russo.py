#!/usr/bin/env python3

import numpy as np
import math

### The following code sorts the output spikes from the Spike simulator into 2D.csv files for use in the Russo algorithm ###

#Load .txt files containing neuron id's and spikes
stim1_ids = np.genfromtxt('output_spikes_posttraining_stim1SpikeIDs.txt')
stim1_times = np.genfromtxt('output_spikes_posttraining_stim1SpikeTimes.txt')

#Specify the number of neurons in each excitatory layer and each inhibitory layer, and number of layers
excit_dim = 32*32
inhib_dim = 12*12
num_layers = 3

#Extract the neurons of interest
extracted_layer = 3 #Identify the neuron layer of interest (e.g. 3rd layer)
extracted_mask = np.where((excit_dim*(extracted_layer-1) < stim1_ids) & (stim1_ids <= extracted_layer*excit_dim)) #Returns an array of indeces for the neurons in the layer of interest

extracted_ids = np.take(stim1_ids, extracted_mask) #Returns an array of spike IDs, restricted to the layer of interest
extracted_times = np.take(stim1_times, extracted_mask) #Returns an array of spike times, restricted to the layer of interest

extracted_ids = np.reshape(extracted_ids, len(extracted_ids[0])) #Re-shape the array into a column array so that can be used by np.bincount later
extracted_ids = extracted_ids.astype(int) #Convert from spike output (float)

#Identify the neuron that spikes the max number of times, and return the number of times it spikes
max_spikes = np.max(np.bincount(extracted_ids))

#Initialize a NaN array with rows = number of unique neurons in the layer, and columns = number of spikes of the maximally active neuron
Russo_array = np.zeros([excit_dim, max_spikes])
Russo_array[:, :] = np.nan

print(max_spikes)
print(np.max(extracted_times))

#Iterate through each neuron of interest

for ii in range(0, len(Russo_array[:, 0])):
  #Extract a binary mask containing the indeces of when the neuron of interest has fired
  temp_mask = np.where(extracted_ids == (excit_dim*(extracted_layer - 1) + ii + 1))
  #Use the mask to identify all the spike times associated with that neuron, and assign it to the 'Russo_array'
  Russo_array[ii, 0:(np.size(np.take(extracted_times, temp_mask)))] = np.take(extracted_times, temp_mask)

#The Russo algorithm throws an error if a row contains no spikes, therefore condensed_array removes these
condensed_array = np.zeros([np.sum(Russo_array[:, 0] > 0), max_spikes])
condensed_array[:, :] = np.nan

temp_counter = 0
for ii in range(0, len(Russo_array[:, 0])):
  if math.isnan(Russo_array[ii, 0]) == 0: #Checks if the first element is NaN, in which case the entire row is; if not, then fills condensed_array with that row
    condensed_array[temp_counter, :] = Russo_array[ii, :]
    temp_counter += 1

print(condensed_array[0:10, 0:10])

#Output file as CSV
#np.savetxt("Russo_array_stim1.csv", condensed_array, delimiter=',')








