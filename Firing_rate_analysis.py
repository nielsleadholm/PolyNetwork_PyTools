#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#The code can be used to visualize the firing rates of neurons from the Spike simulator, as well as to evaluate the
#information content in the firing rates of the neurons

params = {'number_stimuli' : 2,
	'network_layer': 3,
	'number_of_presentations' : 2,
	'duration_of_presentations' : 1,
	'excit_dim' : 32*32,
	'inhib_dim' : 12*12,
	'num_layers' : 3,
	'mean_FR_visualization_bool' : 1,
	'information_theory_bool' : 0}

def main():
	data_dic = extract_spike_data(params)

	#Iterate through each stimulus

	FR_dic = dict()
	#Extract the mean firing rates
	if params['mean_FR_visualization_bool'] == 1:
		vector_size = params['excit_dim']*params['num_layers'] + params['inhib_dim']*params['num_layers']
		for stimuli_iter in range(0, params['number_stimuli']):
			FR_dic[stimuli_iter] = extract_mean_firing_rates(params, stimuli_iter, data_dic, vector_size)


		#Take the difference in the mean firing rates
		FR_difference = FR_dic[0] - FR_dic[1]

		#Plot the results
		plt.figure(figsize=(12,9))
		plt.scatter(np.arange(0,(vector_size)), FR_difference, s=5)
		plt.show()

def extract_spike_data(params):

	data_dic = dict()

	#Iterate through each stimulus
	for ii in range(0, params['number_stimuli']):
		#Use Pandas to generate a 'data-frame', where each row in this case has a name (ids or times), and the 
		#columns in those rows contain the values of interest
		#Note the stimuli in the file names are index from 1, not 0
		data_dic[ii] = pd.DataFrame(
		  data = {
		      "ids": np.fromfile("output_spikes_posttraining_stim" + str(ii+1) + "SpikeIDs.txt", dtype=np.int32, sep=' '),
		      "times": np.fromfile("output_spikes_posttraining_stim" + str(ii+1) + "SpikeTimes.txt", dtype=np.float32, sep=' '),
		  }
		)

	return data_dic

def extract_mean_firing_rates(params, stimuli_iter, data_dic, vector_size):

	#Initialize a vector to hold the firing rates of each neuron
	FR_vec = np.zeros(vector_size)

	#Iterate through each stimulus presentaiton
	for presentation_iter in range(0,params['number_of_presentations']):
	  #Apply a mask to the times data to extract spikes in the period of interest
	  mask = ((data_dic[stimuli_iter]["times"] > (presentation_iter-1)*params['duration_of_presentations']) & 
	  	(data_dic[stimuli_iter]["times"] <= presentation_iter*params['duration_of_presentations']))
	  
	  temp_vec = np.zeros(vector_size)
	  
	  #Iterate through each neuron ID, counting the total number of appearances in the masked-array
	  for ID_iter in range(0, vector_size):
	    temp_vec[ID_iter] = np.count_nonzero(data_dic[stimuli_iter]["ids"][mask] == ID_iter)
	  
	  #Divide these values by the duration of the presentation
	  temp_vec[:] = temp_vec[:] / params['duration_of_presentations']
	  
	  #Add this vector of firing rates to the previous one
	  FR_vec[:] = FR_vec[:] + temp_vec[:]
	  
	#Normalize the firing rate vector by the number of presentations
	FR_vec[:] = FR_vec[:] / params['number_of_presentations']


	return FR_vec

main()
