#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#The code can be used to visualize the firing rates of neurons from the Spike simulator, as well as to evaluate the
#information content in the firing rates of the neurons

params = {'number_stimuli' : 2,
	'network_layer': 3,
	'number_of_presentations' : 50,
	'duration_of_presentations' : 1,
	'excit_dim' : 32*32,
	'inhib_dim' : 12*12,
	'num_layers' : 3,
	'mean_FR_visualization_bool' : 0,
	'information_theory_bool' : 0}




# *** it will be useful later to specify which layer e.g. information theory should be performed on, so that the entire network is not included
# *** in particular, while FR visualization includes inhibitory neurons, these could be excluded from the information theory analysis




def main():
	data_dic = extract_spike_data(params)

	#Iterate through each stimulus

	FR_dic = dict() #Stores arrays containing the firing rates of each neuron, for each stimulus presentation
	mean_FR_dic = dict() #Stores the mean firing rates of each neuron for a given stimulus presentation
	#Extract the mean firing rates
	vector_size = params['excit_dim']*params['num_layers'] + params['inhib_dim']*params['num_layers']
	for stimuli_iter in range(0, params['number_stimuli']):
		FR_dic[stimuli_iter] = extract_firing_rates(params, stimuli_iter, data_dic, vector_size)
		mean_FR_dic[stimuli_iter] = find_mean_firing_rates(params, FR_dic[stimuli_iter], vector_size)

	(lower_threshold, upper_threshold) = information_theory_discretize(params, FR_dic, vector_size)
	information_theory_dic = information_theory_counting(params, FR_dic, vector_size, lower_threshold, upper_threshold)
	information_theory_results = information_theory_calculation(params, information_theory_dic)

	fig, ax = plt.subplots()

	plt.hist(information_theory_results, bins=np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
	ax.set_ylabel('Number of Neurons')
	ax.set_xlabel('Information (bits)')

	plt.show()

	if params['mean_FR_visualization_bool'] == 1:
		#Take the difference in the mean firing rates
		mean_FR_difference = mean_FR_dic[0] - mean_FR_dic[1]

		#Plot the results
		plt.figure(figsize=(12,9))
		plt.scatter(np.arange(0,(vector_size)), mean_FR_difference, s=5)
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

def extract_firing_rates(params, stimuli_iter, data_dic, vector_size):

	#Initialize a vector to hold the firing rates of each neuron
	FR_array = np.zeros([vector_size, params['number_of_presentations']])

	#Iterate through each stimulus presentaiton
	for presentation_iter in range(0,params['number_of_presentations']):
	  #Apply a mask to the times data to extract spikes in the period of interest
	  mask = ((data_dic[stimuli_iter]["times"] > (presentation_iter-1)*params['duration_of_presentations']) & 
	  	(data_dic[stimuli_iter]["times"] <= presentation_iter*params['duration_of_presentations']))
	  
	  #Iterate through each neuron ID, counting the total number of appearances in the masked-array
	  for ID_iter in range(0, vector_size):
	    FR_array[ID_iter][presentation_iter] = np.count_nonzero(data_dic[stimuli_iter]["ids"][mask] == ID_iter)
	  
	  #Divide these values by the duration of the presentation
	  FR_array[:][presentation_iter] = FR_array[:][presentation_iter] / params['duration_of_presentations']

	return FR_array

def find_mean_firing_rates(params, FR_array, vector_size):
	mean_FR = np.zeros(vector_size)

	mean_FR = np.sum(FR_array, axis = 1)
	mean_FR = mean_FR / params['number_of_presentations']

	return mean_FR

#Find the firing rate thresholds that determine if a firing rate is low, medium or high
def information_theory_discretize(params, FR_dic, vector_size):
	#Note that as used in the Hutter thesis (2018), each neuron has its own thresholds
	#These are based on the minimal and maximal firing rate obtained across all presentations, the difference of which is divided into three equal bins

	#Vector of minimum firing rates for each neuron (across presentations of all stimuli)
	#Minimum is first taken for each particular stimulus (and so iterating through them), and then across all stimuli
	temp_min_array = np.zeros([vector_size, params['number_stimuli']])
	for stimuli_iter in range(0, params['number_stimuli']):
		temp_min_array[:, stimuli_iter] = np.amin(FR_dic[stimuli_iter], axis=1)
	min_vector = np.amin(temp_min_array, axis = 1)

	#Vector of maximum firing rates for each neuron (across presentations of all stimuli)
	temp_max_array = np.zeros([vector_size, params['number_stimuli']])
	for stimuli_iter in range(0, params['number_stimuli']):
		temp_max_array[:, stimuli_iter] = np.amax(FR_dic[stimuli_iter], axis=1)
	max_vector = np.amax(temp_max_array, axis = 1)

	#Generate the vector containing the thresholds for separating low-medium and medium-high for each neuron
	lower_threshold = (max_vector - min_vector)*(1/3)
	upper_threshold = (max_vector - min_vector)*(2/3)

	return (lower_threshold, upper_threshold)


def information_theory_counting(params, FR_dic, vector_size, lower_threshold, upper_threshold):
	#Information can be encoded in firing rates by discretizing the rates into e.g. low, medium, and high rates, which will be done here

	information_theory_dic = dict()	
	#For each stimulus, find the number of times that a particular neuron's firing rate was low, medium, or high
	for stimuli_iter in range(0, params['number_stimuli']):
		firing_rate_counter = np.zeros([vector_size, 3]) #Array to store these counts

		#Apply a mask such that all firing rates relative to a particula threshold return a 1, then sum their values
		firing_rate_counter[:, 0] = np.sum(FR_dic[stimuli_iter]<lower_threshold[:, None], axis=1) #lower counts
		firing_rate_counter[:, 2] = np.sum(FR_dic[stimuli_iter]>upper_threshold[:, None], axis=1) #upper counts
		firing_rate_counter[:, 1] = params['number_of_presentations'] - (firing_rate_counter[:, 0] + firing_rate_counter[:, 2]) #mid firing rate counts

		#Check that all of the separate counts sum appropriately
		assert np.all((firing_rate_counter[:, 0]+firing_rate_counter[:, 1]+firing_rate_counter[:, 2]) == (np.ones(vector_size)*params['number_of_presentations']))

		information_theory_dic[stimuli_iter] = firing_rate_counter

	#Return an array containing the number of presentations where the neuron activity was low, medium, and high respectively
	return information_theory_dic

def information_theory_calculation(params, information_theory_dic):
	#Information_theory_dic contains an array for each stimulus presentation
	#This array contains the number of counts of low, medium and high firing rates for each neuron
	no_math_error = 0.00000000000001 #Prevent division by zero


	# *** Initially find the result for stimulus 1 presentation


	#The conditional probabilities of a particular neuron having low, medium and high activity for a particular stimulus
	conditional_prob_array = information_theory_dic[0]/params['number_of_presentations']
	# print(np.shape(conditional_prob_array))
	# print(conditional_prob_array[0:3, 0:3])

	#The marginal propabailities of a particular neuron having low, medium and high activity 
	marginal_prob_array = (information_theory_dic[0]+information_theory_dic[1])/(params['number_of_presentations']*params['number_stimuli'])
	# print(np.shape(marginal_prob_array))
	# print(marginal_prob_array[0:3, 0:3])

	information_low = np.multiply(conditional_prob_array[:, 0], np.log2(np.divide(conditional_prob_array[:, 0], marginal_prob_array[:, 0]+no_math_error)+no_math_error))
	information_mid = np.multiply(conditional_prob_array[:, 1], np.log2(np.divide(conditional_prob_array[:, 1], marginal_prob_array[:, 1]+no_math_error)+no_math_error))
	information_high = np.multiply(conditional_prob_array[:, 2], np.log2(np.divide(conditional_prob_array[:, 2], marginal_prob_array[:, 2]+no_math_error)+no_math_error))
	
	information_theory_results = information_low + information_mid + information_high

	return information_theory_results

main()
