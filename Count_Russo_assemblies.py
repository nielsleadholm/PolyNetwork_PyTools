#!/usr/bin/env python3

import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy.io as sio
import warnings
import matplotlib.pyplot as plt
import matplotlib
import pickle
import time

#This code can be used after applying the Russo algorithm to extract polychronous assemblies from parallel spike-train data
#Using the extracted assemblies, the following algorithm will scan for and count the number of instances of each assembly, relative to a particular stimulus presentation

#Definitions of parameters:
#epsilon0 : primary cut-off margin for spike times in assembly search, given in seconds; note it covers both sides of the prescribed spike times, so an epsilon of 3ms corresponds to a bin/window of 3ms total, with 1.5ms on either side
#Russo_bin_size : the temporal resolution (bin size) in seconds used when running the Russo algorithm
#network_layer : the layer that should be analysed
#number_of_presentations : how many times each stimulus is presented in the datasets
#duration_of_presentations : length of each stimulus presentation, in seconds
#shuffle_Boolean: load dataset where spikes have been shuffled in time, preserving firing rates but breaking any temporal relations
#synchrony bool : treat all lags extracted by the Russo algorithm as 0
#templates_alt_stream_bool: check for the presence of assemblies, if using the Russo assembly templates on the alternative stream, but where the neuron indices are corrected for that stream
#epsilon_iter_bool : determine whether the number of PNGs captured should be analyzed over a series of increasing epsilon windows

params = {'epsilon0' : 0.005,
	'Russo_bin_size' : 0.003,
	'number_stimuli' : 2,
	'network_layer': 3,
	'group_dim': 5*5,
	'number_of_presentations' : 50,
	'duration_of_presentations' : 0.2,
	'shuffle_Boolean' : False,
	'synchrony_bool' : False,
	'templates_alt_stream_bool' : False,
	'epsilon_iter_bool' : False,
	'epsilon_iter_step' : 0.00025,
	'epsilon_max' : 0.015,
	'information_theory_bool' : True}

def main(params):

	#stimuli_iter determines which stimulus was presented for the purpose of Russo assembly extraction; dataset_iter (below) determines which stimulus was presented
	# for the purpose of scanning for PNG activations
	stimuli_iter = 0 #Temporarily set to just the first stimulus

	assemblies_list = import_assemblies(params, stimuli_iter)
	number_Russo_assemblies = len(assemblies_list[0])
	print("The number of Russo assemblies being analyzed for stimulus " + str(stimuli_iter+1) + " is " + str(number_Russo_assemblies))

	#Initialize array to hold results from iterating the epsilon value; the final value corresponds to the total number of steps that are taken, as an integer value
	epsilon_results = np.empty([params['number_stimuli'], params['number_stimuli'], number_Russo_assemblies, int((params['epsilon_max']-params['epsilon0'])/params['epsilon_iter_step'])]) 
	information_theory_data = np.empty([params['number_stimuli'], number_Russo_assemblies])

	#Iterate through each data set; note the stimulus file names are indexed from 1
	for dataset_iter in range (params['number_stimuli']):
		if params['shuffle_Boolean'] == True:
			spike_data = np.genfromtxt('./Processing_Data/shuffled_posttraining_stim' + str(dataset_iter+1) + '_layer' + str(params['network_layer']) +'_Russo.csv', delimiter=',') 
		else:
			spike_data = np.genfromtxt('./Processing_Data/posttraining_stim' + str(dataset_iter+1) + '_layer' + str(params['network_layer']) + '_Russo.csv', delimiter=',') 

		#Iterate through each assembly
		for assembly_iter in range(number_Russo_assemblies):
			#Extract the neuron indeces and time delays for the current assembly of interest
			#Notes on the below code: #list/map/int converts the values into int; IDs - 1 changes indexing from 1 (Matlab) to 0
			Russo_assembly_ids = [IDs - 1 for IDs in list(map(int, assemblies_list[0][assembly_iter]))]
			Russo_assembly_times = [lags * params['Russo_bin_size'] for lags in assemblies_list[1][assembly_iter]]

			if params['templates_alt_stream_bool'] == True:
				if check_assembly_side(params, Russo_assembly_ids) == True:
					for ii in range(len(Russo_assembly_ids)):
						Russo_assembly_ids[ii] = Russo_assembly_ids[ii] + dataset_iter * params['group_dim']

			#Search for activations of an assembly
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				(activation_array, number_candidate_assemblies) = find_assembly_activations(params, Russo_assembly_times, Russo_assembly_ids, spike_data, epsilon=params['epsilon0'])

			if params['information_theory_bool'] == True:
				#Scan for the activation of each assembly during a particular stimulus presentaiton
				information_theory_data[dataset_iter, assembly_iter] = information_theory_scanning(params, activation_array)

			if params['epsilon_iter_bool'] == True:
				#Run the specific 'epsilon analysis', where increasing values of epsilon are used for eventual plotting
				epsilon_results = analysis_epsilon(params, Russo_assembly_times, Russo_assembly_ids, spike_data, number_candidate_assemblies,
					epsilon_results, dataset_iter, stimuli_iter, assembly_iter)

	if params['information_theory_bool'] == True:

		#Use this activation data to calculate the information (bits) that each assembly carries about the stimulus presented
		information_theory_results = information_theory_calculation(params, information_theory_data)

		fig, ax = plt.subplots()
		plt.hist(information_theory_results, bins=np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
		ax.set_ylabel('Number of Assemblies')
		ax.set_xlabel('Information (bits)')
		plt.savefig('./Processing_Data/PNG_information_content_stim' + str(stimuli_iter+1) + '_layer' + str(params['network_layer']) +'.png')
		plt.show()

		print("The number of assemblies that carried >0.80 bit of information was : " + str(np.sum(information_theory_results>=0.80)))

		UnitTest_information_theory_calculation(params, information_theory_data)


	if params['epsilon_iter_bool'] == True:

		dataset_iter = 0 # *** temporary fixing of this variable

		fig, ax = plt.subplots()
		epsilon_x_axis = np.arange(1, len(epsilon_results[dataset_iter, stimuli_iter, 0, :])+1)*params['epsilon_iter_step']*1000

		for assembly_iter in range(0, number_Russo_assemblies):
			plt.scatter(epsilon_x_axis, epsilon_results[dataset_iter, stimuli_iter, assembly_iter, :])
		
		ax.set_ylim(0, 1)
		ax.set_ylabel('Proportion of Assembly Activations')
		ax.set_xlabel('Epsilon (ms)')
		plt.savefig('./Processing_Data/Epsilon_curves_stim' + str(stimuli_iter+1) + '_layer' + str(params['network_layer']) +'.png')
		plt.show()

	return 0


#Check that an assembly isn't already ascribed to the alternative side of the stream, in which case discount it for the 'templates_alt_stream_bool' search
def check_assembly_side(params, Russo_assembly_ids): 
	for jj in range(len(Russo_assembly_ids)):
		if Russo_assembly_ids[jj] >= params['group_dim']:
			return False
	return True


#Import assemblies extracted from the Russo-algorithm
def import_assemblies(params, stimuli_iter):
	#Load the Matlab data file
	#Note stimuli are index from 0 in Python, but from 1 in the file names/simulations
	with open('./Processing_Data/Russo_extracted_assemblies_stim' + str(stimuli_iter+1) + '_layer' + str(params['network_layer']) + '.data', 'rb') as filehandle:
		assemblies_list = pickle.load(filehandle)
	
	return assemblies_list


#Iterate through candidate activations of an assembly; any time the first neuron of the prescribed assembly spikes, it is considered a candidate activation; defined as a separate function to assist appropriate warning handling
def find_assembly_activations(params, Russo_assembly_times, Russo_assembly_ids, spike_data, epsilon):

	warnings.warn("NaN_Comparison", RuntimeWarning) #Prevents the NaN comparison below from generating a run-time error
	number_candidate_assemblies = np.sum(spike_data[Russo_assembly_ids[0], :] >= 0)

	candidate_activations = spike_data[Russo_assembly_ids[0], 0:number_candidate_assemblies] #Array of spike times when the first neuron in the assembly spikes

	#Create the upper and lower bounds
	(upper_bound_array, lower_bound_array) = create_boundaries(epsilon, Russo_assembly_times, candidate_activations, number_candidate_assemblies, synchrony_bool=params['synchrony_bool'])

	#Assess if assemblies were active based on the now-defined boundaries
	activation_array = evaluate_assembly_activation(Russo_assembly_times, Russo_assembly_ids, spike_data, upper_bound_array, lower_bound_array, number_candidate_assemblies, candidate_activations)

	return (activation_array, number_candidate_assemblies)


#Create the upper and lower limits of the spike times, in vectorized format
def create_boundaries(epsilon, Russo_assembly_times, candidate_activations, number_candidate_assemblies, synchrony_bool):
	
	#If checking for information/precision in a 'synchronous assembly' (i.e. assuming lags between neuron spikes were not significant), set assembly times to a zero vector
	if synchrony_bool == True:
		Russo_assembly_times[1:] = np.zeros(len(Russo_assembly_times[1:]))

	#Notes on the below - np.reshape enables broadcasting, which is sotherwise prevented by arrays having shape (n,) rather than (n,1)
	#Also note the first assembly neuron is not included in the boundary arrays, as it is by definition within the boundary
	upper_bound_array = (np.broadcast_to(candidate_activations, (len(Russo_assembly_times[1:]), number_candidate_assemblies)) 
		+ np.reshape(Russo_assembly_times[1:], (-1,1)) + epsilon/2)
	lower_bound_array = (np.broadcast_to(candidate_activations, (len(Russo_assembly_times[1:]), number_candidate_assemblies)) 
		+ np.reshape(Russo_assembly_times[1:], (-1,1)) - epsilon/2) 

	return (upper_bound_array, lower_bound_array)


#Vectorized assessment of whether a set of candidate activations exist within the spike data, determined by the upper and lower bounds
def evaluate_assembly_activation(Russo_assembly_times, Russo_assembly_ids, spike_data, upper_bound_array, lower_bound_array, number_candidate_assemblies, candidate_activations):

	activation_array = np.empty([2, number_candidate_assemblies]) #Stores whether an assembly has activated, and at what time

	#Create an array to store booleans for each assembly neuron (excluding the first) in each candidate activation
	activation_bool_array = np.empty([len(Russo_assembly_ids[1:]), number_candidate_assemblies])

	#Reshape the spike data so that broadcasting can be used during the application of the array-format boundaries
	reshaped_spike_data = np.transpose(np.broadcast_to(spike_data[Russo_assembly_ids[1:],:], 
			(number_candidate_assemblies, len(Russo_assembly_ids[1:]), np.shape((spike_data[Russo_assembly_ids[1:],:]))[1])))
	
	activation_bool_array = np.any((reshaped_spike_data <= upper_bound_array) & (reshaped_spike_data >= lower_bound_array), axis=0)

	#For each candidate activation, check if all the neurons in that assembly were active, in which case return a 1
	activation_bool = np.all(activation_bool_array, axis=0)
	activation_array[0, :] = activation_bool
	activation_array[1, :] = np.multiply(activation_bool, candidate_activations) #Stores the times of assemblies that were active

	return activation_array


def analysis_epsilon(params, Russo_assembly_times, Russo_assembly_ids, spike_data, number_candidate_assemblies,
			epsilon_results, dataset_iter, stimuli_iter, assembly_iter):

	epsilon = params['epsilon_iter_step']

	#Iterate through each value of epsilon
	for ii in range(0, len(epsilon_results[dataset_iter, stimuli_iter, assembly_iter, :])):

		activations_count = 0
		epsilon = epsilon + params['epsilon_iter_step'] #How many ms epsilon is iterated by 

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			(activation_array, number_candidate_assemblies) = find_assembly_activations(params, Russo_assembly_times, Russo_assembly_ids, spike_data, epsilon)

		epsilon_results[dataset_iter, stimuli_iter, assembly_iter, ii] = np.sum(activation_array[0,:])/number_candidate_assemblies

	return epsilon_results


def information_theory_scanning(params, activation_array):
	#For each stimulus, given that it has been presented, determine whether each assembly was active
	activation_counter = 0
	for ii in range(0, params['number_of_presentations']):
	#Extract from assembly_activations if the assembly had any activations in that interval, and if it did then record a 1, 0 otherwise
		activation_counter += np.any((activation_array[1, :] >= (ii*params['duration_of_presentations'])) & (activation_array[1, :] < ((ii+1)*params['duration_of_presentations'])))
	
	#Final activation_counter provides a value containing the number of presentations when the assembly was active
	return activation_counter


# Test information theory calculation by analysing idealised data
def UnitTest_information_theory_calculation(params, information_theory_data):
		temp_information_theory_data = information_theory_data #Copy of information theory data

		#Set every assembly (second dimension) to be active for every presentation of stimulus 1 (first dimension) only
		temp_information_theory_data[0, :] = params['number_of_presentations']
		temp_information_theory_data[1, :] = 0
		temp_information_theory_results = information_theory_calculation(params, temp_information_theory_data)
		assert np.sum(temp_information_theory_results>=0.80) == len(temp_information_theory_results), "Unit Test Failure: Idealized data does not have perfect information."

		#Set every assembly to be active for presentation of stimulus 2 only
		temp_information_theory_data[0, :] = 0
		temp_information_theory_data[1, :] = params['number_of_presentations']
		temp_information_theory_results = information_theory_calculation(params, temp_information_theory_data)
		assert np.sum(temp_information_theory_results>=0.80) == len(temp_information_theory_results), "Unit Test Failure: Idealized data does not have perfect information."

		# Test in a case of no information (activation for either stimulus equally likely)
		temp_information_theory_data[0, :] = params['number_of_presentations']/2
		temp_information_theory_data[1, :] = params['number_of_presentations']/2
		temp_information_theory_results = information_theory_calculation(params, temp_information_theory_data)
		assert np.sum(temp_information_theory_results>=0.80) == 0, "Unit Test Failure: Artificially uninformative data still has information."

		return None


def information_theory_calculation(params, information_theory_data):

	#Information_theory_data is indexed by [dataset_iter, assembly_iter]; thus the row indicates which stimulus was presented, and the 
	#column value indicates how many presentations were associated with at least one activation of that assembly

	no_math_error = 0.00000001 #Prevent division by zero

	#The probabilities of a particular assembly being active for each stimulus
	conditional_prob_array = information_theory_data/params['number_of_presentations']
	marginal_prob_array = np.sum(information_theory_data, axis=0)/(params['number_of_presentations']*params['number_stimuli'])

	information1 = np.multiply(conditional_prob_array[0, :], np.log2(np.divide(conditional_prob_array[0, :], marginal_prob_array)+no_math_error))
	information2 = np.multiply(1-conditional_prob_array[0, :], np.log2(np.divide(1-conditional_prob_array[0, :], (1-marginal_prob_array+no_math_error))+no_math_error))

	information_theory_results = information1+information2

	return information_theory_results


main(params)
