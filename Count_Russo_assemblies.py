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
#Using the extracted assemblies, the following algorithm will count the number of instances of each assembly, relative to a particular stimulus



#Definitions of parameters:
#epsilon0 : primary cut-off margin for spike times in assembly search, given in seconds; note it falls covers both sides of the prescribed spike times, so an epsilon of 3ms corresponds to a bin/window of 3ms
#Russo_bin_size
#number_stimuli
#dataset_duration : length of each data-set, in seconds

params = {'epsilon0' : 0.0005,
	'Russo_bin_size' : 0.003,
	'number_stimuli' : 2,
	'network_layer': 3,
	'dataset_duration' : 50,
	'epsilon_iter_bool' : 1,
	'epsilon_iter_step' : 0.00025,
	'epsilon_max' : 0.015,
	'shuffle_Boolean' : 0,
	'Poisson_Boolean' : 0,
	'epsilon_plotting_Boolean' : 1,
	'comparative_plotting_Boolean' : 0}


def main(params):


	stimuli_iter = 0

	assemblies_list = import_assemblies(params, stimuli_iter)
	number_Russo_assemblies = 20 #len(assemblies_list[0])

	#Initialize array to hold main analysis results; final value relates to the number of analysis metrics that are used
	analysis_results = np.empty([params['number_stimuli'], params['number_stimuli'], number_Russo_assemblies, 12])
	#The first dimension of analysis_results is the stimuli associated with a particular dataset, while the second is the stimuli associated with the Russo-extracted assemblies

	#Initialize array to hold results from iterating the epsilon value; the final value corresponds to the total number of steps that are taken, as an integer value
	epsilon_results = np.empty([params['number_stimuli'], params['number_stimuli'], number_Russo_assemblies, int((params['epsilon_max']-params['epsilon0'])/params['epsilon_iter_step'])]) 

	#Iterate through each data set; note the stimuli file names are indexed from 1
	for dataset_iter in range (0, params['number_stimuli']):
		if params['shuffle_Boolean'] == 1:
			#If shuffle_Boolean is set to 1, then loads shuffled data
			spike_data = np.genfromtxt('shuffled_posttraining_stim' + str(dataset_iter+1) + '_layer' + str(params['network_layer']) +'_Russo.csv', delimiter=',') 
		elif params['Poisson_Boolean'] == 1:
			spike_data = np.genfromtxt('Poisson_spikes_stim1.csv', delimiter=',')
		else:
			spike_data = np.genfromtxt('posttraining_stim' + str(dataset_iter+1) + '_layer' + str(params['network_layer']) + '_Russo.csv', delimiter=',') 


		#Iterate through each stimulus



		#Iterate through each assembly
		for assembly_iter in range(0, number_Russo_assemblies):
			#Within each assembly, define the neuron indeces that actually compose it, as well as their idealized spike times


			#Extract the neuron indeces and time delays for the current assembly of interest
			#Notes on the below code: #list/map/int converts the values into int; IDs - 1 changes indexing from 1 (Matlab) to from 0 (Python)
			Russo_assembly_ids = [IDs - 1 for IDs in list(map(int, assemblies_list[0][assembly_iter]))]
			Russo_assembly_times = [lags * params['Russo_bin_size'] for lags in assemblies_list[1][assembly_iter]]


			#Search for activations of an assembly (with both the primary and broad epsilon)
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")

				(activation_array, number_candidate_assemblies) = find_assembly_activations(params, Russo_assembly_times, Russo_assembly_ids, spike_data, epsilon=params['epsilon0'])

			#Run analyses that are performed for an assembly in a single dataset
			analysis_results = analysis_metrics(params, Russo_assembly_times, Russo_assembly_ids, spike_data, activation_array, 
				analysis_results, dataset_iter, stimuli_iter, assembly_iter)

			if params['epsilon_iter_bool'] == 1:
				#Run the specific 'epsilon analysis', where increasing values of epsilon are used for eventual plotting
				epsilon_results = analysis_epsilon(params, Russo_assembly_times, Russo_assembly_ids, spike_data, number_candidate_assemblies, analysis_results,
					epsilon_results, dataset_iter, stimuli_iter, assembly_iter)
		


	# *** temporary fixing of this variable
	dataset_iter = 0



	if params['epsilon_plotting_Boolean'] == 1:

		epsilon_x_axis = np.arange(1, len(epsilon_results[dataset_iter, stimuli_iter, 0, :])+1)*params['epsilon_iter_step']*1000

		for assembly_iter in range(0, number_Russo_assemblies):
			plt.scatter(epsilon_x_axis, epsilon_results[dataset_iter, stimuli_iter, assembly_iter, :])
		
		plt.show()
		
		#Output raw results as a CSV file
		with open('epsilon_results.data', 'wb') as filehandle:
			pickle.dump(epsilon_results, filehandle)


	#Run analyses that are performed to compare the activity of an assembly across datasets
	difference_in_assembly_counts = comparative_metrics(params, analysis_results, stimuli_iter, number_Russo_assemblies)
	
	if params['comparative_plotting_Boolean'] == 1:
		#Comparative plotting
		comparative_x_axis = np.arange(0, number_Russo_assemblies)

		plt.scatter(comparative_x_axis, analysis_results[0, stimuli_iter, :, 0], label='Stimulus 1 presentation')
		plt.scatter(comparative_x_axis, analysis_results[1, stimuli_iter, :, 0], label='Stimulus 2 presentation')
		plt.scatter(comparative_x_axis, difference_in_assembly_counts, label='Difference in counts')

		plt.title('Number of assembly occurences')
		plt.legend()
		plt.show()


	#Information theory analysis, using the times at which the assemblies occur



	#Output information theory analysis

	

	return 0

#Import assemblies extracted from the Russo-algorithm
def import_assemblies(params, stimuli_iter):
	#Load the Matlab data file
	
	#Note stimuli are index from 0 in Python, but from 1 in the file names/simulations
	with open('Russo_extracted_assemblies_stim' + str(stimuli_iter+1) + '_layer' + str(params['network_layer']) + '.data', 'rb') as filehandle:
		assemblies_list = pickle.load(filehandle)
	
	return assemblies_list


#Iterate through candidate activations of an assembly; any time the first neuron of the prescribed assembly spikes, it is considered a candidate activation; defined as a separate function to assist appropriate warning handling
def find_assembly_activations(params, Russo_assembly_times, Russo_assembly_ids, spike_data, epsilon):

	warnings.warn("NaN_Comparison", RuntimeWarning) #Prevents the NaN comparison below from generating a run-time error
	number_candidate_assemblies = np.sum(spike_data[Russo_assembly_ids[0], :] >= 0)

	candidate_activations = spike_data[Russo_assembly_ids[0], 0:number_candidate_assemblies] #Array of spike times when the first neuron in the assembly spikes

	#Create the upper and lower bounds
	(upper_bound_array, lower_bound_array) = create_boundaries(epsilon, Russo_assembly_times, candidate_activations, number_candidate_assemblies)

	#Assess if assemblies were active based on the now-defined boundaries
	activation_array = evaluate_assembly_activation(Russo_assembly_times, Russo_assembly_ids, spike_data, upper_bound_array, lower_bound_array, number_candidate_assemblies, candidate_activations)

	return (activation_array, number_candidate_assemblies)


#Create the upper and lower limits of the spike times, in vectorized format
def create_boundaries(epsilon, Russo_assembly_times, candidate_activations, number_candidate_assemblies):
	
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


def analysis_metrics(params, Russo_assembly_times, Russo_assembly_ids, spike_data, activation_array, analysis_results, dataset_iter, stimuli_iter, assembly_iter):
	
	### Results using primary epsilon ###

	#Total count of assembly activations
	analysis_results[dataset_iter, stimuli_iter, assembly_iter, 0] = np.sum(activation_array[0, :])

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		number_assembly_neuron_spikes = count_assembly_neuron_spikes(Russo_assembly_ids, spike_data)
	
	#Spike count of first neuron in the assembly
	analysis_results[dataset_iter, stimuli_iter, assembly_iter, 1] = number_assembly_neuron_spikes[0]

	#Spike count of the maximally firing neuron in the assembly
	analysis_results[dataset_iter, stimuli_iter, assembly_iter, 2] = np.amax(number_assembly_neuron_spikes)

	#Spike count of the minimally firing neuron in the assembly
	analysis_results[dataset_iter, stimuli_iter, assembly_iter, 3] = np.amin(number_assembly_neuron_spikes)

	#Ratio of assembly activations to spiking of the first neuron in the assembly
	analysis_results[dataset_iter, stimuli_iter, assembly_iter, 4] = (analysis_results[dataset_iter, stimuli_iter, assembly_iter, 0] /
		analysis_results[dataset_iter, stimuli_iter, assembly_iter, 1])

	#Ratio of assembly activations to maximally spiking neuron in the assembly
	analysis_results[dataset_iter, stimuli_iter, assembly_iter, 5] = (analysis_results[dataset_iter, stimuli_iter, assembly_iter, 0] /
		analysis_results[dataset_iter, stimuli_iter, assembly_iter, 2])

	#Ratio of assembly activations to minimally spiking neuron in the assembly
	analysis_results[dataset_iter, stimuli_iter, assembly_iter, 6] = (analysis_results[dataset_iter, stimuli_iter, assembly_iter, 0] /
		analysis_results[dataset_iter, stimuli_iter, assembly_iter, 3])

	#Average firing rate of first neuron in the assembly
	analysis_results[dataset_iter, stimuli_iter, assembly_iter, 7] = (analysis_results[dataset_iter, stimuli_iter, assembly_iter, 1] /
		params['dataset_duration'])

	#Average firing rate of maximally firing neuron in the assembly
	analysis_results[dataset_iter, stimuli_iter, assembly_iter, 8] = (analysis_results[dataset_iter, stimuli_iter, assembly_iter, 2] /
		params['dataset_duration'])

	#Average firing rate of minimally firing neuron in the assembly
	analysis_results[dataset_iter, stimuli_iter, assembly_iter, 9] = (analysis_results[dataset_iter, stimuli_iter, assembly_iter, 3] /
		params['dataset_duration'])


	return analysis_results


#Counts how many times the neurons in a particular assembly spike in the data-set, regardless of how many of these spikes are related to an assembly activation
def count_assembly_neuron_spikes(Russo_assembly_ids, spike_data):
	
	warnings.warn("NaN_Comparison", RuntimeWarning) #Prevents the NaN comparison below from generating a run-time error
	number_assembly_neuron_spikes = np.sum(spike_data[Russo_assembly_ids, :] >= 0, axis=1)

	return number_assembly_neuron_spikes

def comparative_metrics(params, analysis_results, stimuli_iter, number_Russo_assemblies):

	#Subtract the firing counts of each assembly for the two different stimuli from each other
	#The stimulus of interest should be the first term in the subtraction
	difference_in_assembly_counts = analysis_results[0, stimuli_iter, :, 0] - analysis_results[1, stimuli_iter, :, 0]

	return difference_in_assembly_counts

def analysis_epsilon(params, Russo_assembly_times, Russo_assembly_ids, spike_data, number_candidate_assemblies, analysis_results,
			epsilon_results, dataset_iter, stimuli_iter, assembly_iter):

	epsilon = params['epsilon0']

	#Iterate through each value of epsilon
	for ii in range(0, len(epsilon_results[dataset_iter, stimuli_iter, assembly_iter, :])):

		activations_count = 0
		epsilon = epsilon + params['epsilon_iter_step'] #How many ms epsilon is iterated by 

		(activation_array, number_candidate_assemblies) = find_assembly_activations(params, Russo_assembly_times, Russo_assembly_ids, spike_data, epsilon)

		epsilon_results[dataset_iter, stimuli_iter, assembly_iter, ii] = np.sum(activation_array[0,:])/number_candidate_assemblies

	return epsilon_results

main(params)

#Output all the assemblies that were able to be separated in the training data-set by a factor of e.g. 80%

#Check performance of these assemblies with the learned parameters on a hold-out dataset, before making comments on information content (otherwise can argue that have just over-fit the data to have high information)

