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
#epsilon0 : primary cut-off margin for spike times in assembly search, given in seconds; note it falls on either side of the prescribed spike times, so an epsilon of 1.5ms corresponds to a window of 3ms
#Russo_bin_size
#number_stimuli
#epsilon1 : broad cut-off margin for spike times in assembly search, given in seconds; used to evaluate how much increasing epsilon changes the number of assemblies that are identified
#dataset_duration : length of each data-set, in seconds

params = {'epsilon0' : 0.00125,
	'Russo_bin_size' : 0.003,
	'number_stimuli' : 2,
	'network_layer': 3,
	'epsilon1' : 0.015,
	'dataset_duration' : 50,
	'epsilon_iter_bool' : 0,
	'epsilon_iter_step' : 0.0005,
	'epsilon_max' : 0.015,
	'shuffle_Boolean' : 0,
	'Poisson_Boolean' : 0,
	'epsilon_plotting_Boolean' : 0,
	'comparative_plotting_Boolean' : 1}


def main(params):


	stimuli_iter = 0

	assemblies_list = import_assemblies(params, stimuli_iter)
	number_Russo_assemblies = 200 #len(assemblies_list[0])


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

		tic = time.time()
		
		#Iterate through each assembly
		for assembly_iter in range(0+2500, number_Russo_assemblies+2500):
			#Within each assembly, define the neuron indeces that actually compose it, as well as their idealized spike times

			#Extract the neuron indeces and time delays for the current assembly of interest
			Russo_assembly_ids = [IDs - 1 for IDs in list(map(int, assemblies_list[0][assembly_iter]))] #list/map/int converts the values into int, IDs - 1 changes indexing from 1 (Matlab) to from 0 (Python)
			Russo_assembly_times = [lags * params['Russo_bin_size'] for lags in assemblies_list[1][assembly_iter]]

			#Search for activations of an assembly (with both the primary and broad epsilon)
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				(activation_array, number_candidate_assemblies) = iterate_through_candidates(params, Russo_assembly_times, Russo_assembly_ids, spike_data)

			#Run analyses that are performed for an assembly in a single dataset
			analysis_results = analysis_metrics(params, Russo_assembly_times, Russo_assembly_ids, spike_data, activation_array, 
				analysis_results, dataset_iter, stimuli_iter, assembly_iter - 2500)

			#print(analysis_results[dataset_iter, stimuli_iter, assembly_iter, :])

			#Run the specific 'epsilon analysis', where increasing values of epsilon are used for eventual plotting
			epsilon_results = analysis_epsilon(params, Russo_assembly_times, Russo_assembly_ids, spike_data, number_candidate_assemblies, analysis_results,
				epsilon_results, dataset_iter, stimuli_iter, assembly_iter - 2500)
		
		toc = time.time() - tic
		print(toc)

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

		plt.scatter(comparative_x_axis, difference_in_assembly_counts, label='Difference in counts')
		plt.scatter(comparative_x_axis, analysis_results[0, stimuli_iter, :, 0], label='Stimulus 1 presentation')
		plt.scatter(comparative_x_axis, analysis_results[1, stimuli_iter, :, 0], label='Stimulus 2 presentation')

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


#Create simple dataset for integration testing
def create_example_dataset(Russo_assembly_times, Russo_assembly_ids):


	# ** important that test example ultimately includes multiple occurences of an assembly **

	spike_data = np.empty((10, 5))
	spike_data[:] = np.NaN

	#Create an array of 'ground truth' candidate activations, which indexed from 1-4 are increasingly poor examples of a true activation
	#Also assigns a spike time for the first neuron in the assembly, to test whether this can be successfully recovered
	candidate_activations = np.empty((4, 5))

	for ii in range(0,4):
		candidate_activations[ii, :] = Russo_assembly_times + ii*0.02 #Off-set the beginning of the assembly activation by a certain amount of time
		candidate_activations[ii, 1:] = candidate_activations[ii, 1:] + ii*0.002*np.power(-1, ii) #Off-set the spike times of all assembly neurons other than the first by a certain ammount (essentially non-random 'noise'); offset alternates between being negative and positive
		#print(candidate_activations[ii])
		spike_data[Russo_assembly_ids, ii] = np.transpose(candidate_activations[ii, :])

	#For neurons that did not spike, assign them arbitrary values
	non_spiked_ids = np.array([2, 3, 6, 7, 9]) - 1
	spike_data[non_spiked_ids, 0] = 5


	return spike_data

#Iterate through candidate activations of an assembly; any time the first neuron of the prescribed assembly spikes, it is considered a candidate activation; defined as a separate function to assist appropriate warning handling
def iterate_through_candidates(params, Russo_assembly_times, Russo_assembly_ids, spike_data):


	warnings.warn("NaN_Comparison", RuntimeWarning) #Prevents the NaN comparison below from generating a run-time error
	number_candidate_assemblies = np.sum(spike_data[Russo_assembly_ids[0], :] >= 0)
	#print(number_candidate_assemblies)

	activation_array = np.empty([2, 2, number_candidate_assemblies]) #Tracks whethr an assembly has activated, and at what time
	#First dimension of activation array determines whether it is for the primary or broad epsilon

	#Iterate through the primary (0) and the broad epsilon (1)
	for ii in range(0, 2):

		epsilon = params['epsilon' + str(ii)]

		for jj in range(0, number_candidate_assemblies):
			first_neuron_spike_time = spike_data[Russo_assembly_ids[0], jj]
			(upper_bound, lower_bound) = create_boundaries(epsilon, Russo_assembly_times, first_neuron_spike_time)
			activation_array[ii, 0, jj] = evaluate_assembly_activation(Russo_assembly_times, Russo_assembly_ids, spike_data, upper_bound, lower_bound)
			activation_array[ii, 1, jj] = first_neuron_spike_time*activation_array[ii, 0, jj] #A non-zero value for the time of assembly activation is only assigned if the assembly activated

	return (activation_array, number_candidate_assemblies)

#Create the upper and lower limits of the spike times
def create_boundaries(epsilon, Russo_assembly_times, first_neuron_spike_time):
	
	#Note the first assembly neuron is not included in the boundary arrays, as it is by definition within the boundary
	upper_bound = Russo_assembly_times[1:] + first_neuron_spike_time + epsilon
	lower_bound = Russo_assembly_times[1:] + first_neuron_spike_time - epsilon

	return (upper_bound, lower_bound)

#Non-vectorized implementation; the function will return a 1 if all spike times fall in the prescribed range, 0 otherwise
def evaluate_assembly_activation(Russo_assembly_times, Russo_assembly_ids, spike_data, upper_bound, lower_bound):

	bool_iter = 1
	#Note in the iteration, the first neuron in the Russo assembly is skipped, as by definition this is active at the necessary time
	for kk in range(0, len(upper_bound)):
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			warnings.warn("NaN_Comparison", RuntimeWarning) #Prevents the NaN comparison below from generating a run-time error
			bool_iter = bool_iter * np.any((spike_data[Russo_assembly_ids[kk+1],:] <= upper_bound[kk]) & (spike_data[Russo_assembly_ids[kk+1],:] >= lower_bound[kk]))
	
	return bool_iter

def analysis_metrics(params, Russo_assembly_times, Russo_assembly_ids, spike_data, activation_array, analysis_results, dataset_iter, stimuli_iter, assembly_iter):
	
	### Results using primary epsilon ###

	#Total count of assembly activations
	analysis_results[dataset_iter, stimuli_iter, assembly_iter, 0] = np.sum(activation_array[0, 0, :])

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


	### Results using broad epsilon ###

	#Epsilon sensitivity: ratio of assemblies captured with broad epsilon
	#Ratio of assembly activations to spiking of the first neuron in the assembly
	analysis_results[dataset_iter, stimuli_iter, assembly_iter, 10] = (np.sum(activation_array[1, 0, :]) /
		number_assembly_neuron_spikes[0])

	#Difference between ratio of assembly activations for primary vs broad epsilon 
	analysis_results[dataset_iter, stimuli_iter, assembly_iter, 11] = (analysis_results[dataset_iter, stimuli_iter, assembly_iter, 10] - 
		analysis_results[dataset_iter, stimuli_iter, assembly_iter, 4])

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

		#Iterate through the number of candidate assemblies
		for jj in range(0, number_candidate_assemblies):
			first_neuron_spike_time = spike_data[Russo_assembly_ids[0], jj]
		
			#Call the boundary making function
			(upper_bound, lower_bound) = create_boundaries(epsilon, Russo_assembly_times, first_neuron_spike_time)

			#Call the assembly assembly activation evaluation function
			activations_count = activations_count + evaluate_assembly_activation(Russo_assembly_times, Russo_assembly_ids, spike_data, upper_bound, lower_bound)
		
		#Update the value of the epsilon_results array
		epsilon_results[dataset_iter, stimuli_iter, assembly_iter, ii] = activations_count/number_candidate_assemblies

	return epsilon_results

main(params)

#Output all the assemblies that were able to be separated in the training data-set by a factor of e.g. 80%

#Check performance of these assemblies with the learned parameters on a hold-out dataset, before making comments on information content (otherwise can argue that have just over-fit the data to have high information)

