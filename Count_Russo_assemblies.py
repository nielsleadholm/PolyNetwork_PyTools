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
#number_of_presentations : how many times each stimulus is presented in the datasets
#duration_of_presentations : length of each stimulus presentation, in seconds
#synchrony bool : check whether, if all lags detected by the Russo algorithm for a given assembly are treated as zero, how does this effect e.g. information content
#templates_alt_stream_bool: check whether, if using the Russo assembly templates on the alternative stream, but where the neuron indices are corrected for that stream, how much information is present

params = {'epsilon0' : 0.005,
	'Russo_bin_size' : 0.003,
	'number_stimuli' : 2,
	'network_layer': 3,
	'group_dim': 16*16,
	'number_of_presentations' : 50,
	'duration_of_presentations' : 0.2,
	'epsilon_iter_step' : 0.00025,
	'epsilon_max' : 0.015,
	'shuffle_Boolean' : 0,
	'Poisson_Boolean' : 0,
	'synchrony_bool' : 0,
	'templates_alt_stream_bool' : False,
	'comparative_plotting_Boolean' : 0,
	'epsilon_iter_bool' : True,
	'information_theory_bool' : False}


def main(params):


	stimuli_iter = 0

	assemblies_list = import_assemblies(params, stimuli_iter)
	number_Russo_assemblies = len(assemblies_list[0])
	print(number_Russo_assemblies)
	off_set_assembly_index = 0 # *** temporary variable to help look at different subsets of the  Russo assemblies

	#Initialize array to hold main analysis results; final value relates to the number of analysis metrics that are used
	analysis_results = np.empty([params['number_stimuli'], params['number_stimuli'], number_Russo_assemblies, 12])
	#The first dimension of analysis_results is the stimuli associated with a particular dataset, while the second is the stimuli associated with the Russo-extracted assemblies

	#Initialize array to hold results from iterating the epsilon value; the final value corresponds to the total number of steps that are taken, as an integer value
	epsilon_results = np.empty([params['number_stimuli'], params['number_stimuli'], number_Russo_assemblies, int((params['epsilon_max']-params['epsilon0'])/params['epsilon_iter_step'])]) 

	information_theory_data = np.empty([params['number_stimuli'], number_Russo_assemblies])

	#Iterate through each data set; note the stimuli file names are indexed from 1
	for dataset_iter in range (0, params['number_stimuli']):
		if params['shuffle_Boolean'] == 1:
			#If shuffle_Boolean is set to 1, then loads shuffled data
			spike_data = np.genfromtxt('./Processing_Data/shuffled_posttraining_stim' + str(dataset_iter+1) + '_layer' + str(params['network_layer']) +'_Russo.csv', delimiter=',') 
		elif params['Poisson_Boolean'] == 1:
			spike_data = np.genfromtxt('./Processing_Data/Poisson_spikes_stim1.csv', delimiter=',')
		else:
			spike_data = np.genfromtxt('./Processing_Data/posttraining_stim' + str(dataset_iter+1) + '_layer' + str(params['network_layer']) + '_Russo.csv', delimiter=',') 


		#Iterate through each stimulus



		#Iterate through each assembly
		for assembly_iter in range(0, number_Russo_assemblies):
			#Within each assembly, define the neuron indeces that actually compose it, as well as their idealized spike times


			#Extract the neuron indeces and time delays for the current assembly of interest
			#Notes on the below code: #list/map/int converts the values into int; IDs - 1 changes indexing from 1 (Matlab) to from 0 (Python)
			Russo_assembly_ids = [IDs - 1 for IDs in list(map(int, assemblies_list[0][assembly_iter + off_set_assembly_index]))]
			Russo_assembly_times = [lags * params['Russo_bin_size'] for lags in assemblies_list[1][assembly_iter + off_set_assembly_index]]

			#print(Russo_assembly_ids)
			if params['templates_alt_stream_bool'] == 1:
				if check_assembly_side(params, Russo_assembly_ids) == True:
					for ii in range(len(Russo_assembly_ids)):
						Russo_assembly_ids[ii] = Russo_assembly_ids[ii] + dataset_iter * params['group_dim']
			#print(Russo_assembly_ids)


			#Search for activations of an assembly
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				(activation_array, number_candidate_assemblies) = find_assembly_activations(params, Russo_assembly_times, Russo_assembly_ids, spike_data, epsilon=params['epsilon0'])


			information_theory_data[dataset_iter, assembly_iter] = information_theory_counting(params, activation_array)

			#Run analyses that are performed for an assembly in a single dataset
			analysis_results = analysis_metrics(params, Russo_assembly_times, Russo_assembly_ids, spike_data, activation_array, 
				analysis_results, dataset_iter, stimuli_iter, assembly_iter)

			if params['epsilon_iter_bool'] == 1:
				#Run the specific 'epsilon analysis', where increasing values of epsilon are used for eventual plotting
				epsilon_results = analysis_epsilon(params, Russo_assembly_times, Russo_assembly_ids, spike_data, number_candidate_assemblies, analysis_results,
					epsilon_results, dataset_iter, stimuli_iter, assembly_iter)
		


	if params['information_theory_bool'] == 1:

		info_timer = time.time()

		information_theory_results = information_theory_calculation(params, information_theory_data)

		info_timer_total = time.time() - info_timer
		print(info_timer_total)


		fig, ax = plt.subplots()

		plt.hist(information_theory_results, bins=np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
		#ax.set_xlim(0, 1)
		ax.set_ylabel('Number of Assemblies')
		ax.set_xlabel('Information (bits)')

		plt.savefig('./Processing_Data/PNG_information_content_stim' + str(stimuli_iter+1) + '_layer' + str(params['network_layer']) +'.png')
		plt.show()

		# *** test that information theory calculation is working by confirming that an idealized activation_array results in 1 bit of information
		# information_theory_data[0, :] = params['number_of_presentations']
		# information_theory_data[1, :] = 0

		# information_theory_results = information_theory_calculation(params, information_theory_data)

		# information_theory_data[0, :] = 0
		# information_theory_data[1, :] = params['number_of_presentations']

		# information_theory_results = information_theory_calculation(params, information_theory_data)

		# # *** test in a case of no information (i.e. assembly activation is essentially a coin toss)
		# information_theory_data[0, :] = params['number_of_presentations']/2
		# information_theory_data[1, :] = params['number_of_presentations']/2

		# information_theory_results = information_theory_calculation(params, information_theory_data)


	# *** temporary fixing of this variable
	dataset_iter = 0



	if params['epsilon_iter_bool'] == 1:

		fig, ax = plt.subplots()

		epsilon_x_axis = np.arange(1, len(epsilon_results[dataset_iter, stimuli_iter, 0, :])+1)*params['epsilon_iter_step']*1000

		for assembly_iter in range(0, number_Russo_assemblies):
			plt.scatter(epsilon_x_axis, epsilon_results[dataset_iter, stimuli_iter, assembly_iter, :])
		
		ax.set_ylim(0, 1)
		ax.set_ylabel('Proportion of Assembly Activations')
		ax.set_xlabel('Epsilon (ms)')
		plt.savefig('./Processing_Data/Epsilon_curves_stim' + str(stimuli_iter+1) + '_layer' + str(params['network_layer']) +'.png')
		plt.show()
		
		#Output raw results as a CSV file
		# with open('./Processing_Data/epsilon_results.data', 'wb') as filehandle:
		# 	pickle.dump(epsilon_results, filehandle)


	#Run analyses that are performed to compare the activity of an assembly across datasets
	difference_in_assembly_counts = comparative_metrics(params, analysis_results, stimuli_iter, number_Russo_assemblies)
	
	if params['comparative_plotting_Boolean'] == 1:
		#Comparative plotting
		comparative_x_axis = np.arange(0, number_Russo_assemblies)

		plt.scatter(comparative_x_axis, analysis_results[0, stimuli_iter, :, 0], label='Stimulus 1 presentation', c='#e31a1c')
		plt.scatter(comparative_x_axis, analysis_results[1, stimuli_iter, :, 0], label='Stimulus 2 presentation', c='#2171b5')
		plt.scatter(comparative_x_axis, difference_in_assembly_counts, label='Difference in counts', c='#a922e7')

		plt.title('Number of assembly occurences')
		plt.legend()
		plt.show()


	#Information theory analysis, using the times at which the assemblies occur



	#Output information theory analysis

	

	return 0

#Check that an assembly isn't already for the alternative side of the stream, in which case discount it for the 'templates_alt_stream_bool' search
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
	if synchrony_bool == 1:
		#print(Russo_assembly_times[1:])
		Russo_assembly_times[1:] = np.zeros(len(Russo_assembly_times[1:]))
		#print(Russo_assembly_times[1:])

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
		(params['number_of_presentations']*params['duration_of_presentations']))

	#Average firing rate of maximally firing neuron in the assembly
	analysis_results[dataset_iter, stimuli_iter, assembly_iter, 8] = (analysis_results[dataset_iter, stimuli_iter, assembly_iter, 2] /
		(params['number_of_presentations']*params['duration_of_presentations']))

	#Average firing rate of minimally firing neuron in the assembly
	analysis_results[dataset_iter, stimuli_iter, assembly_iter, 9] = (analysis_results[dataset_iter, stimuli_iter, assembly_iter, 3] /
		(params['number_of_presentations']*params['duration_of_presentations']))


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

def information_theory_counting(params, activation_array):
	#Information can be either encoded in the discrete on-off of a particular PNG to a particular stimulus, or it be a discretized low/high 'assembly rate'
	#i.e. how many times over a given time interval is it active

	#I will begin by implementing the former (i.e a given assembly can be on or off)

	#For each stimulus, given that it has been presented, find the probability that a particular PNG is on at some point during the presentation period
	#This will be the number of presentation periods in which the stimulus is presented, divided by the number of presentations of that stimulus
	#Iterate through each time interval of presentation
	activation_counter = 0
	for ii in range(0, params['number_of_presentations']):
	#Extract from assembly_activations if the assembly had any activations in that interval, and if it did then record a 1, 0 otherwise
		activation_counter += np.any((activation_array[1, :] >= (ii*params['duration_of_presentations'])) & (activation_array[1, :] < ((ii+1)*params['duration_of_presentations'])))

	#Return a value containing the number of presentations when the assembly was active
	# *** In main() function, store this value in an array that has separate columns for each stimulus presented
	return activation_counter

def information_theory_calculation(params, information_theory_data):

	#Information_theory_data is indexed by [dataset_iter, assembly_iter]; thus the row indicates which stimulus was presented, and the 
	#column value indicates how many presentations were associated with at least one activation of that assembly


	#print(information_theory_data)

	no_math_error = 0.00000000000001

	#The probabilities of a particular assembly being active for each stimulus
	conditional_prob_array = information_theory_data/params['number_of_presentations']

	print('The maximum number of activations of a given assembly to stimulus 1 was ' + str(np.amax(information_theory_data[0,:])))
	print('The number of assemblies that had this many activations to stimulus 1 was ' + str(sum(information_theory_data[0,:] == np.amax(information_theory_data[0,:]))))

	#print(np.shape(conditional_prob_array))
	marginal_prob_array = np.sum(information_theory_data, axis=0)/(params['number_of_presentations']*params['number_stimuli'])
	#print(np.shape(marginal_prob_array))

	#Find the total probability that a given assembly is on during a presentation (i.e. irrespective of what stimulus is presented)
	#div_array = np.divide(conditional_prob_array[0, :], marginal_prob_array)
	#print(np.shape(div_array))

	#div_array = np.transpose(div_array[:,None])
	#print(np.shape(div_array))

	#print(np.shape(conditional_prob_array[0, :][:,None]))

	#print(conditional_prob_array)


	information1 = np.multiply(conditional_prob_array[0, :], np.log2(np.divide(conditional_prob_array[0, :], marginal_prob_array)+no_math_error))

	# print(np.shape(information1))
	# print(information1)

	# print(1-conditional_prob_array[0, :])
	# print(1-marginal_prob_array)
	# print(np.divide(1-conditional_prob_array[0, :], (1-marginal_prob_array)+no_math_error)+no_math_error)



	information2 = np.multiply(1-conditional_prob_array[0, :], np.log2(np.divide(1-conditional_prob_array[0, :], (1-marginal_prob_array+no_math_error))+no_math_error))

	# print(information2)

	# print("Total information is:")

	information_theory_results = information1+information2


	return information_theory_results


main(params)

#Output all the assemblies that were able to be separated in the training data-set by a factor of e.g. 80%

#Check performance of these assemblies with the learned parameters on a hold-out dataset, before making comments on information content (otherwise can argue that have just over-fit the data to have high information)

