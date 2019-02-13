#!/usr/bin/env python3

import numpy as np
from scipy.spatial.distance import pdist, squareform
import warnings
import matplotlib.pyplot as plt
import matplotlib

#This code can be used after applying the Russo algorithm to extract polychronous assemblies from parallel spike-train data
#Using the extracted assemblies, the following algorithm will count the number of instances of each assembly, relative to a particular stimulus



#Definitions of parameters:
#epsilon0 : primary cut-off margin for spike times in assembly search, given in seconds; note it falls on either side of the prescribed spike times, so an epsilon of 1.5ms corresponds to a window of 3ms
#Russo_bin_size
#number_stimuli
#epsilon1 : broad cut-off margin for spike times in assembly search, given in seconds; used to evaluate how much increasing epsilon changes the number of assemblies that are identified
#dataset_duration : length of each data-set, in seconds

params = {'epsilon0' : 0.0005,
	'Russo_bin_size' : 0.003,
	'number_stimuli' : 2,
	'epsilon1' : 0.015,
	'dataset_duration' : 400,
	'epsilon_iter_bool' : 1,
	'epsilon_max' : 0.05}

#Assemblies extracted from stimulus 2 dataset with bin width 3ms and pruning

#Assembly 28
# Russo_assembly_times = np.array([0, 0, 4, 6, 9, 14])*params['Russo_bin_size']/1000
# Russo_assembly_ids = np.array([189, 177, 153, 201, 165, 213]) - 1 #Note the neuron ids are indexed in the data from 1, so the minus 1 corrects for indexing into Python arrays

#Assembly 30
# Russo_assembly_times = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])*params['Russo_bin_size']/1000
# Russo_assembly_ids = np.array([180, 144, 192, 424, 392, 328, 156, 488, 456]) - 1 #Note the neuron ids are indexed in the data from 1, so the minus 1 corrects for indexing into Python arrays

#Assemblies extracted from stimulus 1 dataset with bin width 3ms and pruning

#Assembly 27
Russo_assembly_times = np.array([0.000, 0.000, 0.000, 0.000, 0.000, 0.003, 0.003, 0.003, 0.006, 0.006, 0.006])
Russo_assembly_ids = np.array([424, 328, 392, 360, 296, 192, 144, 180, 488, 552, 456]) - 1 #Note the neuron ids are indexed in the data from 1, so the minus 1 corrects for indexing into Python arrays


def main(params, Russo_assembly_times, Russo_assembly_ids):

	number_Russo_assemblies = 1
	#Initialize array to hold main analysis results; final value relates to the number of analysis metrics that are used
	analysis_results = np.empty([params['number_stimuli'], params['number_stimuli'], number_Russo_assemblies, 12])
	#The first dimension of analysis_results is the stimuli associated with a particular dataset, while the second is the stimuli associated with the Russo-extracted assemblies



	#Initialize array to hold results from iterating the epsilon value; the final value corresponds to the total number of steps that are taken, as an integer value
	epsilon_results = np.empty([params['number_stimuli'], params['number_stimuli'], number_Russo_assemblies, int((params['epsilon_max']-params['epsilon0'])*1000)]) 

	#Iterate through each data set; note the stimuli file names are indexed from 1
	for dataset_iter in range (0, params['number_stimuli']):
		spike_data = np.genfromtxt('posttraining_stim' + str(dataset_iter+1) + '_Russo.csv', delimiter=',') 

		#Iterate through each stimulus
		stimuli_iter = 0

		#Load data on the structure and spike times of Russo-identified assemblies; convert the spike times into seconds from number of lags in ms; change index from starting at 1 to starting at 0

		#Iterate through each assembly
		assembly_iter=0

		#Search for activations of an assembly (with both the primary aand broad epsilon)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			activation_array = iterate_through_candidates(params, Russo_assembly_times, Russo_assembly_ids, spike_data)

		#Run analyses that are performed for an assembly in a single dataset
		analysis_results = analysis_metrics(params, Russo_assembly_times, Russo_assembly_ids, spike_data, activation_array, 
			analysis_results, dataset_iter, stimuli_iter, assembly_iter)

		#print(analysis_results[dataset_iter, stimuli_iter, assembly_iter, :])

		#Run the specific 'epsilon analysis', where increasing values of epsilon are used for eventual plotting
		epsilon_results = analysis_epsilon(params, Russo_assembly_times, Russo_assembly_ids, spike_data, activation_array, analysis_results,
			epsilon_results, dataset_iter, stimuli_iter, assembly_iter)

		print(epsilon_results[dataset_iter, stimuli_iter, assembly_iter, :])

		x_axis = np.arange(1, len(epsilon_results[dataset_iter, stimuli_iter, assembly_iter, :])+1)
		
		plt.scatter(x_axis, epsilon_results[dataset_iter, stimuli_iter, assembly_iter, :])


	#Run analyses that are perfomred to compare the activity of an assembly across datasets
	comparative_metrics(params)

	plt.show()
	#Output raw results as a CSV file

	#Information theory analysis, using the times at which the assemblies occur

	#Output information theory analysis

	return 0

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

	return activation_array

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

def comparative_metrics(params):

	#Ratio of assembly activation count for the stimulus it ostensbily encodes, vs other stimuli 

	# ***need to consider accounting for >2 stimuli

	#Ratio of assembly activation ratio (i.e. how many times the assembly activated when the first neuron in the assembly activated), for the stimulus it ostensibly encodes, vs other stimuli

	return 0

def analysis_epsilon(params, Russo_assembly_times, Russo_assembly_ids, spike_data, activation_array, analysis_results,
			epsilon_results, dataset_iter, stimuli_iter, assembly_iter):

	number_candidate_assemblies = int(analysis_results[dataset_iter, stimuli_iter, assembly_iter, 0])
	epsilon = params['epsilon0']

	#Iterate through each value of epsilon
	for ii in range(0, len(epsilon_results[dataset_iter, stimuli_iter, assembly_iter, :])):

		activations_count = 0
		epsilon = epsilon + 0.0005 #How many ms epsilon is iterated by 

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

main(params, Russo_assembly_times, Russo_assembly_ids)




#Output all the assemblies that were able to be separated in the training data-set by a factor of e.g. 80%



#Check performance of these assemblies with the learned parameters on a hold-out dataset, before making comments on information content (otherwise can argue that have just over-fit the data to have high information)



#Manually written out assemblies etc. that may be useful for testing; note that Russo assembly lags need to be multiplied by the Russo-bin width

	# test_array2 = np.array([[0.,    0.02,  0.04,  0.06,    np.NaN]
	# [5.,      np.NaN,   np.NaN,   np.NaN,   np.NaN]
	# [5.,      np.NaN,   np.NaN,   np.NaN,   np.NaN]
	# [0.012, 0.03,  0.056, 0.066,   np.NaN]
	# [0.018, 0.036, 0.062, 0.072,   np.NaN]
	# [5.,      np.NaN,   np.NaN,   np.NaN,   np.NaN]
	# [5.,      np.NaN,   np.NaN,   np.NaN,   np.NaN]
	# [0.027, 0.045, 0.071, 0.081,   np.NaN]
	# [5.,      nan.NaN,   np.NaN,   nan.NaN,   np.NaN]
	# [0.042, 0.06,  0.086, 0.096,   np.NaN]])

#Assemblies extracted from stimulus 1 dataset with bin width 3ms and pruning

# #Assembly 1
# Russo_assembly_times = np.array([0, 5])/1000
# Russo_assembly_ids = np.array([21, 85]) - 1 #Note the neuron ids are indexed in the data from 1, so the minus 1 corrects for indexing into Python arrays

#Assembly 2
# Russo_assembly_times = np.array([0, 3])/1000
# Russo_assembly_ids = np.array([29, 93]) - 1 #Note the neuron ids are indexed in the data from 1, so the minus 1 corrects for indexing into Python arrays

# #Assembly 3
# Russo_assembly_times = np.array([0, 4])/1000
# Russo_assembly_ids = np.array([60, 124]) - 1 #Note the neuron ids are indexed in the data from 1, so the minus 1 corrects for indexing into Python arrays

# #Assembly 4
# Russo_assembly_times = np.array([0, 4])/1000
# Russo_assembly_ids = np.array([149, 161]) - 1 #Note the neuron ids are indexed in the data from 1, so the minus 1 corrects for indexing into Python arrays

#Assembly 5
# Russo_assembly_times = np.array([0, 2])/1000
# Russo_assembly_ids = np.array([149, 197]) - 1 #Note the neuron ids are indexed in the data from 1, so the minus 1 corrects for indexing into Python arrays

#Assembly 6
# Russo_assembly_times = np.array([0, 3])/1000
# Russo_assembly_ids = np.array([185, 161]) - 1 #Note the neuron ids are indexed in the data from 1, so the minus 1 corrects for indexing into Python arrays

#Assembly 7
# Russo_assembly_times = np.array([0, 2])/1000
# Russo_assembly_ids = np.array([208, 220]) - 1 #Note the neuron ids are indexed in the data from 1, so the minus 1 corrects for indexing into Python arrays

#Assembly 8
# Russo_assembly_times = np.array([0, 1])/1000
# Russo_assembly_ids = np.array([212, 224]) - 1 #Note the neuron ids are indexed in the data from 1, so the minus 1 corrects for indexing into Python arrays


# #Assembly 22
# Russo_assembly_times = np.array([0, 0, 5, 6])/1000
# Russo_assembly_ids = np.array([178, 190, 202, 154]) - 1 #Note the neuron ids are indexed in the data from 1, so the minus 1 corrects for indexing into Python arrays

#Assembly 23
# Russo_assembly_times = np.array([0, 0, 3, 7])/1000
# Russo_assembly_ids = np.array([191, 179, 155, 167]) - 1 #Note the neuron ids are indexed in the data from 1, so the minus 1 corrects for indexing into Python arrays

# #Assembly 24
# Russo_assembly_times = np.array([0, 0, 0, 5, 5])/1000
# Russo_assembly_ids = np.array([188, 152, 176, 200, 372]) - 1 #Note the neuron ids are indexed in the data from 1, so the minus 1 corrects for indexing into Python arrays

#Assembly 25
# Russo_assembly_times = np.array([0.000, 0.003, 0.015, 0.021, 0.027, 0.027])
# Russo_assembly_ids = np.array([177, 189, 153, 201, 165, 213]) - 1 #Note the neuron ids are indexed in the data from 1, so the minus 1 corrects for indexing into Python arrays

#Assembly 26
# Russo_assembly_times = np.array([0, 0, 0, 0, 2, 2])/1000
# Russo_assembly_ids = np.array([168, 392, 424, 328, 552, 456]) - 1 #Note the neuron ids are indexed in the data from 1, so the minus 1 corrects for indexing into Python arrays

#Assembly 27
# Russo_assembly_times = np.array([0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.001, 0.001, 0.002, 0.002, 0.002,])
# Russo_assembly_ids = np.array([424, 328, 392, 360, 296, 192, 144, 180, 488, 552, 456]) - 1 #Note the neuron ids are indexed in the data from 1, so the minus 1 corrects for indexing into Python arrays

#Assembly for testing
# Russo_assembly_times = np.array([0.000, 0.012, 0.018, 0.027, 0.042])
# Russo_assembly_ids = np.array([1, 4, 5, 8, 10]) - 1 #Note the neuron ids are indexed in the data from 1, so the minus 1 corrects for indexing into Python arrays


