#!/usr/bin/env python3

import numpy as np
from scipy.spatial.distance import pdist, squareform
import warnings

#This code can be used after applying the Russo algorithm to extract polychronous assemblies from parallel spike-train data
#Using the extracted assemblies, the following algorithm will count the number of instances of each assembly, relative to a particular stimulus



#Definitions of parameters:
#epsilon : cut-off margin, in seconds

params = {'epsilon' : 0.003}
Russo_assembly_times = np.array([0.000, 0.012, 0.018, 0.027, 0.042])
Russo_assembly_ids = np.array([1, 4, 5, 8, 10]) - 1 #Note the neuron ids are indexed in the data from 1, so the minus 1 corrects for indexing into Python arrays


def main(params, Russo_assembly_times, Russo_assembly_ids):

	spike_data = create_example_dataset(Russo_assembly_times, Russo_assembly_ids)

	print(spike_data)

	#Iterate through each assembly



	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		activation_array = iterate_through_candidates(params, Russo_assembly_times, Russo_assembly_ids, spike_data)

	total_activations = np.sum(activation_array[0, :])
	
	print(activation_array)
	print(total_activations)

	return 0

#Load data set for analysis
def load_data():

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


	return spike_data

#Iterate through candidate activations of an assembly; any time the first neuron of the prescribed assembly spikes, it is considered a candidate activation; defined as a separate function to assist appropriate warning handling
def iterate_through_candidates(params, Russo_assembly_times, Russo_assembly_ids, spike_data):

	warnings.warn("NaN_Comparison", RuntimeWarning) #Prevents the NaN comparison below from generating a run-time error
	number_candidate_assemblies = np.sum(spike_data[0, :] >= 0)

	# *** NB THE FIRST VERSION OF THIS CODE HAS ONLY BEEN WRITTEN TO HANDLE THE 1st NEURON IN THE INDEX/i.e. a single artificial assembly

	activation_array = np.empty([2, number_candidate_assemblies]) #Tracks whethr an assembly has activated, and at what time

	for jj in range(0, number_candidate_assemblies):
		first_neuron_spike_time = spike_data[Russo_assembly_ids[0], jj]
		(upper_bound, lower_bound) = create_boundaries(params, Russo_assembly_times, first_neuron_spike_time)
		activation_array[0, jj] = evaluate_assembly_activation(Russo_assembly_times, Russo_assembly_ids, spike_data, upper_bound, lower_bound)
		activation_array[1, jj] = first_neuron_spike_time*activation_array[0, jj] #A non-zero value for the time of assembly activation is only assigned if the assembly activated

	return activation_array

#Create the upper and lower limits of the spike times
def create_boundaries(params, Russo_assembly_times, first_neuron_spike_time):
	#NB that the first neuron will always evaluate to 1, as it is by definition within the given range

	#Note the first assembly neuron is not included in the boundary arrays, as it is by definition within the boundary
	upper_bound = Russo_assembly_times[1:] + first_neuron_spike_time + params['epsilon']
	lower_bound = Russo_assembly_times[1:] + first_neuron_spike_time - params['epsilon']

	# print(upper_bound)
	# print(lower_bound)

	return (upper_bound, lower_bound)

#Non-vectorized implementation; the function will return a 1 if all spike times fall in the prescribed range, 0 otherwise
def evaluate_assembly_activation(Russo_assembly_times, Russo_assembly_ids, spike_data, upper_bound, lower_bound):

	bool_iter = 1
	#Note in the iteration, the first neuron is skipped, as by definition this is active at the necessary time
	for kk in range(1, len(Russo_assembly_ids)):
		bool_iter = bool_iter * np.any((spike_data[Russo_assembly_ids[kk],:] <= upper_bound[kk-1]) & (spike_data[Russo_assembly_ids[kk],:] >= lower_bound[kk-1]))
	
	return bool_iter	



main(params, Russo_assembly_times, Russo_assembly_ids)






#Output all the assemblies that were able to be separated in the training data-set by a factor of e.g. 80%



#Check performance of these assemblies with the learned parameters on a hold-out dataset, before making comments on information content (otherwise can argue that have just over-fit the data to have high information)


