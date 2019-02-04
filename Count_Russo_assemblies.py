#!/usr/bin/env python3

import numpy as np
from scipy.spatial.distance import pdist, squareform
import warnings

#This code can be used after applying the Russo algorithm to extract polychronous assemblies from parallel spike-train data
#Using the extracted assemblies, the following algorithm will count the number of instances of each assembly, relative to a particular stimulus



#Definitions of parameters:
#epsilon : cut-off margin, in seconds

params = {'epsilon' : 0.003}
Russo_assembly_times = np.array([0.003, 0.012, 0.018, 0.027, 0.042])
Russo_assembly_ids = np.array([1, 4, 5, 8, 10]) - 1 #Note the neuron ids are indexed in the data from 1, so the minus 1 corrects for indexing into Python arrays


def main(params, Russo_assembly_times, Russo_assembly_ids):

	spike_data = create_example_dataset(Russo_assembly_times, Russo_assembly_ids)
	#print(spike_data)

	#Iterate through each assembly


	#Iterate through candidate activations of an assembly; any time the first neuron of the prescribed assembly spikes, it is considered a candidate activation
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		activation_array = iterate_through_candidates(params, Russo_assembly_times, Russo_assembly_ids, spike_data)

	print(activation_array)

	return 0

#Load data set for analysis
def load_data():

	return 0

#Create simple dataset for integration testing
def create_example_dataset(Russo_assembly_times, Russo_assembly_ids):

	spike_data = np.empty((10, 5))
	spike_data[:] = np.NaN

	#Create an array of 'ground truth' candidate activations, which indexed from 1-4 are increasingly poor examples of a true activation
	#Also assigns a prescribed spike time for the first neuron in the assembly, to test whether this can be successfully recovered
	candidate_activations = np.empty((4, 5))

	for ii in range(0,4):
		candidate_activations[ii, :] = Russo_assembly_times + ii*20 #Off-set the beginning of the assembly activation by a certain amount of time
		candidate_activations[ii, 1:] = candidate_activations[ii, 1:] + ii*2*np.power(-1, ii) #Off-set the spike times of all assembly neurons other than the first by a certain ammount (essentially non-random 'noise'); offset alternates between being negative and positive
		#print(candidate_activations[ii])
		spike_data[Russo_assembly_ids, ii] = candidate_activations[ii, :]

	#For neurons that did not spike, assign them a value (here a synchronous spike is used)
	non_spiked_ids = np.array([2, 3, 6, 7, 9]) - 1
	spike_data[non_spiked_ids, 0] = 5

	return spike_data

#Iterate through candidate assembly activations; defined as a separate function to assist appropriate warning handling
def iterate_through_candidates(params, Russo_assembly_times, Russo_assembly_ids, spike_data):

	warnings.warn("NaN_Comparison", RuntimeWarning) #Prevents the NaN comparison below from generating a run-time error
	number_candidate_assemblies = np.sum(spike_data[0, :] > 0)

	# *** NB THE FIRST VERSION OF THIS CODE HAS ONLY BEEN WRITTEN TO HANDLE THE 1st NEURON IN THE INDEX/i.e. a single artificial assembly

	activation_array = np.empty([2, number_candidate_assemblies])

	for jj in range(0, number_candidate_assemblies):
		first_neuron_spike_time = spike_data[Russo_assembly_ids[0], jj]
		(upper_bound, lower_bound) = create_boundaries(params, Russo_assembly_times, first_neuron_spike_time)
		activation_array[0, jj] = evaluate_assembly_activation(Russo_assembly_times, Russo_assembly_ids, spike_data, upper_bound, lower_bound)
		activation_array[1, jj] = first_neuron_spike_time

	return activation_array #Note the returned assembly activation time will only take a non-zero value if the assembly was actually activated

#Create the upper and lower limits of the spike times
def create_boundaries(params, Russo_assembly_times, first_neuron_spike_time):
	#NB that the first neuron will always evaluate to 1, as it is by definition within the given range

	upper_bound = Russo_assembly_times + first_neuron_spike_time + params['epsilon']
	lower_bound = Russo_assembly_times + first_neuron_spike_time - params['epsilon']

	return (upper_bound, lower_bound)

	# test_warning_array = np.array([1, 3, 5, np.NaN]) #Test if warnings can still be thrown in nested functions (out of curiosity)
	# test_warning_bool = (test_array > 0)

	# print(upper_bound)
	# print(lower_bound)

def evaluate_assembly_activation(Russo_assembly_times, Russo_assembly_ids, spike_data, upper_bound, lower_bound):

	#Non-vectorized implementation; bool_iter will return a 1 if all spike times fall in the prescribed range, 0 otherwise
	bool_iter = 1
	for kk in range(0, len(Russo_assembly_ids)):
		bool_iter = bool_iter * np.any((spike_data[Russo_assembly_ids[kk],:] <= upper_bound[kk]) & (spike_data[Russo_assembly_ids[kk],:] >= lower_bound[kk]))
	
	return bool_iter	

	#boolean_array = (np.all(spike_data[Russo_assembly_ids,:] <= upper_bound) and np.all(spike_data[Russo_assembly_ids,:] >= lower_bound))
	# print(boolean_array)

	#Return the time at which the assembly was activated


main(params, Russo_assembly_times, Russo_assembly_ids)



# #Take the absolute difference between the candidate activation and the Russo-prescribed spike times
# diff = np.absolute(Russo_assembly_times - example_terrible_activation)
# print(diff)

# #Perform a vectorized comparison to epsilon
# boolean_array = (diff <= epsilon)
# print(boolean_array)

# #Return the truth value of the entire array (mathematically equivalent to taking the product of all the array units together)
# activated = np.all(boolean_array)
# print(activated)

#Print the number of activations associated with a particular assembly


# *** Test the implementation on known ground-truth data (integration testing)






# *** the below was just some very quick code to see if I could perform the Gaussian kernel - I was not yet convinced that it was working, although changing the units
# of time to seconds appeared to improve the separation following the kernel ***

#Calculate the Gaussian kernel between the Russo_assembly_times and the example activation
#pairwise_dists = squareform(pdist())


# sigma = 1

# kernel_vector = (1 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-np.power((Russo_assembly_times - example_perfect_activation), 2) / (2 * np.power(sigma, 2)))
# kernel_product = np.prod(kernel_vector)

# #print(kernel_vector)
# print(kernel_product)

# kernel_vector = (1 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-np.power((Russo_assembly_times - example_good_activation), 2) / (2 * np.power(sigma, 2)))
# kernel_product = np.prod(kernel_vector)

# #print(kernel_vector)
# print(kernel_product)

# kernel_vector = (1 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-np.power((Russo_assembly_times - example_bad_activation), 2) / (2 * np.power(sigma, 2)))
# kernel_product = np.prod(kernel_vector)
# print(kernel_product)


# kernel_vector = (1 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-np.power((Russo_assembly_times - example_terrible_activation), 2) / (2 * np.power(sigma, 2)))
# kernel_product = np.prod(kernel_vector)
# print(kernel_product)

# *** ? Variance needs to be set dynamically





#Output all the assemblies that were able to be separated in the training data-set by a factor of e.g. 80%



#Check performance of these assemblies with the learned parameters on a hold-out dataset, before making comments on information content (otherwise can argue that have just over-fit the data to have high information)


