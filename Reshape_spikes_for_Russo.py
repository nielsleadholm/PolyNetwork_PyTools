#!/usr/bin/env python3

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import warnings

### The following tool sorts the output spikes from the Spike simulator (1D arrays) into 2D .csv files suitable for analysis by the Russo algorithm ###

#For details on the Russo algorithm, please see the publication "Cell assemblies at multiple time scales with arbitrary lag constellations", Russo and Durstewitz, 2017
#Spike is a simulator written in C++/CUDA for spiking neural networks, more information on which can be found here: https://sites.google.com/view/spike-simulator/home
#This code assumes the user is analyzing neural networks wherein each layer is a 2D lattice, and multiple layers are present (see for example the network used in the paper 
#"The Emergence of Polychronization and Feature Binding in a Spiking Neural Network Model of the Primate Ventral Visual System", Eguchi et al, 2018)

#The Russo algorithm requires a 2D array where rows represent neuron IDs, and each neurons nth spike is indicated by the column
#Each array entry specifies the time (in seconds) at which the spike occurred; empty entries are required to be filled with a NaN identifier
#As the Russo algorithm does not accept neurons that never spike (i.e. empty rows), the following code adds a single spike at a random time-point for such neurons;
#This latter functionality can be removed with the add_Russo_random_spike parameter below

#The user must provide (below) parameters that were used in generating the Spike neural network simulation
#Specify the number of neurons in each excitatory layer and each inhibitory layer, the number of layers, layer of interest, and the number of stimuli
#If using the 'binary network' architecture, _dim should specify the total size of each layer, and not the individual streams
#max_plot_time determines how many ms of data should be plotted here
#random_test_to_screen prints to screen additional tests that require visual inspection by the user
#shuffle_Boolean randomly shuffles the neuron ids locations, so that their firing rates remain the same, but any fixed temporal relationships are broken

params = {'extracted_layer' : 1,
	'max_plot_time' : 0.4,
	'excit_dim' : 5*5*2,
	'number_of_presentations' : 50,
	'duration_of_presentations' : 0.2,
	'inhib_dim' : None,
	'num_stimuli' : 2,
	'add_Russo_random_spike' : True,
	'manual_test_to_screen' : False,
	'plot_Boolean' : True,
	'save_output_Boolean' : True,
	'shuffle_Boolean' : True}


#Loop through each stimulus
def main(params):

	test_suite(params)

	for jj in range(params['num_stimuli']):
		(spike_ids, spike_times) = load_spikes(jj) #NB that neurons IDs begin at 1

		if params['shuffle_Boolean'] == True:
			spike_ids = shuffle_spikes(params, spike_ids, spike_times)

		extracted_mask = extract_mask(params, spike_ids)

		(extracted_ids, extracted_times) = extract_spikes(spike_ids, spike_times, extracted_mask)

		max_spikes = np.max(np.bincount(extracted_ids)) #Find the number of spikes assoc. with the max-spiking neuron

		Russo_array = initialize_Russo(params, max_spikes)

		Russo_array = populate_Russo(params, extracted_ids, extracted_times, Russo_array)

		if params['plot_Boolean'] == True:
			#The plot function will generate a RunTime warning due to a NaN comparison; however, the expected output of this code is correct, so the warning is not displayed
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				plot_Russo(params, Russo_array, jj)

		if params['save_output_Boolean'] == True:
			if params['shuffle_Boolean'] == True:
				np.savetxt("./Processing_Data/shuffled_posttraining_stim" + str(jj+1) + "_layer" + str(params['extracted_layer']) + "_Russo.csv", Russo_array, delimiter=',')
			else:
				np.savetxt("./Processing_Data/posttraining_stim" + str(jj+1) + "_layer" + str(params['extracted_layer']) + "_Russo.csv", Russo_array, delimiter=',')

	return None


#Load spike ids and times
def load_spikes(stimuli_iter):
	spike_ids = np.genfromtxt('./Processing_Data/output_spikes_posttraining_stim' + str(stimuli_iter+1) +'SpikeIDs.txt')
	spike_times = np.genfromtxt('./Processing_Data/output_spikes_posttraining_stim' + str(stimuli_iter+1) +'SpikeTimes.txt')
	return (spike_ids, spike_times)


#Within each stimulus presentation, shuffle the association between spike IDs and their spike times, so that spikes still occur at different times, but with different neurons
#This maintains each neuron's firing rate, but breaks any temporal associations between a particular neuron and a particular spike time
def shuffle_spikes(params, spike_ids, spike_times):
	shuffled_ids = []

	#Iterate through all time-windows; note the addition of 1, as the last presentation is still associated with a window
	for ii in range(params['number_of_presentations']+1):

		#Find the indices of spikes in the given window of interest, and use these indices to extract the neuron IDs that spiked in the window
		temp_IDs = spike_ids[np.where((ii*params['duration_of_presentations'] < spike_times) & (spike_times <= (ii+1)*params['duration_of_presentations']))]

		#Shuffle those IDs and append them
		np.random.shuffle(temp_IDs)
		shuffled_ids.extend(temp_IDs)

	assert len(shuffled_ids) == len(spike_ids), "Size of spike ID array not preserved after shuffling."
	return np.asarray(shuffled_ids)


#Return extraction mask defining the spikes of interest
def extract_mask(params, spike_ids):
	#Return an array of indeces for the neurons in the layer of interest
	extracted_mask = np.where((params['excit_dim']*(params['extracted_layer']-1) < spike_ids) & (spike_ids <= params['extracted_layer']*params['excit_dim'])) 
	return extracted_mask


#Extract and format spike ids and times of interest
def extract_spikes(spike_ids, spike_times, extracted_mask):
	extracted_ids = np.take(spike_ids, extracted_mask) #An array of spike IDs, restricted to the layer of interest
	extracted_times = np.take(spike_times, extracted_mask) #An array of spike times, restricted to the layer of interest

	extracted_ids = np.reshape(extracted_ids, len(extracted_ids[0])) #Re-shape the array into a column array
	extracted_ids = extracted_ids.astype(int) #Convert from Spike simulator's output (float)
	
	return (extracted_ids, extracted_times)


#Initialize a NaN array with rows = number of unique neurons in the layer, and columns = number of spikes of the maximally active neuron
def initialize_Russo(params, max_spikes):
	Russo_array = np.zeros([params['excit_dim'], max_spikes])
	Russo_array[:, :] = np.nan
	return Russo_array


#Iterate through each neuron of interest, inserting its spikes into the Russo-suitable array; if a neuron never spikes, insert a single random spike
def populate_Russo(params, extracted_ids, extracted_times, Russo_array):
	for ii in range(0, params['excit_dim']):
	  #Extract a binary mask containing the indeces of when the neuron of interest has fired
	  temp_mask = np.where(extracted_ids == (params['excit_dim']*(params['extracted_layer'] - 1) + ii + 1))

	  #Use the mask to identify all the spike times associated with that neuron, and assign it to Russo_array
	  Russo_array[ii, 0:(np.size(np.take(extracted_times, temp_mask)))] = np.take(extracted_times, temp_mask)

	  if ((math.isnan(Russo_array[ii, 0]) == 1) and (params['add_Russo_random_spike'] == 1)): #If the first element is NaN, the entire row is (i.e. the neuron never spiked)
	    Russo_array[ii, 0] = np.random.random()*np.max(extracted_times) #Assigns the neuron a single spike, the time of which is sampled from a continuous uniform distribution

	return(Russo_array)


def plot_Russo(params, Russo_array, stimuli_iter):
	plt.figure(stimuli_iter)

	warnings.warn("NaN_Comparison", RuntimeWarning)

	for ii in range(0, params['excit_dim']):
		#Plot each neuron's spikes in turn; note the y-valuess are multiplied by (ii+1), so that the y-axis labels correspond to the neuron index (original simulation, beginning at 1), and not the Russo_array index (which begins at 0)
		plt.scatter(Russo_array[ii,(Russo_array[ii, :]<params['max_plot_time'])], np.ones(len(Russo_array[ii,(Russo_array[ii, :]<params['max_plot_time'])]))*(ii+1), c='k', marker='.')
		
	plt.show()




### Testing Functions ###

#Unit test for extract_mask function (extracting first layer)
def test_extract_mask_FirstLayer():
	test_params = {'extracted_layer' : 1, 'excit_dim' : 5}
	test_spike_ids = np.array([4, 9, 1, 3, 9, 2, 10, 7, 5, 6, 2, 8, 1, 1])
	exp = np.array([0, 2, 3, 5, 8, 10, 12, 13])
	exp = np.reshape(exp, [1, len(exp)])

	obs = extract_mask(test_params, test_spike_ids)

	assert np.all(obs == exp)

#Unit test for extract_mask function (extracting higher layers)
def test_extract_mask_HigherLayer():

	test_params = {'extracted_layer' : 2, 'excit_dim' : 5}
	test_spike_ids = np.array([4, 9, 1, 3, 9, 2, 10, 7, 5, 6, 2, 8, 1, 1])
	exp = np.array([1, 4, 6, 7, 9, 11])
	exp = np.reshape(exp, [1, len(exp)])

	obs = extract_mask(test_params, test_spike_ids)

	assert np.all(obs == exp)

#Unit test for extract_spikes function
def test_extract_spikes():
	test_spike_ids = np.array([4, 9, 1, 3, 9, 2, 10, 7])
	test_spike_times = np.array([0.04, 0.06, 0.9, 1.2, 1.8, 4.0, 5.9, 10.2])
	test_extracted_mask = np.array([0, 2, 3, 5])
	test_extracted_mask = np.reshape(test_extracted_mask, [1, len(test_extracted_mask)])

	exp_ids = ([4, 1, 3, 2])
	exp_times = ([0.04, 0.9, 1.2, 4.0])

	(obs_ids, obs_times) = extract_spikes(test_spike_ids, test_spike_times, test_extracted_mask)

	assert (np.all(exp_ids == obs_ids) and np.all(exp_times == obs_times))

#Unit test for populate Russo function when first layer is of interest
def test_populate_Russo_FirstLayer(params):
	test_params = {'extracted_layer' : 1, 'excit_dim' : 5, 'add_Russo_random_spike' : params['add_Russo_random_spike']}
	test_extracted_ids = np.array([5, 1, 1, 3, 1, 2])
	test_max_spikes = 3
	test_extracted_times = np.array([0.04, 0.08, 0.45, 0.9, 1.2, 4.0])
	
	exp = np.array([[0.08, 0.45, 1.2], [4.0, math.nan, math.nan], [0.9, math.nan, math.nan], [np.float, math.nan, math.nan], [0.04, math.nan, math.nan]])

	test_Russo_array = initialize_Russo(test_params, test_max_spikes)

	obs = populate_Russo(test_params, test_extracted_ids, test_extracted_times, test_Russo_array)

	assert np.all(obs[0, 0:3] == exp[0, 0:3])

	#NB: only the first line is tested automatically
	if params['manual_test_to_screen'] == 1:	
		print("The expected Russo array (first layer extraction) is below; if add_Russo_random_spike==0, <float> will be NaN in observed. \n")
		print(exp)
		print("The expected Russo array shape (first layer extraction) is \n")
		print(np.shape(exp))
		print("The observed Russo array (first layer extraction) is \n")
		print(obs)
		print("The observed Russo array shape (first layer extraction) is \n")
		print(np.shape(obs))
		print("\n")

#Unit test for populate Russo function when higher layers are of interest
def test_populate_Russo_HigherLayer(params):
	test_params = {'extracted_layer' : 2, 'excit_dim' : 5, 'add_Russo_random_spike' : params['add_Russo_random_spike']}
	test_extracted_ids = np.array([10, 6, 6, 8, 6, 7])
	test_max_spikes = 3
	test_extracted_times = np.array([0.04, 0.08, 0.45, 0.9, 1.2, 4.0])
	
	exp = np.array([[0.08, 0.45, 1.2], [4.0, math.nan, math.nan], [0.9, math.nan, math.nan], [np.float, math.nan, math.nan], [0.04, math.nan, math.nan]])

	test_Russo_array = initialize_Russo(test_params, test_max_spikes)

	obs = populate_Russo(test_params, test_extracted_ids, test_extracted_times, test_Russo_array)

	assert np.all(obs[0, 0:3] == exp[0, 0:3])

	#NB: only the first row is tested automatically; below offers a manual printout
	if params['manual_test_to_screen'] == 1:	
		print("The expected Russo array (higher layer extraction) is below; if add_Russo_random_spike==0, <float> will be NaN in observed.\n")
		print(exp)
		print("The expected Russo array shape (higher layer extraction) is \n")
		print(np.shape(exp))
		print("The observed Russo array (higher layer extraction) is \n")
		print(obs)
		print("The observed Russo array shape (higher layer extraction) is \n")
		print(np.shape(obs))
		print("\n")

def test_suite(params):
	test_extract_mask_FirstLayer()
	test_extract_mask_HigherLayer()
	test_extract_spikes()
	test_populate_Russo_FirstLayer(params)
	test_populate_Russo_HigherLayer(params)
	return

main(params)

