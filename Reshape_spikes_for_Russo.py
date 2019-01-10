#!/usr/bin/env python3

import numpy as np
import math

### The following tool sorts the output spikes from the Spike simulator (1D arrays) into 2D .csv files suitable for analysis by the Russo algorithm ###

#For details on the Russo algorithm, please see the publication "Cell assemblies at multiple time scales with arbitrary lag constellations", Russo and Durstewitz, 2017
#Spike is a simulator written in C++/CUDA for spiking neural networks, more information on which can be found here: https://sites.google.com/view/spike-simulator/home
#This code assumes the user is analyzing neural networks wherein each layer is a 2D lattice, and multiple layers are present (see for example the network used in the paper 
#"The Emergence of Polychronization and Feature Binding in a Spiking Neural Network Model of the Primate Ventral Visual System", Eguchi et al, 2018)

#The Russo algorithm requires a 2D array where rows represent neuron IDs, and each neurons nth spike is indicated by the column
#Each array entry specifies the time (in seconds) at which the spike occurred; empty entries are required to be filled with a NaN identifier
#As the Russo algorithm does not accept neurons that never spike (i.e. empty rows), the following code adds a single spike at a random time-point for such neurons

#The user must provide (below) parameters that were used in generating the Spike neural network simulation
#Specify the number of neurons in each excitatory layer and each inhibitory layer, the number of layers, layer of interest, and the number of stimuli
params = {'excit_dim' : 32*32,
'inhib_dim' : 12*12,
'num_layers' : 3,
'extracted_layer' : 3,
'num_stimuli' : 2,
'run_test_suite' : 1,
'manual_test_to_screen' : 0}

#Loop through each stimulus
def main(params):

	if params['run_test_suite'] == 1:
		test_suite(params)
		print("All unit tests have successfully passed.")

	for jj in range(0, params['num_stimuli']):
		(spike_ids, spike_times) = load_spikes(jj) #NB that neurons IDs begin at 1

		# print(np.shape(spike_ids))
		# print(spike_ids[0:20])
		# print(np.max(spike_ids))
		# print(np.min(spike_ids))
		# print(np.shape(spike_times))
		extracted_mask = extract_mask(params, spike_ids)

		#print(extracted_mask)
		# print(np.shape(extracted_mask))

		(extracted_ids, extracted_times) = extract_spikes(spike_ids, spike_times, extracted_mask)

		# print(np.shape(extracted_ids))
		# print(np.shape(extracted_times))
		# print(extracted_ids[0:5])
		# print(extracted_times[0:5])

		max_spikes = np.max(np.bincount(extracted_ids)) #Find the number of spikes assoc. with the max-spiking neuron
		# print(max_spikes)

		Russo_array = initialize_Russo(params, max_spikes)

		# print(np.shape(Russo_array))
		# print(Russo_array[0:5, 0:5])

		Russo_array = populate_Russo(params, extracted_ids, extracted_times, Russo_array)

		# print(Russo_array[0:20, 0:5])
		# print(np.shape(Russo_array))

		#Output file as CSV
		#np.savetxt("posttraining_stim" + str(jj+1) + "_Russo.csv", Russo_array, delimiter=',')

	return 0

#Load spike ids and times
def load_spikes(stimuli_iter):
	spike_ids = np.genfromtxt('output_spikes_posttraining_stim' + str(stimuli_iter+1) +'SpikeIDs.txt')
	spike_times = np.genfromtxt('output_spikes_posttraining_stim' + str(stimuli_iter+1) +'SpikeTimes.txt')
	return (spike_ids, spike_times)

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
	  #print(Russo_array[ii, 0:-1])

	  if math.isnan(Russo_array[ii, 0]) == 1: #If the first element is NaN, the entire row is (i.e. the neuron never spiked)
	    Russo_array[ii, 0] = np.random.random()*np.max(extracted_times) #Assigns the neuron a single spike, the time of which is sampled from a continuous uniform distribution

	return(Russo_array)



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
	test_params = {'extracted_layer' : 1, 'excit_dim' : 5}
	test_extracted_ids = np.array([4, 1, 1, 3, 1, 2])
	test_max_spikes = 3
	test_extracted_times = np.array([0.04, 0.08, 0.45, 0.9, 1.2, 4.0])
	
	exp = np.array([[0.08, 0.45, 1.2], [4.0, math.nan, math.nan], [0.9, math.nan, math.nan], [np.float, math.nan, math.nan], [0.04, math.nan, math.nan]])

	test_Russo_array = initialize_Russo(test_params, test_max_spikes)

	obs = populate_Russo(test_params, test_extracted_ids, test_extracted_times, test_Russo_array)

	assert np.all(obs[0, 0:3] == exp[0, 0:3])

	#NB: only the first line is tested automatically
	if params['manual_test_to_screen'] == 1:	
		print("The expected Russo array (first layer extraction) is \n")
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
	test_params = {'extracted_layer' : 2, 'excit_dim' : 5}
	test_extracted_ids = np.array([10, 6, 6, 8, 6, 7])
	test_max_spikes = 3
	test_extracted_times = np.array([0.04, 0.08, 0.45, 0.9, 1.2, 4.0])
	
	exp = np.array([[0.08, 0.45, 1.2], [4.0, math.nan, math.nan], [0.9, math.nan, math.nan], [np.float, math.nan, math.nan], [0.04, math.nan, math.nan]])

	test_Russo_array = initialize_Russo(test_params, test_max_spikes)

	obs = populate_Russo(test_params, test_extracted_ids, test_extracted_times, test_Russo_array)

	assert np.all(obs[0, 0:3] == exp[0, 0:3])

	#NB: only the first row is tested automatically; below offers a manual printout
	if params['manual_test_to_screen'] == 1:	
		print("The expected Russo array (higher layer extraction) is \n")
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




#Load .txt files containing neuron id's and spikes
# for jj in range(0, num_stimuli):
# 	spike_ids = np.genfromtxt('output_spikes_posttraining_stim' + str(jj+1) +'SpikeIDs.txt')
# 	spike_times = np.genfromtxt('output_spikes_posttraining_stim1SpikeTimes.txt')

# 	#Extract the neurons of interest
# 	extracted_mask = np.where((excit_dim*(extracted_layer-1) < spike_ids) & (spike_ids <= extracted_layer*excit_dim)) #Returns an array of indeces for the neurons in the layer of interest

# 	extracted_ids = np.take(spike_ids, extracted_mask) #Returns an array of spike IDs, restricted to the layer of interest
# 	extracted_times = np.take(spike_times, extracted_mask) #Returns an array of spike times, restricted to the layer of interest

# 	extracted_ids = np.reshape(extracted_ids, len(extracted_ids[0])) #Re-shaping the array into a column array
# 	extracted_ids = extracted_ids.astype(int) #Convert from spike output (float)

# 	#Identify the neuron that spikes the max number of times, and return the number of times it spikes, used later
# 	max_spikes = np.max(np.bincount(extracted_ids))

# 	#Initialize a NaN array with rows = number of unique neurons in the layer, and columns = number of spikes of the maximally active neuron
# 	Russo_array = np.zeros([excit_dim, max_spikes])
# 	Russo_array[:, :] = np.nan

# 	print(max_spikes)
# 	print(np.max(extracted_times))

# 	#Iterate through each neuron of interest, inserting its spikes into the Russo-suitable array; if a neuron never spikes, insert a single random spike
# 	for ii in range(0, len(Russo_array[:, 0])):
# 	  #Extract a binary mask containing the indeces of when the neuron of interest has fired
# 	  temp_mask = np.where(extracted_ids == (excit_dim*(extracted_layer - 1) + ii + 1))
# 	  #Use the mask to identify all the spike times associated with that neuron, and assign it to the 'Russo_array'
# 	  Russo_array[ii, 0:(np.size(np.take(extracted_times, temp_mask)))] = np.take(extracted_times, temp_mask)
# 	  if math.isnan(Russo_array[ii, 0]) == 1: #If the first element is NaN, the entire row is (i.e. the neuron never spiked)
# 	    Russo_array[ii, 0] = np.random.random()*np.max(extracted_times) #Assigns the neuron a single spike, time of which is sampled from a continuous uniform distribution

# 	#print(Russo_array[0:20, 0:5])
# 	#print(np.shape(Russo_array))

# 	#Output file as CSV
# 	#np.savetxt("posttraining_stim" + str(jj+1) + "_Russo.csv", Russo_array, delimiter=',')








