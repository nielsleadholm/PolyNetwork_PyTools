#!/usr/bin/env python3

import numpy as np
np.set_printoptions(threshold=np.inf)
import os
import matplotlib.pyplot as plt


#######
# The following is a utility function for use with the Spike GPU-powered simulator
# It enables a user to specify the probability with which groups of neurons share synapses
# In addition, the number of synapses, synaptic delays, and initial weights can be specified
#######

#prob_connection defines the probability of a connection existing between a given pair of neurons
#prob_connection can be set to 1 if a form of all-to-all connectivity is desired
#mult_synapses defines, if a connection exists, the precise number of synapses shared
#delay_max is the maximum synaptic delay in seconds
#num_layers should *not* include the input source layer
params = {'prob_connection' : 0.6,
'mult_synapses' : 4,
'delay_max' : 0.010,
'weight_min' : 0.01,
'weight_max' : 0.1,
'ffTrue_latFalse_bool' : 0,
'input_layer_dim' : 5*5,
'output_layer_dim' : 5*5,
'delay_STD' : 0.75,
'num_layers' : 3}

#Set-up synapses
def main(params):

	for layer in range(params['num_layers']):
		for source_L_R in range(2):
			for receiver_L_R in range(2):
				if source_L_R == receiver_L_R:
					temp_prob_connection = params['prob_connection']
				#If the connections are contralateral, assign the q = 1 - p probability of connections
				else:
					temp_prob_connection = 1 - params['prob_connection']
				input_IDs_list, output_IDs_list, num_connections = sample_connections(params, temp_prob_connection)
				delays_list = generate_delays(params, num_connections)
				weights_list = generate_weights(params, num_connections)
				
				output_synapse_data(input_IDs_list, output_IDs_list, delays_list, weights_list, (layer, source_L_R, receiver_L_R))

	return 0

#Randomly determine which pre and post synaptic neurons share a connection
def sample_connections(params, temp_prob_connection):

	#Initialize list to hold synapse ID values
	input_IDs_list = []
	output_IDs_list = []

	for in_ID in range(params['input_layer_dim']):
		for out_ID in range(params['output_layer_dim']):
			if np.random.uniform(0,1) <= temp_prob_connection:
				#Create multiple synapses per connection
				for ii in range(params['mult_synapses']):
					input_IDs_list.append(in_ID)
					output_IDs_list.append(out_ID)


	num_connections = int(len(input_IDs_list)/params['mult_synapses'])
	assert(len(input_IDs_list)%params['mult_synapses']==0)

	return input_IDs_list, output_IDs_list, num_connections


#Create a matching delays vector containing the synapse delays, determined by delay_max
def generate_delays(params, num_connections):

	delays_list = []

	for ii in range(num_connections):
		delay_value = 0
		for jj in range(params['mult_synapses']):
			delay_value += params['delay_max']/params['mult_synapses']
			#Add normally distribute noise, corrected for the unit (seconds) of the delay
			delays_list.append(abs(delay_value + (np.random.normal(0, params['delay_STD'])/1000)) + (10**(-5))) #Take the absolute value to ensure no negative delays
			#Note that in addition, a small value is added to the weight; this ensures the weights are sufficiently large so as to not cause floating point errors later in Spike

	return delays_list

def generate_weights(params, num_connections):

	weights_list = []

	for ii in range(num_connections):
		for jj in range(params['mult_synapses']):
			#Asign each synapse a weight that is uniformly distributed between the mean and max weight range
			weights_list.append((params['weight_max'] - params['weight_min']) * np.random.uniform() + params['weight_min'])

	return weights_list

def output_synapse_data(input_IDs_list, output_IDs_list, delays_list, weights_list, file_name_tuple):

	#Overwrite any earlier generated connectivity data files
	if params['ffTrue_latFalse_bool'] == 1:
		file_name = 'Connectivity_Data_ff_' + str(file_name_tuple[0]) + str(file_name_tuple[1]) + str(file_name_tuple[2]) + '.syn'
	else:
		file_name = 'Connectivity_Data_lat_' + str(file_name_tuple[0]) + str(file_name_tuple[1]) + str(file_name_tuple[2]) + '.syn'
			
	with open(file_name, 'wb') as f1:
		#First convert the list into a numpy array and then specify data type to enable binary reading in C++ later
		input_IDs_list = np.asarray(input_IDs_list)
		input_IDs_list = input_IDs_list.astype(np.float32)
		print(len(input_IDs_list))
		print(input_IDs_list[100:105])
		f1.write(input_IDs_list)

	#Append to the newly created connectivity data file
	with open(file_name, 'ab') as f2:
		output_IDs_list = np.asarray(output_IDs_list)
		output_IDs_list = output_IDs_list.astype(np.float32)
		print(len(output_IDs_list))
		print(output_IDs_list[100:105])
		f2.write(output_IDs_list)

	with open(file_name, 'ab') as f3:
		weights_list = np.asarray(weights_list)
		weights_list = weights_list.astype(np.float32)
		print(len(weights_list))
		print(weights_list[100:105])
		f3.write(weights_list)

	with open(file_name, 'ab') as f4:
		delays_list = np.asarray(delays_list)
		delays_list = delays_list.astype(np.float32)
		print(len(delays_list))
		print(delays_list[100:105])
		f4.write(delays_list)

	return 0

main(params)





