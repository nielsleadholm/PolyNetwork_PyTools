#!/usr/bin/env python3

import numpy as np
import os
import math
import matplotlib.pyplot as plt

#######
# The following is a utility function for use with the Spike GPU-powered simulator
# It enables a user to specify the probability with which groups of neurons share synapses
# In addition, the number of synapses, synaptic delays, and initial weights can be specified
# It is intended to be used with the 'binary stream' architecture, where a 'left' and 'right' side of the network have different connectivities
#######

#prob_connection parameters define the probability of a connection existing between a given pair of neurons, and is specified for ips (ipsilateral) and contra (contralateral) connections of the binary stream
#mult_synapses defines, if a connection exists, the precise number of synapses shared
#delay_mean and std (in ms) determine the parameters of the log-normal distribution (parameters in standard normal rather than log-normal form) used to generate delays (output as seconds)
#ffTrue_latFalse_bool determines if feed-forward or lateral connections should be generated (set to True if the former)
#num_layers should *not* include the input source layer
#simulation_timestep is used to ensure delays do not fall below the minimum time-resoluton of the Spike simulator
params = {'ips_prob_connection' : 0.8,
'contra_prob_connection' : 0.2,
'mult_synapses' : 1,
'delay_mean' : 3.4,
'delay_std' : 2.3,
'weight_min' : 0.005,
'weight_max' : 0.015,
'ffTrue_latFalse_bool' : True,
'input_layer_dim' : 16*16,
'output_layer_dim' : 16*16,
'simulation_timestep' : 0.0001,
'num_layers' : 3}

#Set-up synapses
def main(params):

	for layer in range(params['num_layers']):
		#Iterate through the left and right hand 'streams' that are receiving or projecting connections
		for source_L_R in range(2):
			for receiver_L_R in range(2):
				#Ipsilateral connections
				if source_L_R == receiver_L_R:
					temp_prob_connection = params['ips_prob_connection']
				#Contralateral connections
				else:
					temp_prob_connection = params['contra_prob_connection']
				input_IDs_list, output_IDs_list, num_connections = sample_connections(params, temp_prob_connection)
				delays_list = generate_delays(params, num_connections)
				weights_list = generate_weights(params, num_connections)
				
				output_synapse_data(input_IDs_list, output_IDs_list, delays_list, weights_list, (layer, source_L_R, receiver_L_R))

	return 0

#Randomly determine which pre and post synaptic neurons share a connection
def sample_connections(params, temp_prob_connection):

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


#Create a matching delays vector containing the synapse delays, determined by the log-normal distribution
def generate_delays(params, num_connections):

	#Convert the mean and std of the normal distribution to the log-normal distribution form
	lognorm_mu = math.log(params['delay_mean']) - (0.5)*math.log((params['delay_std']/params['delay_mean'])**2 + 1)
	lognorm_std = math.sqrt(math.log((params['delay_std']/params['delay_mean'])**2 + 1))

	delays_list = []

	for ii in range(num_connections):
		for jj in range(params['mult_synapses']):

			delays_list.append(np.random.lognormal(lognorm_mu, lognorm_std)/1000 + params['simulation_timestep'])
			#Note that a small value is added to the delay; this ensures the delays are sufficiently large so as to not cause floating point errors later in Spike

	return delays_list

def generate_weights(params, num_connections):

	weights_list = []

	for ii in range(num_connections):
		for jj in range(params['mult_synapses']):
			#Asign each synapse a weight that is uniformly distributed between the min and max weight range
			weights_list.append((params['weight_max'] - params['weight_min']) * np.random.uniform() + params['weight_min'])

	return weights_list

def output_synapse_data(input_IDs_list, output_IDs_list, delays_list, weights_list, file_name_tuple):

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




