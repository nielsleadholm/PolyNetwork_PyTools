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

#When creating conductiong delays, the number of synapses are used to determine even intervals up until delay-max
#Gaussian noise (delay_STD) is then added to these values

#In the current version, weights are sampled from a uniform distribution, and so some neurons may receive in total
#more weight than others (i.e. they are not normalized to each neuron)

Gaussian_bool = False

if Gaussian_bool == False:
	#prob_connection defines the probability of a connection existing between a given pair of neurons
	#prob_connection can be set to 1 if a form of all-to-all connectivity is desired
	#mult_synapses defines, if a connection exists, the precise number of synapses shared
	#delay_max is the maximum synaptic delay in seconds
	#num_layers should *not* include the input source layer
	params = {'prob_connection' : 0.2,
	'mult_synapses' : 1,
	'delay_max' : 0.010,
	'Gamma_delay_distribution':True,
	'Gamma_delay_params':(1.5,1.5),
	'weight_min' : 0.005,
	'weight_max' : 0.015,
	'ffTrue_latFalse_bool' : False,
	'input_layer_dim' : 16*16,
	'output_layer_dim' : 16*16,
	'simulation_timestep' : 0.0001,
	'delay_STD' : 1.5,
	'num_layers' : 3}
else:
	params = {'Gaussian_std' : 3,
	'num_unique_presynaptic_neurons' : 3,
	'mult_synapses' : 4,
	'delay_max' : 0.020,
	'weight_min' : 0.005,
	'weight_max' : 0.015,
	'ffTrue_latFalse_bool' : 1,
	'input_layer_dim' : 32*32,
	'output_layer_dim' : 32*32,
	'simulation_timestep' : 0.0001,
	'delay_STD' : 1.5,
	'num_layers' : 3}

#Set-up synapses
def main(params):

	for layer in range(params['num_layers']):
		for source_L_R in range(2):
			for receiver_L_R in range(2):
				if source_L_R == receiver_L_R:
					temp_prob_connection = params['prob_connection']

				#If the connections are contralateral, assign the q = 1 - p probability of connections

				# *** edit - changed this to be 0.01 by default, as I want essentially no lateral connectivity, and setting a non-zero value prevents error messages
				else:
					temp_prob_connection = 0.001 #1 - params['prob_connection'] 
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

	if Gaussian_bool == 0:
		input_IDs_list, output_IDs_list = random_sample(params, temp_prob_connection, input_IDs_list, output_IDs_list)
	else:
		input_IDs_list, output_IDs_list = Gaussian_sample(params, input_IDs_list, output_IDs_list)

	num_connections = int(len(input_IDs_list)/params['mult_synapses'])
	assert(len(input_IDs_list)%params['mult_synapses']==0)

	return input_IDs_list, output_IDs_list, num_connections

def random_sample(params, temp_prob_connection, input_IDs_list, output_IDs_list):

	for in_ID in range(params['input_layer_dim']):
		for out_ID in range(params['output_layer_dim']):
			if np.random.uniform(0,1) <= temp_prob_connection:
				#Create multiple synapses per connection
				for ii in range(params['mult_synapses']):
					input_IDs_list.append(in_ID)
					output_IDs_list.append(out_ID)

	return input_IDs_list, output_IDs_list

def Gaussian_sample(params, input_IDs_list, output_IDs_list):

#Iterate through each post-synaptic neuron (which is either in the same 
	#layer for lateral, or the layer above for feedforward)
#Based on 2-D Gaussian noise, and the number of unique inputs, sample neurons from the layer below
	#Account for accidentally selecting the same neuron on one of these iterations
#When a connection is made, make multiple synapses 

# *** how do I deal with layers? how does the original synapse method work with it

	post_neuron_counter = 0

	for ii in range(0, np.sqrt(params['input_layer_dim'])):
		for jj in range(0, np.squrt(params['input_layer_dim'])):
			for kk in range(0, params['num_unique_presynaptic_neurons']):
				x_Gaussian = np.random.normal(0, Gaussian_SD)
				y_Gaussian = np.random.normal(0, Gaussian_SD)
				x_pre = int(np.round(ii + x_Gaussian))
				y_pre = int(np.round(jj + y_Gaussian))
				#print("Neuron post index is %d" % post_neuron_counter)
				#Checks if randomly selected pre-synaptic neuron is outside of the possible indeces; if passes, then creates synapse
				if  0 <= x_pre < x_dim and 0 <= y_pre < y_dim:
					pre_index = x_pre * 32 + y_pre + 1
					#Iterate through eah additional synapse for a given connection pair
					for mm in range(0, num_synapses_per_pair):
						#f_PreIDs.write("-%d" % (ii+1))
						if post_neuron_counter == magic_index:
							visual_array[x_pre, y_pre] += 1
						f_PostIDs.write("%d\n" % post_neuron_counter)
				else:
					outside_array_counter += 1
					#print("Indexed outside of array") #If outside of indeces, the particular synapse is skipped over. Note therefore that neurons at the edge of the layer will have on average fewer input synapses
	
			post_neuron_counter += 1

	return input_IDs_list, output_IDs_list



#Create a matching delays vector containing the synapse delays, determined by delay_max
def generate_delays(params, num_connections):

	delays_list = []

	for ii in range(num_connections):
		delay_value = 0
		for jj in range(params['mult_synapses']):

			if params['Gamma_delay_distribution'] == False:
				delay_value += params['delay_max']/params['mult_synapses']
				#Add normally distribute noise, corrected for the unit (seconds) of the delay
				delays_list.append(abs(delay_value + (np.random.normal(0, params['delay_STD'])/1000)) + params['simulation_timestep']) #Take the absolute value to ensure no negative delays
				#Note that in addition, a small value is added to the weight; this ensures the weights are sufficiently large so as to not cause floating point errors later in Spike
			
			elif params['Gamma_delay_distribution'] == True:
				delays_list.append((np.random.gamma(params['Gamma_delay_params'][0], params['Gamma_delay_params'][1]) / 1000) + params['simulation_timestep'])

	return delays_list

def generate_weights(params, num_connections):

	weights_list = []

	for ii in range(num_connections):
		for jj in range(params['mult_synapses']):
			#Asign each synapse a weight that is uniformly distributed between the mean and max weight range
			weights_list.append((params['weight_max'] - params['weight_min']) * np.random.uniform() + params['weight_min'])

	print(np.mean(weights_list))

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
		#print(len(input_IDs_list))
		#print(input_IDs_list[100:105])
		f1.write(input_IDs_list)

	#Append to the newly created connectivity data file
	with open(file_name, 'ab') as f2:
		output_IDs_list = np.asarray(output_IDs_list)
		output_IDs_list = output_IDs_list.astype(np.float32)
		#print(len(output_IDs_list))
		#print(output_IDs_list[100:105])
		f2.write(output_IDs_list)

	with open(file_name, 'ab') as f3:
		weights_list = np.asarray(weights_list)
		weights_list = weights_list.astype(np.float32)
		#print(len(weights_list))
		#print(weights_list[100:105])
		f3.write(weights_list)

	with open(file_name, 'ab') as f4:
		delays_list = np.asarray(delays_list)
		delays_list = delays_list.astype(np.float32)
		#print(len(delays_list))
		#print(delays_list[100:110])
		f4.write(delays_list)

	return 0

main(params)





