#!/usr/bin/env python3

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib

#Generates ground-truth data that can be used to perform integration tests on the analysis code used in Polychrony Pytools, and the Matlab 'Russo algorithm'
#Inserts the activation of several hand-crafted assemblies into Poisson spiking data; the test is successful if the information measure of each assembly 
#matches the ground truth

#The data mimics that of the 'binary network' where two stimuli are presented in an alternating fashion to two different sides of a feed-forward network,
#and the activity is analysed in the final layer

params = {'number_of_presentaitons' : 50,
	'duration_of_presentations' : 0.2,
	'jitter_std' : 0.0,
	'Poisson_rate' : 25,
	'refractory_duration' : 0.003}

def main(params):

	#Create array for all the spiking data

	assembly_IDs, assembly_times = create_assemblies()

	for stimulus_iter in range(2):
		spike_IDs = []
		spike_times = []

		#Iterate through the number of stimuli presentations
		for presentation_iter in range(params['number_of_presentaitons']):
			#Generate the spike times for each assembly in the presentation window
			temp_spike_IDs, temp_spike_times = generate_window_spikes(params, presentation_iter, stimulus_iter, assembly_IDs, assembly_times)

			#Sort these
			sorted_spike_IDs, sorted_spike_times = sort_spikes(temp_spike_IDs, temp_spike_times)

			#Add spikes drawn from Poisson firing rate such that the firing rate of each neuron in the window is at the desired threshold
			window_spike_IDs, window_spike_times = add_Poisson_firing(params, presentation_iter, sorted_spike_IDs, sorted_spike_times)

			spike_IDs.extend(window_spike_IDs)
			spike_times.extend(window_spike_times)


		spike_IDs = np.asarray(spike_IDs)
		spike_times = np.asarray(spike_times)

		mask = spike_times <1.0
		plt.scatter(spike_times[mask], spike_IDs[mask])
		plt.show()

		np.savetxt('Processing_Data/output_spikes_posttraining_stim' + str(stimulus_iter+1) + 'SpikeIDs.txt', spike_IDs)
		np.savetxt('Processing_Data/output_spikes_posttraining_stim' + str(stimulus_iter+1) + 'SpikeTimes.txt', spike_times)

	return None


def create_assemblies():
	#Specify the 5 assemblies, each testing for a different possibility
	#NB that in Spike, neuron IDs begin at 1, and all times are in seconds
	assembly_IDs = np.zeros((5,5))
	assembly_times = np.zeros((5,5))
	#PNG carrying maximal information about stimulus 1
	assembly_IDs[0, :] = np.arange(1,6)
	assembly_times[0, :] = [0.003, 0.0, 0.006, 0.012, 0.006]

	#PNG carrying no information about either stimulus, as it is active during every stimulus presentation
	assembly_IDs[1, :] = np.arange(6,11)
	assembly_times[1, :] = [0.0, 0.009, 0.012, 0.003, 0.0]

	#PNG carrying no information about either stimulus, as it is active 50% of the time to each stimulus
	assembly_IDs[2, :] = np.arange(11,16)
	assembly_times[2, :] = [0.015, 0.012, 0.0, 0.003, 0.006]

	#PNG carrying maximal information, but the PNG is in fact synchronous, and thus information can be fully described by synchrony
	assembly_IDs[3, :] = np.arange(16,21)
	assembly_times[3, :] = [0.0, 0.001, 0.0, 0.001, 0.0]

	#PNG that also occurs to stimulus 2, but in 'different' neurons which are just the shifted versions (i.e. by indices) of the same neurons in the stimulus 1 stream
	assembly_IDs[4, :] = np.arange(21,26)
	assembly_times[4, :] = [0.003, 0.0, 0.006, 0.015, 0.003]

	return assembly_IDs, assembly_times


#For each assembly, generate the spikes that occur within the given presentation window
def generate_window_spikes(params, presentation_iter, stimulus_iter, assembly_IDs, assembly_times):

	temp_spike_times = []
	temp_spike_IDs = []

	if stimulus_iter == 0:
		#Iterate through each assembly
		for assembly_iter in range(5):
			#All assemblies but number 3 always occur for stimulus 1; assembly 3 has a 50% probability of occuring
			if assembly_iter==2 and np.random.random()<=0.5:
				pass
			else:
				#The start time of an assembly's activation is determined by a uniform distriution over the current presentation window
				# - note this window is slightly shortened by the maximum lag in an assembly, ensuring it's activity remains in the window
				#It is offset by the current presentation iteration, and some additional noise (jitter) is added to each spike-time
				temp_spike_times.extend(np.random.random()*(params['duration_of_presentations']-0.015) + 
					assembly_times[assembly_iter,:] + 
					presentation_iter*params['duration_of_presentations'] +
					np.random.normal(0, params['jitter_std'], size=5))
				temp_spike_IDs.extend(assembly_IDs[assembly_iter,:])

	if stimulus_iter == 1:
		for assembly_iter in range(5):
			#Assembly 3 has a 50% probability of occuring
			if assembly_iter==2 and np.random.random()<=0.5:
				pass
			#Assembly 1 and 4 never occur for stimulus 2
			elif assembly_iter == 0 or assembly_iter == 3:
				pass
			#Assembly 5 always occurs, but the neuron IDs are shifted
			elif assembly_iter==4:
				temp_spike_times.extend(np.random.random()*(params['duration_of_presentations']-0.015) + 
					assembly_times[assembly_iter,:] + 
					presentation_iter*params['duration_of_presentations'] +
					np.random.normal(0, params['jitter_std'], size=5))
				temp_spike_IDs.extend(assembly_IDs[assembly_iter,:]+25)
			#Assembly 2 always occurs for stimulus 2
			else:
				temp_spike_times.extend(np.random.random()*(params['duration_of_presentations']-0.015) + 
					assembly_times[assembly_iter,:] + 
					presentation_iter*params['duration_of_presentations'] +
					np.random.normal(0, params['jitter_std'], size=5))
				temp_spike_IDs.extend(assembly_IDs[assembly_iter,:])

	return temp_spike_IDs, temp_spike_times


def sort_spikes(temp_spike_IDs, temp_spike_times):

	sorting_indices = np.argsort(temp_spike_times)
	sorted_spike_IDs = np.take_along_axis(np.asarray(temp_spike_IDs), sorting_indices, axis=0)
	sorted_spike_times = np.take_along_axis(np.asarray(temp_spike_times), sorting_indices, axis=0)

	return sorted_spike_IDs, sorted_spike_times


#Randomly sample and add Poisson data until the firing rate for each neuron is the same as the desired Poisson rate
def add_Poisson_firing(params, presentation_iter, sorted_spike_IDs, sorted_spike_times):

	#Iterate through each neuron; remember that they are indexed in spike_IDs from 1, not 0
	for neuron_iter in range(50):
		#Extract the number of spikes associated with that neuron, and calculate it's rate
		firing_rate = np.sum(sorted_spike_IDs==neuron_iter+1)/params['duration_of_presentations']
		#While its firing rate is below the desired threshold, continue adding spikes
		while firing_rate < params['Poisson_rate']:

			#Generate a spike time from a uniform random distrubtion, over the interval of interest (i.e. Poisson-like)
			spike_time = np.random.random()*params['duration_of_presentations'] + presentation_iter*params['duration_of_presentations']

			current_times = sorted_spike_times[np.where(sorted_spike_IDs==neuron_iter+1)]
			#If a randomly sampled neuron fires at the same time as a neuron that is already firing (within the refractory period), then re-draw that sample
			if np.any((current_times >= spike_time-params['refractory_duration']) & (current_times <= spike_time+params['refractory_duration'])):
				pass
			else:
				insertion_index = np.searchsorted(sorted_spike_times, spike_time)
				sorted_spike_times = np.insert(sorted_spike_times, insertion_index, spike_time)
				sorted_spike_IDs = np.insert(sorted_spike_IDs, insertion_index, neuron_iter+1)
				assert len(sorted_spike_IDs) == len(sorted_spike_times), "Number of insertions unequal."
				firing_rate = np.sum(sorted_spike_IDs==neuron_iter+1)/params['duration_of_presentations']

	return sorted_spike_IDs, sorted_spike_times

main(params)




