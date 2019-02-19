#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle


params = {'epsilon0' : 0.0005,
	'Russo_bin_size' : 0.003,
	'number_stimuli' : 2,
	'network_layer': 3,
	'epsilon1' : 0.015,
	'dataset_duration' : 400,
	'epsilon_iter_bool' : 1,
	'epsilon_iter_step' : 0.00005,
	'epsilon_max' : 0.005,
	'shuffle_Boolean' : 0,
	'Poisson_Boolean' : 0}


with open('epsilon_results_stim1_layer3.data', 'rb') as filehandle:
	epsilon_results = pickle.load(filehandle)

fig, ax = plt.subplots()

# *** NB I have adjusted x_axis to display Epsilon as I would ultimately prefer it (i.e. the whole range) rather than 
#as it is currently implemented (only one side of the window of spike capture) ***

x_axis = np.arange(1, len(epsilon_results[0, 0, 0, :])+1)*params['epsilon_iter_step']*1000*2

plt.scatter(x_axis, epsilon_results[0, 0, 0, :], label='Stimulus 1 Presentation', c='#e31a1c')
plt.scatter(x_axis, epsilon_results[1, 0, 0, :], label='Stimulus 2 Presentation', c='#2171b5')


#with open('epsilon_results_stim1_layer3_shuffled.data', 'rb') as filehandle:
#	epsilon_results = pickle.load(filehandle)

#plt.scatter(x_axis, epsilon_results[0, 0, 0, :], label='Stimulus 1 Presentation (Shuffled)', c='#e77072')
#plt.scatter(x_axis, epsilon_results[1, 0, 0, :], label='Stimulus 2 Presentation (Shuffled)', c='#62a6df')

ax.set_ylim(0, 1)
ax.set_ylabel('Proportion of Assembly Activations')
ax.set_xlabel('Epsilon (ms)')
plt.title('Stimulus 1 Associated Assembly')
plt.legend(loc='lower right')


plt.show()