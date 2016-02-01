from __future__ import division
import numpy as np
import argparse
import hmm
''' 
model = hmm.MultinomialHMM(n_components=n_states)
'''
if __name__ == '__main__':
	argparser = argparse.ArgumentParser()
	argparser.add_argument('hmm_filename', type=str, help='the data of label and word array')
	argparser.add_argument('out_dir', type=str, help='the file of output data')
	args = argparser.parse_args()
	args = vars(args)
	
	filename = args['hmm_filename']
	out_dir = args['out_dir']

	input_array = np.loadtxt(filename, delimiter=' ')
	states = input_array[:,0]
	observations = input_array[:,1:]
	## Toy example
	states = ["Rainy", "Sunny"]
	n_states = len(states)

	observations = [[1,2], [2,4], [4,5], [5,5]]
	n_observations = len(observations)

	start_probability = np.array([0.6, 0.4])

	transition_probability = np.array([
	  [0.7, 0.3],
	  [0.4, 0.6]
	])

	emission_probability = np.array([
	  [0.1, 0.4, 0, 0.5],
	  [0.6, 0.3, 0, 0.1]
	])

	model = hmm.MultinomialHMM(n_components=n_states)
	model._set_startprob(start_probability)
	model._set_transmat(transition_probability)
	model._set_emissionprob(emission_probability)

	# predict a sequence of hidden states based on visible states
	bob_says = [0, 3, 1, 1, 3, 0]
	logprob, alice_hears = model.decode(bob_says, algorithm="viterbi")
	print "Bob says:", ", ".join(map(lambda x: str(observations[x]), bob_says))
	print "Alice hears:", ", ".join(map(lambda x: states[x], alice_hears))
