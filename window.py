import argparse
import datetime
import numpy as np
from utils import *
from collections import Counter

import random

SENSOR_DICT = {
	# PIR, 34 index is always broken and not provide any info
	'PIR': (1, 4, 13, 16, 18, 26, 31, 32, 37, 38, 39, 40),
	'Temp': (2, 5, 7, 14, 19, 27, 35),
	'Humi': (3, 6, 8, 15, 20, 28, 36),
	'Light': (9, 11, 22, 23, 41),
	'Sound': (10, 12, 24, 25, 29, 30, 42, 43, 44),
	'Magnet': (17, 21, 33),
	'WallBtn': (45,46,47,48),
}
DATATYPE_DICT = { # 0 means discrete, 1 means continuous
	'PIR': 0,
	'Temp': 1,
	'Humi': 1,
	'Light': 1,
	'Sound': 1,
	'Magnet': 1,
	'WallBtn': 0,
}
LABEL_DICT = {
 	'selfStudy': 0,
	'meeting': 1,
	'vacant': 2,
}
TIME_INTERVAL = 5 # 5min before, 5 min after

def window(input_x, input_timestamps, out_dir, label_filename, mode='hmm'):
	if mode=='svm':
		save_file_name = '{}/svm_sparse_windowed_total'.format(out_dir)
	elif mode=='hmm':
		save_file_name = '{}/hmm_word_windowed_total'.format(out_dir)
		from sklearn import mixture
		dpgmm = mixture.DPGMM(n_components=10, covariance_type='full', alpha=10)
		with Timer('Fitting DPGMM ...'):
			dpgmm.fit(input_x)
	else:
		print 'ERROR:mode'
		return
	open(save_file_name,'w').close() ## create empty file to be appended

	lines = []
	label_for_plot = []
	with open(label_filename, 'r') as fp:
		for line in fp:
			lines.append( line.rstrip().split(';') )

	assert len(input_x)==len(input_timestamps)
	data_windowed = []
	sample_num = len(input_x) / (TIME_INTERVAL*60)
	label_num = len(lines)
	ptr = 1
	label_ptr = 1 ## skip the first line
	while ptr < sample_num and label_ptr < label_num:
		current_stamp = datetime.datetime.strptime(lines[label_ptr][0], '%Y-%m-%d %H:%M:%S')
		while datetime.datetime.fromtimestamp(input_timestamps[(ptr-1)*TIME_INTERVAL*60]) < current_stamp - datetime.timedelta(seconds=60*TIME_INTERVAL):
			#print 'skip1', datetime.datetime.fromtimestamp(input_timestamps[(ptr-1)*TIME_INTERVAL*60]), current_stamp - datetime.timedelta(seconds=60*TIME_INTERVAL)
			ptr += 1
		while datetime.datetime.fromtimestamp(input_timestamps[(ptr+1)*TIME_INTERVAL*60 - 1]) > current_stamp + datetime.timedelta(seconds=60*TIME_INTERVAL-1):
			#print 'skip2', datetime.datetime.fromtimestamp(input_timestamps[(ptr+1)*TIME_INTERVAL*60 - 1]), current_stamp + datetime.timedelta(seconds=60*TIME_INTERVAL-1)
			label_ptr += 1
			current_stamp = datetime.datetime.strptime(lines[label_ptr][0], '%Y-%m-%d %H:%M:%S')
		assert datetime.datetime.fromtimestamp(input_timestamps[(ptr-1)*TIME_INTERVAL*60]) == current_stamp - datetime.timedelta(seconds=60*TIME_INTERVAL)
		assert datetime.datetime.fromtimestamp(input_timestamps[(ptr+1)*TIME_INTERVAL*60 - 1]) == current_stamp + datetime.timedelta(seconds=60*TIME_INTERVAL-1)
		#print ptr, label_ptr, current_stamp, datetime.datetime.fromtimestamp(input_timestamps[(ptr-1)*TIME_INTERVAL*60]), datetime.datetime.fromtimestamp(input_timestamps[(ptr+1)*TIME_INTERVAL*60 - 1])
		label = LABEL_DICT[lines[label_ptr][1].rstrip()]
		label_for_plot.append(label)
		if mode=='svm':
			instance_cascade = []
			for instance in input_x[(ptr-1)*TIME_INTERVAL*60: (ptr+1)*TIME_INTERVAL*60]:
				instance_cascade.extend(instance)
			with open(save_file_name, 'a') as fw:
				feature_str = ''
				for i, f in enumerate(instance_cascade):
					if f != 0:
						feature_str = ' '.join( [feature_str, ':'.join([str(i),str(f)])] )
				print >> fw, ' '.join([ str(int(label)), feature_str ])
		elif mode=='hmm':
			words = dpgmm.predict(input_x[(ptr-1)*TIME_INTERVAL*60: (ptr+1)*TIME_INTERVAL*60])
			cnt = Counter(words)
			print cnt
			with open(save_file_name, 'a') as fw:
				feature_str = ''
				for key in list(cnt.keys()):
					feature_str = ' '.join( [feature_str, str(key)] )
				print >> fw, ' '.join([ str(int(label)), feature_str ])
		ptr += 1
		label_ptr += 1
	np.savetxt('{}/label_for_plot'.format(out_dir), label_for_plot)

if __name__ == '__main__':
	argparser = argparse.ArgumentParser()
	argparser.add_argument('raw_filename', type=str, help='the raw data')
	argparser.add_argument('label_filename', type=str, help='the label data')
	argparser.add_argument('out_dir', type=str, help='the file of output data')
	args = argparser.parse_args()
	args = vars(args)

	out_dir = args['out_dir']
	filename = args['raw_filename']
	label_file = args['label_filename']
	'''
	sensor_data = []
	timestamps = []
	for path in [filename]:
		with Timer('open {} with PIR, Light, Sound sensors ...'.format(path)):
			data = np.genfromtxt(path, usecols=[1, 4, 13, 16, 18, 26, 31, 32, 37, 38, 39, 40, 9, 11, 22, 23, 41, 10, 12, 24, 25, 29, 30, 42, 43, 44], delimiter=',').tolist()
			timestamp = np.genfromtxt(path, usecols=[0] , delimiter=',').tolist()
			print "# of data:", len(data)
			sensor_data.extend(data)
			timestamps.extend(timestamp)
	with Timer('Normalizing ...'):
		from sklearn.preprocessing import MinMaxScaler
		normalizer = MinMaxScaler() # feature range (0,1)
		data_normalized = normalizer.fit_transform(sensor_data)
	with Timer('Sparse Coding ...'):
		from reduce import *
		num_ataom = 200
		target_sparsity = 5
		code = sparse_coding(num_ataom, target_sparsity, data_normalized, out_dir)
		print 'number of zeros: {}/{}'.format(countZeros(code), code.shape[1])
	'''
	timestamps = np.genfromtxt(filename, usecols=[0] , delimiter=',').tolist()
	with Timer('Loading codes ...'):
		code = np.loadtxt('output_14_ksvd/100a_2s/codes', delimiter=',')
	print code.shape
	with Timer('Sliding Window ...'):
		window(code, timestamps, out_dir, label_file,'hmm')
	#label = readLabel([label_file])[1:]
	#writeFeature('{}/svm_sparse_windowed_total'.format(out_dir), data_windowed, label)
	'''
	from reduce import *
	with Timer('Sliding Window ...'):
		data_windowed = window(data_normalized, out_dir)
		print 'data_windowed:', data_windowed.shape
	label = readLabel([args['label_filename']])[1:]
	writeFeature('{}/svm_original_total'.format(out_dir), data_windowed, label)
	'''
