import argparse
import datetime
import numpy as np
from utils import *

import random
from sklearn.decomposition import sparse_encode
from sklearn import random_projection
from sklearn.decomposition import sparse_encode
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from sklearn.preprocessing import MinMaxScaler

from ksvd import KSVD
#from svm import svm_problem, svm_parameter
#from svmutil import svm_train, svm_predict, svm_save_model, svm_read_problem
from sklearn import svm

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

TRAIN_SET_RATIO = 0.8

def sparse_coding(n_atom, target_s, input_x, out_dir):
	# sklearn random sampled dictionary
	# sklearn function
	'''
	dictionary = random.sample(input_x, n_atom)
	code = sparse_encode(input_x, dictionary)
	'''
	# pyksvd package
	dictionary, code = KSVD(input_x, n_atom, target_s, 100, print_interval = 1)
	x_recovered = np.dot(code, dictionary)
	error = [np.linalg.norm(e) for e in (input_x - x_recovered)]
	#np.set_printoptions(precision=3, suppress=True)
	#print code
	#print dictionary
	with open('{}/atoms'.format(out_dir), "w") as op:
		for component in dictionary:
			line = ', '.join(str(round(e,3)) for e in component)
			op.write( line + '\n')
	with open('{}/codes'.format(out_dir), "w") as op:
		for coefficient in code:
			line = ', '.join(str(round(e,3)) for e in coefficient)
			op.write( line + '\n')
	with open('{}/coding_error'.format(out_dir), "w") as op:
		for epsilon in error:
			op.write( str(epsilon) + '\n')
	return code

def countZeros(input_x):
	x_sorted = np.array([sorted(instance) for instance in input_x], copy=True)
	num_zero = [max(x_sorted[:,i]) for i in xrange(x_sorted.shape[1])].count(0)
	return num_zero
	
	

def reduction(input_x, out_dir):
	transformer = random_projection.GaussianRandomProjection(MEASUREMENT)
	data_reduced = transformer.fit_transform(code)
	with open('{}/projection'.format(out_dir), "w") as op:
		for component in data_reduced:
			line = ', '.join(str(round(e,3)) for e in component)
        		op.write( line + '\n')
	return data_reduced

def readLabel(filenames):
	label = []
	for filename in filenames:
		with open(filename, 'r') as fr:
			for line in fr:
				label.append(LABEL_DICT[ line.rstrip().split(';')[1] ])
	return label
def writeFeature(fileName, instances, label=[]):
	# wtite features into libsvm format
	if len(label) == 0:
		label = ['-1']*len(instances)
	elif len(label) != len(instances):
		print 'ERROR : len(label) != len(instances)'
	with open(fileName, 'w') as fw:
		for j, features in enumerate(instances):
			feature_str = ''
			for i, f in enumerate(features):
				if f != 0:
					feature_str = ' '.join( [feature_str, ':'.join([str(i),str(f)])] )
			print >> fw, ' '.join([ str(int(label[j])), feature_str ])
	return

def readFeature(fileName, featureNum):
	features = []
	labels = []
	with open(fileName, 'r') as fr:
		for line in fr:
			line = line.split()
			instance = [0]*featureNum
			for f in line[1:]:
				i = int( f.split(':')[0] )
				instance[i] = float(f.split(':')[-1])
			features.append(instance)
			labels.append(int(line[0]))
	return features, labels

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

	sensor_data = []
	timestamps = []
	for path in [filename]:
		with Timer('open {} with PIR, Light, Sound sensors ...'.format(path)):
			data = np.genfromtxt(path, usecols=[1, 4, 13, 16, 18, 26, 31, 32, 37, 38, 39, 40, 9, 11, 22, 23, 41, 10, 12, 24, 25, 29, 30, 42, 43, 44], delimiter=',').tolist()
			timestamp = np.genfromtxt(path, usecols=[0] , delimiter=',').tolist()
			print "# of data:", len(data)
			sensor_data.extend(data)
			timestamps.extend(timestamp)
	with Timer('Standardizing ...'):
		from sklearn.preprocessing import MinMaxScaler, StandardScaler
		stdScalar = StandardScaler() # feature range (0,1)
		data_standardized = stdScalar.fit_transform(sensor_data)
	print data_standardized.shape
	cov = np.cov(np.array(sensor_data).T)
	U, s, V = np.linalg.svd(cov, full_matrices=True)
	print s
	cov = np.cov(data_standardized.T)
	U, s, V = np.linalg.svd(cov, full_matrices=True)
	print s
