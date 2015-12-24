import argparse
import datetime
import numpy as np
from utils import *
import os

import random
from sklearn.decomposition import sparse_encode
from sklearn import random_projection
from sklearn.decomposition import sparse_encode
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.externals import joblib

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
TIME_INTERVAL = 5 # 5min before, 5min after
TRAIN_SET_RATIO = 0.8

def window(input_x, out_dir):
	data_windowed = list()
	sample_num = len(input_x) / (TIME_INTERVAL*60)
	for pt in xrange(1,sample_num):
		instance_cascade = list()
		for instance in input_x[(pt-1)*TIME_INTERVAL*60: (pt+1)*TIME_INTERVAL*60]:
			instance_cascade.extend(instance)
		data_windowed.append(instance_cascade)
	with open('{}/windowed'.format(out_dir), "w") as op:
		print len(data_windowed)
		for instance_cascade in data_windowed:
			line = ', '.join(str(e) for e in instance_cascade)
        		op.write( line + '\n')
	return np.array(data_windowed, dtype=np.float)

def get_dictionary(n_atom, input_x):
	return random.sample(input_x, n_atom)

def sparse_coding(n_atom, input_x, out_dir):
	dictionary = get_dictionary(n_atom, input_x)
	code = sparse_encode(input_x, dictionary)
    
	np.set_printoptions(precision=3, suppress=True)
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
	return code

def sparsity(input_x):
	x_sorted = np.array([sorted(instance) for instance in input_x], copy=True)
	sparsity = [max(x_sorted[:,i]) for i in xrange(x_sorted.shape[1])].count(0)
	return sparsity
	
	

def reduction(eps, input_x, out_dir):
	print 'JL bound:', random_projection.johnson_lindenstrauss_min_dim(len(input_x[0]),eps),'(eps={})'.format(eps)
	transformer = random_projection.GaussianRandomProjection(50,eps)
	data_reduced = transformer.fit_transform(code)
	with open('{}/projection'.format(out_dir), "w") as op:
		for component in data_reduced:
			line = ', '.join(str(round(e,3)) for e in component)
        		op.write( line + '\n')
	return data_reduced

def readLabel(filenames):
	label = list()
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
	#argparser.add_argument('raw_data_dir', type=str, help='the directory of list')
	argparser.add_argument('raw_filename', type=str, help='the raw data')
	argparser.add_argument('label_filename', type=str, help='the label data')
	argparser.add_argument('out_dir', type=str, help='the file of output data')
	argparser.add_argument('n_atom', type=int, help='# of atoms in the dictionary')
	args = argparser.parse_args()
	args = vars(args)
	n_atom = args['n_atom']
	#raw_data_dir = args['raw_data_dir']
	out_dir = args['out_dir']
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	'''
	with open('{}/filename.list'.format(raw_data_dir), 'r') as fp:
		filenames = fp.read().splitlines()
	'''
	filenames = [args['raw_filename']]
	sensor_data = list()
	'''
	for filename in filenames:
		#path = '{}/{}'.format(raw_data_dir, filename)
		path = filename
		with Timer('open {} with PIR, Light, Sound sensors ...'.format(filename)):
			#data = np.genfromtxt(path, usecols=range(1,49)
			data = np.genfromtxt(path, usecols=[1, 4, 13, 16, 18, 26, 31, 32, 37, 38, 39, 40, 9, 11, 22, 23, 41, 10, 12, 24, 25, 29, 30, 42, 43, 44]
				, delimiter=',').tolist()
			print "# of data:", len(data)
			sensor_data.extend(data)
	with Timer('Sliding Window ...'):    
		data_windowed = window(sensor_data, out_dir)
		print 'data_windowed:', data_windowed.shape
	'''
	'''
	with Timer('Sparse Coding ...'):
		code = sparse_coding(n_atom, data_windowed, out_dir)
		print 'sparsity: {}/{}'.format(sparsity(code), code.shape[1])
	eps = 0.5
	with Timer('Random Projection ...'):
		data_reduced = reduction(eps, code, out_dir)
	reduced_dim = data_reduced.shape[1]
	with open('{}/reduction_setting'.format(out_dir), 'w') as fp:
		line = ', '.join(str(e) for e in [n_atom, eps])
		fp.write( line + '\n')
	'''
	'''
	label = readLabel([args['label_filename']])[1:]
	cutIndex = int(TRAIN_SET_RATIO*len(data_windowed))
	writeFeature('{}/svm_train_original'.format(out_dir), data_windowed[:cutIndex], label[:cutIndex]) 
	writeFeature('{}/svm_test_original'.format(out_dir), data_windowed[cutIndex:], label[cutIndex:]) 
	'''
	## SVM training
	#X_train, Y_train = readFeature('./svm_train',reduced_dim)
	X_train, Y_train = readFeature('{}/svm_train_original'.format(out_dir),15600)
	X_test, Y_test = readFeature('{}/svm_test_original'.format(out_dir), 15600)

	#clf = svm.SVC(kernel='rbf', class_weight={0: 60, 1: 3, 2: 1})
	clf = svm.SVC(kernel='linear', class_weight={0: 60, 1: 3, 2: 1})
	clf.fit(X_train, Y_train)
	joblib.dump(clf, '{}/svm.pkl'.format(out_dir))

	p_labels = clf.predict(X_test)
	print confusion_matrix(Y_test, p_labels)
	print f1_score(Y_test, p_labels, average='weighted')  
