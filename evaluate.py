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

def readLabel(filenames):
	label = list()
	for filename in filenames:
		with open(filename, 'r') as fr:
			for line in fr:
				label.append(LABEL_DICT[ line.rstrip().split(';')[1] ])
	return label

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
	argparser.add_argument('out_dir', type=str, help='the file of output data')
	argparser.add_argument('num_measurement', type=str, help='number of measurements')
	args = argparser.parse_args()
	args = vars(args)
	out_dir = args['out_dir']
	MEASUREMENT = int(args['num_measurement'])
	## SVM training
	X_train, Y_train = readFeature('{}/svm_train'.format(out_dir), MEASUREMENT)
	X_test, Y_test = readFeature('{}/svm_test'.format(out_dir), MEASUREMENT)

	#clf = svm.SVC(kernel='rbf', class_weight={0: 60, 1: 3, 2: 1})
	clf = svm.SVC(kernel='poly', class_weight={0: 60, 1: 3, 2: 1})
	clf.fit(X_train, Y_train)

	p_labels = clf.predict(X_test)
	print confusion_matrix(Y_test, p_labels)
	print f1_score(Y_test, p_labels, average='weighted')  
