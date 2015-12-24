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
from sklearn import cross_validation

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
	argparser.add_argument('svm_file', type=str, help='the file in libsvm format')
	argparser.add_argument('num_measurement', type=str, help='number of measurements')
	args = argparser.parse_args()
	args = vars(args)
	svm_file_path = args['svm_file']
	MEASUREMENT = int(args['num_measurement'])

	## SVM training
	X, Y = readFeature(svm_file_path, MEASUREMENT)

	#clf = svm.SVC(kernel='rbf', class_weight={0: 60, 1: 3, 2: 1})
	clf = svm.SVC(kernel='linear', class_weight='auto')
	#clf = svm.SVC(kernel='polyss', class_weight='auto')
	C_s = np.logspace(-10, 5, 15)

	scores = list()
	scores_std = list()
	for C in C_s:
		clf.C = C
		## poly
		'''
		clf.degree
		clf.gamma
		clf.coef0
		'''
		this_scores = cross_validation.cross_val_score(clf, X, Y, n_jobs=4) ## 4 CPU core
		scores.append(np.mean(this_scores))
		scores_std.append(np.std(this_scores))
	print scores
	print scores_std
