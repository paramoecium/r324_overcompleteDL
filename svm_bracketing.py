import argparse
import datetime
import numpy as np
from utils import *

import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

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

def bracketing(X, Y):
	#clf = svm.SVC(kernel='rbf', class_weight={0: 60, 1: 3, 2: 1})
	clf = svm.SVC(kernel='linear', class_weight='auto')
	#clf = svm.SVC(kernel='polyss', class_weight='auto')
	C_s = np.logspace(-5, 10, 15)

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
		with Timer('Cross Validation(C={}) ...'.format(C)):
			this_scores = cross_validation.cross_val_score(clf, X, Y, n_jobs=4) ## 4 CPU core
		scores.append(np.mean(this_scores))
		scores_std.append(np.std(this_scores))
		print "accuracy mean:", scores[-1], "std:", scores_std[-1]
	print scores
	print scores_std

def tree_bracketing(X, Y):
	X = np.array(X)
	Y = np.array(Y)
	'''
	## find the best C for each level's svm
	print 'level 0: vacant vs not vacant'
	X_0 = X
	Y_0 = [int(y==2) for y in Y] ## vacant = 1, else = 0
	clf_0 = svm.SVC(kernel='linear', class_weight='auto')
	C_s = np.logspace(-5, 10, 15)
	scores = list()
	scores_std = list()
	for C in C_s:
		clf_0.C = C
		with Timer('Cross Validation(C={}) ...'.format(C)):
			this_scores = cross_validation.cross_val_score(clf_0, X_0, Y_0, n_jobs=4) ## 4 CPU core
		scores.append(np.mean(this_scores))
		scores_std.append(np.std(this_scores))
		print "accuracy mean:", scores[-1], "std:", scores_std[-1]
	C_0 = C_s[np.argmax(scores)]
	print 'best C:', C_0
	
	scores = list()
	scores_std = list()
	print 'level 1: selfStudy vs meeting'
	clf_1 = svm.SVC(kernel='linear', class_weight='auto')
	X_1 = X[Y[:]!=2]
	Y_1 = Y[Y[:]!=2]
	for C in C_s:
		clf_1.C = C
		with Timer('Cross Validation(C={}) ...'.format(C)):
			this_scores = cross_validation.cross_val_score(clf_1, X_1, Y_1, n_jobs=4) ## 4 CPU core
		scores.append(np.mean(this_scores))
		scores_std.append(np.std(this_scores))
		print "accuracy mean:", scores[-1], "std:", scores_std[-1]
	C_1 = C_s[np.argmax(scores)]
	print 'best C:', C_1
	'''
	from sklearn.cross_validation import KFold
	kf = KFold(len(X), n_folds=3, shuffle=True)
	scores = list()
	CV_prediction = np.zeros(len(Y))
	for train_index, test_index in kf:
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]
		## level 0
		Y_train_0 = [int(y==2) for y in Y_train] ## vacant = 1, else = 0
		clf_0 = svm.SVC(kernel='linear', class_weight='auto')
		clf_0.C = 0.193069772888
		with Timer('SVM level 0(C={}) ...'.format(clf_0.C)):
			clf_0.fit(X_train, Y_train_0)
			p_labels_0 = clf_0.predict(X_test)
		## level 1
		X_train_1 = X_train[Y_train[:]!=2]
		Y_train_1 = Y_train[Y_train[:]!=2]
		clf_1 = svm.SVC(kernel='linear', class_weight='auto')
		clf_1.C = 0.0163789370695
		with Timer('SVM level 1(C={}) ...'.format(clf_1.C)):
			clf_1.fit(X_train_1, Y_train_1)
			p_labels_1 = clf_1.predict(X_test)
		p_labels = [p_labels_1[i] if p_labels_0[i]==0 else 2 for i in xrange(len(p_labels_1))]
		score = accuracy_score(Y_test, p_labels)
		scores.append(score)
		CV_prediction[test_index] = p_labels
		print 'k-fold score:', score
	print "accuracy mean:", np.mean(scores), "std:", np.std(scores)
	np.savetxt('./output_14_ksvd/100a_2s/tree_svm_CV_predicted_label',CV_prediction,fmt='%i')

def best_CV_prediction(X, Y):
	X = np.array(X)
	best_svm_C = 26.8269579528
	clf = svm.SVC(kernel='linear', class_weight='auto')
	clf.C = best_svm_C
	with Timer('Cross Validation Prediction(C={}) ...'.format(best_svm_C)):
		Y_predicted = cross_validation.cross_val_predict(clf, X, Y, n_jobs=4) ## 4 CPU core
	print confusion_matrix(Y_predicted, Y)
	print f1_score(Y_predicted, Y, average='weighted')
	np.savetxt('./output_14_ksvd/100a_2s/CV_predicted_label',Y_predicted,fmt='%i')

if __name__ == '__main__':
	argparser = argparse.ArgumentParser()
	argparser.add_argument('svm_file', type=str, help='the file in libsvm format')
	argparser.add_argument('num_measurement', type=str, help='number of measurements')
	args = argparser.parse_args()
	args = vars(args)
	svm_file_path = args['svm_file']
	MEASUREMENT = int(args['num_measurement'])

	X, Y = readFeature(svm_file_path, MEASUREMENT)
	#bracketing(X, Y)
	#best_CV_prediction(X, Y)
	tree_bracketing(X, Y)
