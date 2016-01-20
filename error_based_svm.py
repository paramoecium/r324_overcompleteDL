import numpy as np
output_dir = './output_14_ksvd'
TIME_INTERVAL = 5 # 5min before, 5min after
input_x = np.loadtxt('{}/coding_error'.format(output_dir))
data_windowed = list()
sample_num = len(input_x) / (TIME_INTERVAL*60)
for pt in xrange(1,sample_num):
	instance_cascade = list()
	for instance in input_x[(pt-1)*TIME_INTERVAL*60: (pt+1)*TIME_INTERVAL*60]:
		instance_cascade.extend([instance])
	data_windowed.append([np.mean(instance_cascade), np.var(instance_cascade)])
'''
with open('{}/coding_error_windowed'.format(output_dir), "w") as op:
	print len(data_windowed)
	for instance_cascade in data_windowed:
		line = ', '.join(str(e) for e in instance_cascade)
		op.write( line + '\n')
'''
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

label = readLabel(['./data/truth_data-2014-11-02_to_2014-11-16.txt'])[1:]

X = data_windowed
Y = label

from sklearn import svm
from sklearn import cross_validation

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
	this_scores = cross_validation.cross_val_score(clf, X, Y, n_jobs=4) ## 4 CPU core
	scores.append(np.mean(this_scores))
	scores_std.append(np.std(this_scores))
	print "accuracy mean:", scores[-1], "std:", scores_std[-1]

print scores
print scores_std
