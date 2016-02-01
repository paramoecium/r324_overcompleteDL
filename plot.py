import numpy as np
import matplotlib.pyplot as plt

def plot_ksvd_error_iteration():
	C_s = np.logspace(10, 100, 15)
	error = [0.000143303, 0.000123136, 0.000111273, 0.000105127, 9.91063e-05, 9.3651e-05, 9.02013e-05, 8.75813e-05, 8.50846e-05, 8.35431e-05, 7.37909e-05, 6.9442e-05, 6.82171e-05, 6.65125e-05, 6.73644e-05, 6.88675e-05, 6.66383e-05, 6.52689e-05, 6.32862e-05]
	iteration = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
	import matplotlib.pyplot as plt
	plt.figure(1, figsize=(4, 3))
	plt.clf()
	plt.plot(iteration, error)
	plt.ylabel('Approximation Error')
	plt.xlabel('Iteration')
	plt.show()
	return

def plot_svm_accuracy():
	C_s = np.logspace(-5, 10, 15)
	scores=[0.59492696014364765, 0.69063455683965014, 0.73921819670856026, 0.7955112019373628, 0.78236209889304231, 0.78310596211367134, 0.78310596211367134, 0.78310596211367134, 0.78310596211367134, 0.78310596211367134, 0.78310596211367134, 0.78310596211367134, 0.78310596211367134, 0.78310596211367134, 0.78310596211367134]

	scores_std=[0.079456533268676369, 0.07512384956366884, 0.074008040858436527, 0.083030187964027904, 0.087203509825868247, 0.087753305468542153, 0.087753305468542153, 0.087753305468542153, 0.087753305468542153, 0.087753305468542153, 0.087753305468542153, 0.087753305468542153, 0.087753305468542153, 0.087753305468542153, 0.087753305468542153]

	plt.figure(1, figsize=(4, 3))
	plt.clf()
	plt.semilogx(C_s, scores)
	plt.semilogx(C_s, np.array(scores) + np.array(scores_std), 'b--')
	plt.semilogx(C_s, np.array(scores) - np.array(scores_std), 'b--')
	locs, labels = plt.yticks()
	plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
	plt.ylabel('CV score')
	plt.xlabel('Parameter C')
	plt.ylim(0, 1.1)
	plt.show()
	return

def plot_ksvd_recover_error():
	label_file = './truth_data-2014-11-02_to_2014-11-16.txt'
	error_file = './coding_error'
	CV_prefiction = './tree_svm_CV_predicted_label'
	error = np.loadtxt(error_file)
	label_predicted = np.loadtxt(CV_prefiction)

	import datetime
	from matplotlib.dates import DateFormatter, drange	
	unit = 1 # seconds
	fig = plt.figure()
	ax = fig.add_subplot(111)
	timestamp = drange(datetime.datetime( 2014, 11, 2), datetime.datetime( 2014, 11, 16), datetime.timedelta(seconds=unit))
	#plt.scatter(timestamp, error, alpha=0.5)
	#timestamp = timestamp[86400*1:86400*2]
	#timestamp = timestamp[3600*9:3600*14]	
	#error = error[86400*1:86400*2]
	#error = error[3600*9:3600*14]
	plt.plot(timestamp, error)
	ax.set_xlabel('time'.format(unit))
	ax.xaxis.set_major_formatter( DateFormatter('%m-%d %H:%M') )
	ax.set_xlim(timestamp[0], timestamp[-1])
	ax.set_ylim(0)
	ax.set_ylabel('recovery error')
	#ax.grid(True)
	import matplotlib.collections as collections
	import time
	for i in xrange(len(label_predicted)):
		if label_predicted[i] == 2:
			backgroundColor = 'green'
		elif label_predicted[i] == 0:
			backgroundColor = 'yellow'
		elif label_predicted[i] == 1:
			backgroundColor = 'red'
		else:
			print "unexpected label!!"
		c = collections.BrokenBarHCollection([(timestamp[0]+float(5*60*(i+1))/86400, float(5*60)/86400)], (1,2), facecolor=backgroundColor, alpha=0.5)
		ax.add_collection(c)
	with open(label_file, 'r') as fp:
		for line in fp:
			line = line.rstrip().split(';')
			interval = datetime.datetime.strptime(line[0], '%Y-%m-%d %H:%M:%S') - datetime.datetime( 2014, 11, 2)
			label = line[1]
			if "vacant" in label:
				backgroundColor = 'green'
			elif "selfStudy" in label:
				backgroundColor = 'yellow'
			elif "meeting" in label:
				backgroundColor = 'red'
			else:
				print "unexpected label!!"
			#print (timestamp[0]+interval.total_seconds()/86400, float(5*60)/86400)
			c = collections.BrokenBarHCollection([(timestamp[0]+interval.total_seconds()/86400, float(5*60)/86400)], (0,1), facecolor=backgroundColor, alpha=0.5)
			#c = collections.BrokenBarHCollection([(timestamp[0], 86400)], (0,2), facecolor=backgroundColor, alpha=0.5)
			ax.add_collection(c)
	plt.show()
	return

def plot_atom_MDS():
	from sklearn import manifold
	from sklearn.metrics import euclidean_distances
	atom_file = './atoms'
	atoms = np.loadtxt(atom_file, delimiter=',')
	similarities = euclidean_distances(atoms)
	seed = np.random.RandomState(seed=3)

	mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed, dissimilarity="precomputed", n_jobs=1)
	pos = mds.fit(similarities).embedding_
	plt.figure(1, figsize=(4, 3))
	plt.clf()
	plt.scatter(pos[:, 0], pos[:, 1], s=20, c='black')
	plt.gca().xaxis.set_major_locator(plt.NullLocator())
	plt.gca().yaxis.set_major_locator(plt.NullLocator())
	plt.legend(('atoms', 'XXX'), loc='best')
	plt.show()
	return

def plot_code_MDS():
	'''
## for workstation
import numpy as np
sample_index = np.random.randint(4032,size=1000)
params = ['100a_2s','100a_5s','100a_10s']
for param in params:
	codes = np.genfromtxt('output_14_ksvd/{}/codes'.format(param), delimiter=',')
	r_codes = codes[sample_index * 300]
	np.savetxt('../r_codes_{}'.format(param), r_codes, delimiter=',')

atoms = np.loadtxt('output_14_ksvd/{}/atoms'.format(param), delimiter=',')
raw = np.dot(codes, atoms)
r_codes_original = raw[sample_index * 300]
np.savetxt('../r_codes_original', r_codes_original, delimiter=',')

LABEL_DICT = {'selfStudy': 0, 'meeting': 1, 'vacant': 2}
label = []
with open('data/truth_data-2014-11-02_to_2014-11-16.txt', 'r') as fr:
	for line in fr:
		label.append(LABEL_DICT[ line.rstrip().split(';')[1] ])

sample_label = []
for i in sample_index:
	sample_label.append(label[i])

np.savetxt('../sample_label', sample_label)
	'''
	from sklearn import manifold
	from sklearn.metrics.pairwise import pairwise_distances
	sample_label = np.loadtxt('sample_label')
	COLOR_DICT = {0:'yellow', 1:'red', 2:'green'}
	sample_color = [COLOR_DICT[l] for l in sample_label]
	seed = np.random.RandomState(seed=3)

	fig = plt.figure(1, figsize=(8, 6))
	plt.clf()
	fig.canvas.set_window_title('100a_ks_1000sample')

	codes = np.loadtxt('r_codes_original', delimiter=',')
	similarities = pairwise_distances(codes, metric='euclidean')
	#similarities = pairwise_distances(codes, metric='cosine')
	mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed, dissimilarity="precomputed", n_jobs=1)
	pos = mds.fit(similarities).embedding_
	ax = plt.subplot("141")
	ax.set_title("original")
	ax.scatter(pos[:, 0], pos[:, 1], s=20, c=sample_color, alpha=0.5)
	plt.gca().xaxis.set_major_locator(plt.NullLocator())
	plt.gca().yaxis.set_major_locator(plt.NullLocator())

	codes = np.loadtxt('r_codes_100a_10s', delimiter=',')
	similarities = pairwise_distances(codes, metric='euclidean')
	#similarities = pairwise_distances(codes, metric='cosine')
	mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed, dissimilarity="precomputed", n_jobs=1)
	pos = mds.fit(similarities).embedding_
	ax = plt.subplot("142")
	ax.set_title("100a_10s")
	ax.scatter(pos[:, 0], pos[:, 1], s=20, c=sample_color, alpha=0.5)
	plt.gca().xaxis.set_major_locator(plt.NullLocator())
	plt.gca().yaxis.set_major_locator(plt.NullLocator())

	codes = np.loadtxt('r_codes_100a_5s', delimiter=',')
	similarities = pairwise_distances(codes, metric='euclidean')
	#similarities = pairwise_distances(codes, metric='cosine')
	mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed, dissimilarity="precomputed", n_jobs=1)
	pos = mds.fit(similarities).embedding_
	ax = plt.subplot("143")
	ax.set_title("100a_5s")
	ax.scatter(pos[:, 0], pos[:, 1], s=20, c=sample_color, alpha=0.5)
	plt.gca().xaxis.set_major_locator(plt.NullLocator())
	plt.gca().yaxis.set_major_locator(plt.NullLocator())

	codes = np.loadtxt('r_codes_100a_2s', delimiter=',')
	similarities = pairwise_distances(codes, metric='euclidean')
	#similarities = pairwise_distances(codes, metric='cosine')
	mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed, dissimilarity="precomputed", n_jobs=1)
	pos = mds.fit(similarities).embedding_
	ax = plt.subplot("144")
	ax.set_title("100a_2s")
	ax.scatter(pos[:, 0], pos[:, 1], s=20, c=sample_color, alpha=0.5)
	plt.gca().xaxis.set_major_locator(plt.NullLocator())
	plt.gca().yaxis.set_major_locator(plt.NullLocator())
	
	plt.show()
	return

def plot_atom_n_code_MDS():
	from sklearn import manifold
	from sklearn.metrics.pairwise import pairwise_distances
	seed = np.random.RandomState(seed=3)
	atom_file = './atoms'
	code_file = './r_codes_100a_2s'
	atoms = np.loadtxt(atom_file, delimiter=',')
	codes = np.loadtxt(code_file, delimiter=',')
	sample_label = np.loadtxt('sample_label')
	COLOR_DICT = {0:'yellow', 1:'red', 2:'green'}
	sample_color = [COLOR_DICT[l] for l in sample_label]

	fig = plt.figure(1, figsize=(4, 3))
	plt.clf()
	fig.canvas.set_window_title('100-atom 2-sparse')

	#similarities = pairwise_distances(atoms, metric='euclidean')
	similarities = pairwise_distances(atoms, metric='cosine')
	mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed, dissimilarity="precomputed", n_jobs=1)
	pos = mds.fit(similarities).embedding_
	ax = plt.subplot("131")
	ax.set_title("100 atoms")
	ax.scatter(pos[:, 0], pos[:, 1], s=20, c='blue')
	plt.gca().xaxis.set_major_locator(plt.NullLocator())
	plt.gca().yaxis.set_major_locator(plt.NullLocator())
	ax.legend(('atom', 'XXX'), loc='best')

	#similarities = pairwise_distances(codes, metric='euclidean')
	similarities = pairwise_distances(codes, metric='cosine')
	mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed, dissimilarity="precomputed", n_jobs=1)
	pos = mds.fit(similarities).embedding_
	ax = plt.subplot("132")
	ax.set_title("2-sparse codes of 1000 random samples")
	ax.scatter(pos[:, 0], pos[:, 1], s=20, c=sample_color, alpha=0.5)
	plt.gca().xaxis.set_major_locator(plt.NullLocator())
	plt.gca().yaxis.set_major_locator(plt.NullLocator())
	ax.legend(('code', 'XXX'), loc='best')

	data_recover = np.dot(codes, atoms)
	#similarities = pairwise_distances(np.concatenate((data_recover, atoms), axis=0), metric='euclidean')
	similarities = pairwise_distances(np.concatenate((data_recover, atoms), axis=0), metric='cosine')
	mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed, dissimilarity="precomputed", n_jobs=1)
	pos = mds.fit(similarities).embedding_
	ax = plt.subplot("133")
	ax.set_title("atoms + recovery")
	ax.scatter(pos[:len(data_recover), 0], pos[:len(data_recover), 1], s=20, c=sample_color, alpha=0.5)
	ax.scatter(pos[len(data_recover):, 0], pos[len(data_recover):, 1], s=20, c='blue')
	plt.gca().xaxis.set_major_locator(plt.NullLocator())
	plt.gca().yaxis.set_major_locator(plt.NullLocator())
	ax.legend(('sample', 'atom'), loc='best')

	plt.show()
	return

def plot_confusion_matrix():
	LABEL_DICT = {'selfStudy': 0, 'meeting': 1, 'vacant': 2}
	label_file = './truth_data-2014-11-02_to_2014-11-16.txt'
	#CV_prefiction = './CV_predicted_label'
	CV_prefiction = './tree_svm_CV_predicted_label'
	label_predicted = np.loadtxt(CV_prefiction)
	true_label = []
	with open(label_file, 'r') as fr:
		for line in fr:
			true_label.append(LABEL_DICT[ line.rstrip().split(';')[1] ])
	true_label = true_label[1:]

	from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
	print(classification_report(true_label, label_predicted, target_names=['selfStudy', 'meeting', 'vacant']))
	score = accuracy_score(true_label, label_predicted)
	cm = confusion_matrix(true_label, label_predicted)
	cm = np.array(cm)
	cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	print cm
	print cm_normalized
	plt.figure()
	title = 'Confusion matrix'
	cmap = plt.cm.Blues
	plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	label_names = np.array(['selfStudy','meeting','vacant'])
	plt.xticks(np.arange(len(label_names)), label_names, rotation=45)
	plt.yticks(np.arange(len(label_names)), label_names)
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()
	return

def plot_topics_n_label():
	import datetime
	from matplotlib.dates import DateFormatter, drange, AutoDateLocator
	import colorsys
	start_date = datetime.datetime( 2014, 9, 1)
	end_date = datetime.datetime( 2015, 9, 26)
	label_file = './truth_data-2014-09-01_to_2014-09-26.txt'
	topic_file = '10-topic_LDA_topics_2014-09-01_to_2014-09-26'
	unit = 300 # seconds
	day = 25
	import matplotlib.collections as collections

	topic_distribution = np.loadtxt(topic_file, delimiter=',')
	fig, ax = plt.subplots()
	fig.set_size_inches(3*day,1)
	timestamp = drange(start_date, end_date, datetime.timedelta(seconds=unit))
	label_timestamp = []
	with open(label_file, 'r') as fp:
		for line in fp:
			line = line.rstrip().split(';')
			currentTimestamp = datetime.datetime.strptime(line[0], '%Y-%m-%d %H:%M:%S')
			label_timestamp.append(currentTimestamp)
			interval = currentTimestamp - start_date
			label = line[1]
			if "vacant" in label:
				backgroundColor = 'green'
			elif "selfStudy" in label:
				backgroundColor = 'yellow'
			elif "meeting" in label:
				backgroundColor = 'red'
			else:
				print "unexpected label!!"
			#print (timestamp[0]+interval.total_seconds()/86400, float(5*60)/86400)
			c = collections.BrokenBarHCollection([(timestamp[0]+interval.total_seconds()/86400, float(5*60)/86400)], (0,1.2), facecolor=backgroundColor, alpha=0.5)
			#c = collections.BrokenBarHCollection([(timestamp[0], 86400)], (0,2), facecolor=backgroundColor, alpha=0.5)
			ax.add_collection(c)
	'''
	#for 112-day only
	with open('timestamps_for_plot_2014-09-01_to_2015-02-01', 'r') as fp:
		for line in fp:
			label_timestamp.append(datetime.datetime.strptime(line.rstrip(), '%Y-%m-%d %H:%M:%S'))
	'''
	print len(label_timestamp), topic_distribution.shape
	ax.set_xlabel('time')
	ax.xaxis.set_major_locator(AutoDateLocator(minticks=6*day))
	ax.xaxis.set_major_formatter( DateFormatter('%m-%d %H:%M') )
	ax.set_xlim(label_timestamp[0], label_timestamp[-1])
	ax.set_ylim(0,1.1)
	ax.set_ylabel('topic proportion')
	randHSVcolors = [(i/10.0,0.6,1) for i in xrange(10)]
	randRGBcolors = [colorsys.hsv_to_rgb(c[0],c[1],c[2]) for c in randHSVcolors]
	ax.stackplot(label_timestamp[1:], topic_distribution.T,colors=randRGBcolors)
	#ax.stackplot(label_timestamp, topic_distribution.T,colors=randRGBcolors)
	#plt.show()
	plt.savefig(topic_file+'.png', bbox_inches='tight', dpi=400)
	return
	
if __name__ == '__main__':
	#plot_ksvd_error_iteration()
	#plot_svm_accuracy()
	#plot_ksvd_recover_error()
	#plot_atom_MDS()
	#plot_code_MDS()
	#plot_atom_n_code_MDS()
	#plot_confusion_matrix()
	plot_topics_n_label()
