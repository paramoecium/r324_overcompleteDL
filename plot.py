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

def plot_ksvd_error_recover():
	label_file = './truth_data-2014-11-02_to_2014-11-16.txt'
	error_file = './coding_error'
	error = np.loadtxt(error_file)

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
			c = collections.BrokenBarHCollection([(timestamp[0]+interval.total_seconds()/86400, float(5*60)/86400)], (0,2), facecolor=backgroundColor, alpha=0.5)
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
if __name__ == '__main__':
