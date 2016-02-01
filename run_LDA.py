from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import numpy as np
import argparse

from utils import *

WINDOW_SIZE = 60*10 # 5min before, 5 min after

if __name__ == '__main__':
	argparser = argparse.ArgumentParser()
	argparser.add_argument('windowed_svm_filename', type=str, help='the svm-format data')
	argparser.add_argument('featureNum_svm_file', type=str, help='number of features in the svm file')
	argparser.add_argument('model_dir', type=str, help='the dir of lda model')
	argparser.add_argument('out_dir', type=str, help='the dir of output data')
	args = argparser.parse_args()
	args = vars(args)

	filename = args['windowed_svm_filename']
	featureNum = int(args['featureNum_svm_file'])
	model_dir = args['model_dir']
	out_dir = args['out_dir']
	
	with Timer('Loading file ...'):
		labels = []
		documents = []
		with open(filename, 'r') as fr:
			for line in fr:
				document = []
				line = line.split()
				instance = [0]*featureNum
				for f in line[1:]:
					index, value = [float(s) for s in f.split(':')]
					#TODO np.sign --> ?
					word = str( int((index%WINDOW_SIZE) * np.sign(value)) )
					document.append(word)
				labels.append(int(line[0]))
				documents.append(document)
	with Timer('Generating corpus ...'):
		dictionary = corpora.Dictionary(documents)
		corpus = [dictionary.doc2bow(doc) for doc in documents]
	'''
	with Timer('Generating LDA models ...'):
		ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=100)
		ldamodel.save('{}/lda.model'.format(out_dir))
	print ldamodel.print_topics(num_topics=3,num_words=4)
	'''
	ldamodel = gensim.models.ldamodel.LdaModel.load('{}/lda.model'.format(model_dir))
	X = []
	for bow in corpus:
		distribution = ldamodel.get_document_topics(bow, minimum_probability=0)
		distribution = [p[1] for p in distribution]
		X.append(distribution)

	np.savetxt('{}/LDA_topics'.format(out_dir), X, delimiter=',')

	from svm_bracketing import bracketing
	bracketing(X, labels)
