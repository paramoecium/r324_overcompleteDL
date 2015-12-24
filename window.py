import argparse
import datetime
import numpy as np
from utils import *

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
TIME_INTERVAL = 5 # 5min before, 5min after

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

if __name__ == '__main__':
	argparser = argparse.ArgumentParser()
	argparser.add_argument('raw_filename', type=str, help='the raw data')
	argparser.add_argument('label_filename', type=str, help='the label data')
	argparser.add_argument('out_dir', type=str, help='the file of output data')
	args = argparser.parse_args()
	args = vars(args)
	
	out_dir = args['out_dir']
	filenames = [args['raw_filename']]
	sensor_data = list()
	for filename in filenames:
		path = filename
		with Timer('open {} with PIR, Light, Sound sensors ...'.format(filename)):
			data = np.genfromtxt(path, usecols=[1, 4, 13, 16, 18, 26, 31, 32, 37, 38, 39, 40, 9, 11, 22, 23, 41, 10, 12, 24, 25, 29, 30, 42, 43, 44]
				, delimiter=',').tolist()
			print "# of data:", len(data)
			sensor_data.extend(data)
	with Timer('Sliding Window ...'):    
		data_windowed = window(sensor_data, out_dir)
		print 'data_windowed:', data_windowed.shape
