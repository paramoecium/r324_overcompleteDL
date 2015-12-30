import argparse
import numpy as np
from utils import *

if __name__ == '__main__':
	argparser = argparse.ArgumentParser()
	argparser.add_argument('sparse_code', type=str, help='the sparse encoding')
	argparser.add_argument('out_dir', type=str, help='the file of output data')
	args = argparser.parse_args()
	args = vars(args)

	code_file = args['sparse_code']
	out_dir = args['out_dir']
	
	code = np.loadtxt(code_file, delimiter=',')
	print code

	with Timer('Sparse Coding ...'):
		from reduce import *
		code = sparse_coding(N_ATOM, code, out_dir)
		print 'number of zeros: {}/{}'.format(countZeros(code), code.shape[1])
