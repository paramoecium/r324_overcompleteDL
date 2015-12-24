import argparse
import numpy as np
import shutil

def check(index, vector):
	# import pdb; pdb.set_trace()
	for i, v in enumerate(vector[:45]):
		if np.isnan(v):
			print 'missing value in line', index, ', attr ', i

def smooth(x):
	for i, line in enumerate(x):
		for i2, v2 in enumerate(line[:45]):
			if np.isnan(v2):
				print 'missing value in line', i, ', attr ', i2
				x[i][i2] = x[i-1][i2]

def run(in_dir, out_dir):
	with open('{}/filename.list'.format(in_dir), 'r') as fp:
		filenames = fp.read().splitlines()

	for file in filenames:
		x = np.genfromtxt('{}/{}'.format(in_dir,file), delimiter=',')
		smooth(x)
		x = x.astype('str')
		with open('{}/{}'.format(out_dir,file), 'w') as op:
			for line in x:
				print >> op, ", ".join(line)

	template = '{}/filename.list'
	shutil.copyfile(template.format(in_dir), template.format(out_dir))

def main():
	argparser = argparse.ArgumentParser()
	argparser.add_argument('in_dir', type=str, help='the file of output data')
	argparser.add_argument('out_dir', type=str, help='the file of output data')
	args = argparser.parse_args()
	args = vars(args)

	run(args['in_dir'], args['out_dir'])

if __name__ == '__main__':
	main()