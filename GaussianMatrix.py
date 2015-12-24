import numpy as np
from math import log, ceil
from sklearn import random_projection


class Gaussian_Matrix:
	def __init__(self, delta, N, s, epsilon):
		self.delta = delta      ## distortion (0,1)
		self.N = N              ## original dimensionality
		self.sparsity = s       ## s-sparse, ||x||_0 <= s
		self.epsilon = epsilon  ## 1-epsilon probability
		self.m = self.num_of_measurement(delta, N, s, epsilon)
		self.transformer = random_projection.GaussianRandomProjection(self.m)
	def num_of_measurement(self, delta, N, s, epsilon):
		C = 80.098
		m = ceil( C*(delta**-2)*(s+s*log(N*s**-1)+log(2*epsilon**-1)) )
		print C*(delta**-2)
		print s
		print s*log(N*s**-1)
		print log(2*epsilon**-1)
		return int(m)
	def get_m(self):
		return self.m
	def projection(self, array):
		return self.transformer.fit_transform(array)
if __name__ == '__main__':
	gaussian_matrix = Gaussian_Matrix(0.1, 2000, 100, 0.4)
	print gaussian_matrix.get_m()
