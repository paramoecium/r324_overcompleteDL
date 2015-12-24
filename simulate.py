import random
import numpy as np
import math
from cvxpy import *

N = 2000
k = 300

ITERATION = 100

if __name__ == '__main__':
	#for m in xrange(N,2*k-1,-50):
	for m in xrange(1200,1000,-10):
		successful = 0
		for j in xrange(ITERATION):
			support = random.sample(range(N),k)
			x0 = np.zeros(N)
			x0[support] = np.random.rand(k,1)
			#print x
			A = np.random.uniform(-1,1,(m,N))/(2*m/float(3))**-2
			b = A.dot(x0) ## b = A * x0
			#print b
			# Construct the problem.
			x = Variable(N)
			objective = Minimize(pnorm(x, 1))
			constraints = [A*x==b]
			prob = Problem(objective, constraints)

			# The optimal objective is returned by prob.solve().
			result = prob.solve()
			# The optimal value for x is stored in x.value.
			#print x.value[support]
			#print x0[support]
			good = 0
			for i in xrange(N):
				if (math.fabs(x.value[i]-x0[i])<10**-8):
					good += 1
			if good == N:
				successful += 1
		print '(N={},k={})'.format(N,k), 'm={}'.format(m), '{}/{}={}'.format(successful, ITERATION, successful/float(ITERATION))
		with open('simlog4', 'a') as fw:
			print >> fw, '(N={},k={})'.format(N,k), 'm={}'.format(m), '{}/{}={}'.format(successful, ITERATION, successful/float(ITERATION))
