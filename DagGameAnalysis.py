import numpy as np
import random
import sys
from math import *

# input root index i = 1
def inorder(i):
	global values
	if i < (2**(n-1)):
		if values[2*i] == -2:
			left = inorder(2*i)
		if values[(2*i) + 1] == -2:
			right = inorder((2*i) + 1)
	else:
		left = values[2*i]
		right = values[(2*i) + 1]
	#  paul and carole's strategy	
	if first == 'Paul':
		if floor(log(i,2)) % 2:
			values[i] = min(left,right)
		else:
			values[i] = max(left,right)
	else:
		if floor(log(i,2)) % 2:
			values[i] = max(left,right)
		else:
			values[i] = min(left,right)
	return values[i]

if __name__ == '__main__':
	n = sys.argv[1] # number of turns/levels in the game tree
	first = sys.argv[2] # who goes first

	# -2 used as placeholder for unvisited node
	values = np.concatenate((np.array([-2 for i in xrange(1,((2**n)+1))]), \
			np.array([random.uniform(-1,1) for i in xrange(((2**n)+1),(2**(n+1))+1)])))
	inorder(1)

	# first = 'Carole'
	# n=3
	# values=np.array([-2,-2,-2,-2,-2,-2,-2,-2,0.3,0.4,-0.9,0.1,0,0.8,0.75,-0.2])
	# inorder(1)

