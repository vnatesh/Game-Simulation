#!/usr/bin/env python
import numpy as np
import random
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from math import *
from multiprocessing import Pool
from scipy import stats
from scipy.stats import norm
from functools import partial 
from joblib import Parallel, delayed 

# 
def inorder(i, values, first):

	'''

	Performs in-order transversal of a binary tree, recursively
	filling in node payouts. Input root index i = 1. 
	-2 used as placeholder for nodes that don't have 
	all children with assigned payouts yet

	'''

	if values[i] == -2:
		left = inorder(2*i, values, first)
		right = inorder((2*i) + 1, values, first)
	else:
		return values[i]
	# paul and carole's strategy	
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


def playGame(n):
	first = 'Paul'
	values = np.concatenate((np.array([-2 for i in xrange(1,((2**n)+1))]), \
	np.array([random.uniform(-1,1) for i in xrange(((2**n)+1),(2**(n+1))+1)])))
	return inorder(1, values, first)


def runSim(n, trials):
	return [playGame(n) for i in xrange(trials)]


def parallelSim(numTrials):
    p = Pool(processes = 4)
    prod_x = partial(runSim, trials = numTrials)
    payouts = p.map(prod_x, range(22)) 
    return payouts


def histData(n, numTrials):
	return Parallel(n_jobs=4)(delayed(playGame)(n) for i in range(numTrials))
	

def plotSim():
	payouts = parallelSim(100)
	std = [np.std(i) for i in payouts[1:]]
	mean = [np.mean(i) for i in payouts[1:]]
	cv = (np.array(std) / abs(np.array(mean)))*100
	#
	plt.figure(1)
	plt.subplot(211)
	plt.plot(range(1,22), std, '-ro')
	plt.title("StdDev of Payout")
	plt.xlabel("n")
	plt.ylabel("stddev")
	plt.tight_layout()
	plt.xticks(range(1,22))
	#
	plt.subplot(212)
	plt.plot(range(1,22), mean, '-go')
	plt.title("Mean of Payout")
	plt.xlabel("n")
	plt.ylabel("payout")
	plt.tight_layout()
	plt.xticks(range(1,22))
	plt.show()
	#
	plt.figure(1)
	plt.plot(range(1,22), cv, '-bo')
	plt.title("Coefficient of Variation of Payout")
	plt.xlabel("n")
	plt.ylabel("percent CV")
	plt.tight_layout()
	plt.xticks(range(1,22))
	plt.show()


def plotHist():
	vals12 = histData(12,10000)
	vals15 = histData(15,10000)
	vals18 = histData(18,5818)
	#
	plt.figure(1)
	plt.subplot(311)
	plt.hist(vals12, normed = True, bins = 100, facecolor='g', alpha=0.75)
	xt = plt.xticks()[0]  
	xmin, xmax = min(xt), max(xt)  
	lnspc = np.linspace(xmin, xmax, len(vals12))
	# fit normal distr
	m, s = stats.norm.fit(vals12) # get mean and standard deviation  
	pdf_g = stats.norm.pdf(lnspc, m, s) # now get theoretical values in our interval  
	plt.plot(lnspc, pdf_g, label="Norm")
	plt.title("Distribution of Payouts for n = 12 (10000 trials)")
	plt.xlabel("payout")
	plt.ylabel("normed frequency")
	plt.tight_layout()
	plt.axis([-0.4,0,0,30])
	plt.grid([True])
	#
	#
	plt.subplot(312)
	plt.hist(vals15, normed = True, bins = 100, facecolor='g', alpha=0.75)
	xt = plt.xticks()[0]  
	xmin, xmax = min(xt), max(xt)  
	lnspc = np.linspace(xmin, xmax, len(vals15))
	# fit normal distr
	m, s = stats.norm.fit(vals15) # get mean and standard deviation  
	pdf_g = stats.norm.pdf(lnspc, m, s) # now get theoretical values in our interval  
	plt.plot(lnspc, pdf_g, label="Norm")
	plt.title("Distribution of Payouts for n = 15 (10000 trials)")
	plt.xlabel("payout")
	plt.ylabel("normed frequency")
	plt.tight_layout()
	plt.axis([0.1,0.4,0,15])
	plt.grid([True])
	#
	#
	plt.subplot(313)
	plt.hist(vals18, normed = True, bins = 50, facecolor='g', alpha=0.75)
	xt = plt.xticks()[0]  
	xmin, xmax = min(xt), max(xt)  
	lnspc = np.linspace(xmin, xmax, len(vals18))
	# fit normal distr
	m, s = stats.norm.fit(vals18) # get mean and standard deviation  
	pdf_g = stats.norm.pdf(lnspc, m, s) # now get theoretical values in our interval  
	plt.plot(lnspc, pdf_g, label="Norm")
	plt.title("Distribution of Payouts for n = 18 (5818 trials)")
	plt.xlabel("payout")
	plt.ylabel("normed frequency")
	plt.tight_layout()
	plt.axis([-0.4,0,0,30])
	plt.grid([True])
	plt.show()

# Function to confirm expected sign-pair probabilities
def pairProbs(n, trials):
	pp=[]
	nn=[]
	pn=[]
	nps=[]
	for t in xrange(trials):
		b = np.array([random.uniform(-1,1) for i in xrange(((2**n)+1),(2**(n+1))+1)])
		pospos=0
		negneg=0
		posneg=0
		negpos=0
		for i in xrange(0,len(b),2):
			if b[i] > 0 and b[i+1] > 0:
				pospos+=1
			elif b[i] < 0 and b[i+1] < 0:
				negneg+=1
			elif b[i] > 0 and b[i+1] < 0:
				posneg+=1
			elif b[i] < 0 and b[i+1] > 0:
				negpos+=1
		pp.append(pospos)
		nn.append(negneg)
		pn.append(posneg)
		nps.append(negpos)
	print "(+,+) " + str(np.mean(np.array(pp)) / (2**(n-1))) + '\n' \
	+ "(-,-) " + str(np.mean(np.array(nn)) / (2**(n-1))) + '\n' \
	+ "(+,-) " + str(np.mean(np.array(pn)) / (2**(n-1))) + '\n' \
	+ "(-,+) " + str(np.mean(np.array(nps)) / (2**(n-1)))


