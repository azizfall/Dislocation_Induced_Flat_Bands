import numpy as np
import math
import matplotlib.pyplot as plt
from random import randrange
from random import random 
import sys
import copy

from numba import jit, types, typed , typeof, float64, complex128
from numba.experimental import jitclass
from numba.typed import List
import numba as nb
import ctypes

import vegas 


import multiprocessing 
from multiprocessing import Pool



spec = [
	('ktf', types.float64),
	('l', types.int64),
	('ne', types.float64),
	('kf', types.float64), 
	('p0', float64[:]),
	('cutoff',types.float64),
	('E_eval', types.float64)
]
@jitclass(spec)
class Integrand_class:
	def __init__(self): 
		self.ktf = 0.2336 
		self.l = 0 
		#doping density of 10^20 atoms cm^-3
		self.ne = 1.481847e-5
		self.kf = (3*(np.pi**2)*self.ne)**(1.0/3.0)
		#self.p0 = self.kf*np.array([1.0/np.sqrt(2),1.0/np.sqrt(2)])
		self.p0 = 0.5*np.array([1.0/np.sqrt(3),1.0/np.sqrt(3), 1.0/np.sqrt(3)])
		self.cutoff = 30*self.ktf
		self.E_eval = 0.3

	def set_l(self, l_in):
		self.l = l_in

	def V_k(self, k : np.ndarray):
		return 1/(np.dot(k,k) + self.ktf**2)

	def Energy(self, k : np.ndarray): 
		return np.dot(k,k)/(2)

	def G0_func(self, k : np.ndarray, E : float64, ep : float64):
		ek = self.Energy(k)
		den = (E - ek)**2 + ep**2
		num = E - ek -1j*ep
		return num/den  

	def Integrand(self, k : np.ndarray):

		ep0 = 10
		Ac = 1.2 
		ep = (ep0/(Ac**self.l))

		V_total = self.V_k(k - self.p0)*self.V_k(self.p0 - k)
		result = V_total*self.G0_func(k, self.E_eval, ep)

		return result.real


def Run_Partition(l):
	#l = args_tuple 
	Integrand_object = Integrand_class()
	a = -Integrand_object.cutoff
	b = Integrand_object.cutoff
	int_range = [] 

	for i in range(3):
		int_range.append([a,b]) 

	Integrand_object.set_l(l)
	integ = vegas.Integrator(int_range)
	result = integ(Integrand_object.Integrand, nitn=40, neval=1000000)

	return result.mean


def main(): 
	l_max = 60

	all_tasks = [] 
	for l in range(59,l_max):
		all_tasks.append(l)

	with Pool() as pool:
		all_results = pool.map(Run_Partition, all_tasks)

	results = {} 

	for task, result in zip(all_tasks, all_results):
		l = task
		results[l] = result 

	outF = open("Coulomb_Scattering_Si_ls.txt", "w")

	for l in range(59,l_max): 
		outF.write("l=" + str(l) + " " + str(results[l]) + "\n")


if __name__ == "__main__":
    main()




















