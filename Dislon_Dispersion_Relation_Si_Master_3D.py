import numpy as np
import math
import matplotlib.pyplot as plt
from random import randrange, random
import sys
import os 
import copy

from numba import jit, types, typed, float64, complex128
from numba.experimental import jitclass
from numba.typed import List
import numba as nb
import ctypes

import vegas

from scipy.integrate import nquad 

import multiprocessing
from multiprocessing import Pool

from scipy.optimize import fsolve
from scipy.optimize import root



spec = [
    ('burg', types.float64),
    ('poi', types.float64),
    ('ktf', types.float64),
    ('Z_a', types.float64),
    ('nd', types.float64),
    ('l', types.int64),
    ('ne', types.float64),
    ('ni', types.float64),
    ('kf', types.float64),
    ('Ef', types.float64),
    ('p0', float64[:]),
    ('cutoff', types.float64),
    ('dip_arr', float64[:]),
    ('E_eval', types.float64)
]
@jitclass(spec)
class Integrand_class:
    def __init__(self):
        self.burg = 7.25571
        self.poi = 0.275
        self.ktf = 0.2336
        self.Z_a = 4.285
        self.nd = 0.007403
        self.l = 0
        # doping density of 10^20 atoms cm^-3
        self.ne = 1.481847e-5
        # dislocation density 7.5*10^(14) m^-2 used
        self.ni = 2.1e-6
        self.kf = (3*(np.pi**2)*self.ne)**(1.0/3.0)
        self.Ef = (self.kf**2)/(2)
        self.p0 = self.kf*np.array([1.0/np.sqrt(3), 1.0/np.sqrt(3), 1.0/np.sqrt(3)])
        self.cutoff = 30*self.ktf
        # dipole separation array
        self.dip_arr = self.burg*np.array([0.0, 1.0])
        self.E_eval = 0.3

    def set_l(self, l_in):
        self.l = l_in

    def set_dip_arr(self, sep_in):
        self.dip_arr[0] = sep_in[0]
        self.dip_arr[1] = sep_in[1]

    def set_p0(self, p_in):
        self.p0[0] = p_in[0]
        self.p0[1] = p_in[1]
        self.p0[2] = p_in[2]

    def set_Eval(self, E_in):
        self.E_eval = E_in

    def k_dot_F_dip(self, k: np.ndarray):
        # renormalized potential such that 
        # V(0) = 0
        if np.dot(k, k) == 0:
            return 0.0 + 1j*0.0

        dip_sep = np.zeros(2, dtype=np.float64)
        dip_sep[0] = self.dip_arr[0]
        dip_sep[1] = self.dip_arr[1]

        s_2 = k[0]**2 + k[1]**2
        num_term1 = (self.burg*k[1]*s_2*(1 - self.poi) - self.burg*((k[0])**2)*k[1])
        num_term2 = k[1]*(s_2*(1 - self.poi)*self.burg - self.burg*((k[1])**2))
        num = num_term1 + num_term2
        num = num*(np.cos(np.dot(k, dip_sep)) - 1j*np.sin(np.dot(k, dip_sep)) - 1)
        den = (s_2**2)*(1 - self.poi)

        return num/den

    def k_dot_F_avg(self, k: np.ndarray):
        # renormalized potential such that 
        # V(0) = 0
        if np.dot(k, k) == 0:
            return 0.0 + 1j*0.0

        s_2 = k[0]**2 + k[1]**2
        num_term1 = (self.burg*k[1]*s_2*(1 - self.poi) - self.burg*((k[0])**2)*k[1])
        num_term2 = k[1]*(s_2*(1 - self.poi)*self.burg - self.burg*((k[1])**2))
        num = num_term1 + num_term2

        w2 = np.dot(k,k)
        factor = 2*(1 - np.exp(-w2/(4*np.pi*self.ni)))
        num = num*np.sqrt(factor)

        den = (s_2**2)*(1 - self.poi)

        return num/den

    def V_s(self, k: np.ndarray):
        return (4*np.pi*self.Z_a)/(np.dot(k, k) + self.ktf**2)

    # return a complex vector
    # first element is the numerator 
    # second element is the denominator
    def A_s_dip(self, k: np.ndarray):
        pre_fac = (self.nd/2)*self.V_s(k)
        kF = self.k_dot_F_dip(k)
        val = 1j*pre_fac*kF

        return val 

    def A_s_avg(self, k: np.ndarray):
        pre_fac = (self.nd/2)*self.V_s(k)
        kF = self.k_dot_F_avg(k)
        val = 1j*pre_fac*kF

        return val

    def Energy(self, k: np.ndarray):
        return np.dot(k, k)/(2)

    def G0_func(self, k: np.ndarray, E: float64, ep: float64):
        ek = self.Energy(k) + (self.p0[2]**2)/2
        den = (E - ek)**2 + ep**2
        num = E - ek - 1j*ep
        return num/den

    def Integrand_dip(self, *x):
        ep0 = 10
        Ac = 1.2
        ep = (ep0/(Ac**self.l))

        k = np.zeros( len(x), dtype=np.float64)
        for i in range(len(x)):
            k[i] = x[i]

        A_total = self.A_s_dip(k - self.p0[0:2])*self.A_s_dip(self.p0[0:2] - k)
        result = A_total*self.G0_func(k, self.E_eval, ep)

        return result.real

    def Integrand_avg(self, *x):
        ep0 = 10
        Ac = 1.2
        ep = (ep0/(Ac**self.l))
        k = np.zeros( len(x), dtype=np.float64)
        for i in range(len(x)):
            k[i] = x[i]

        w = k - self.p0[0:2]
        A_total = self.A_s_avg(w)*self.A_s_avg(-w)
        result = A_total*self.G0_func(k, self.E_eval, ep)

        return result.real




def self_energy_dip(k, omega, dip_sep, l):
    Integrand_object = Integrand_class()
    Integrand_object.set_dip_arr(dip_sep)
    self_energy_val = 0

    Integrand_object.set_p0(k)
    Integrand_object.set_Eval(omega)
    Integrand_object.set_l(l)
    a = -Integrand_object.cutoff
    b = Integrand_object.cutoff

    ni = Integrand_object.ni
    meas = 2
    un_pol_const = 2
    prefac = un_pol_const*(ni)*((1/(2*np.pi))**meas)

    int_range = []
    for i in range(2):
        int_range.append([a, b])

    result, _ = nquad(Integrand_object.Integrand_dip , int_range)
    self_energy_val = prefac*result

    return self_energy_val 



def self_energy_avg(k, omega, l):
    Integrand_object = Integrand_class()
    self_energy_val = 0

    Integrand_object.set_p0(k)
    Integrand_object.set_Eval(omega)
    Integrand_object.set_l(l)
    a = -Integrand_object.cutoff
    b = Integrand_object.cutoff

    ni = Integrand_object.ni
    meas = 2
    un_pol_const = 2
    prefac = un_pol_const*(ni)*((1/(2*np.pi))**meas)

    int_range = []
    for i in range(2):
        int_range.append([a, b])

    result, _ = nquad(Integrand_object.Integrand_avg , int_range)
    self_energy_val = prefac*result

    return self_energy_val 


@jit(nopython=True)
def sort_by_absolute_value(arr):
    # Pair each element with its index
    indexed_arr = list(enumerate(arr))
    # Sort by the absolute value of the elements
    sorted_indexed_arr = sorted(indexed_arr, key=lambda x: abs(x[1]))
    # Extract the sorted indices
    sorted_indices = [index for index, value in sorted_indexed_arr]
    # Extract the sorted values based on the indices
    sorted_values = [arr[index] for index in sorted_indices]
    return sorted_indices, sorted_values


#removes elements from a list
#does not preserve order 
@jit(nopython=True)
def remove_elements(arr, list_remove): 
    list_remove.sort(reverse = True)
    j = len(arr) - 1
    for index in list_remove: 
        tmp = arr[j]
        arr[j] = arr[index] 
        arr[index] = tmp 
        j = j - 1 

    for i in range(len(list_remove)): 
        arr.pop()

    return arr 


#Given a list of numbers, this function returns
#a set of numbers that all have a difference of less than or equal to a 
#tolerance value between atleast one other number in the set 
def Bining_func(arr_in, tol):
    Bins = {}
    bin_num = 1
    arr = copy.deepcopy(arr_in)

    while len(Bins) != len(arr_in):
        visited_hash = {} 
        stack = [] 
        stack.append(arr[0])
        Bins[arr[0]] = bin_num

        while len(stack) != 0: 
            node = stack.pop() 
            visited_hash[node] = 1
            for i in range(len(arr)):
                #this node has not been visited 
                if visited_hash.get(arr[i], -10000) == -10000:
                    if abs(node - arr[i]) < tol: 
                        stack.append(arr[i])
                        Bins[arr[i]] = bin_num
                        visited_hash[arr[i]] = 1 

        remove_indices = [] 
        for k in range(len(arr)):
            #key is in Bins
            if Bins.get(arr[k], -10000) != -10000:
                remove_indices.append(k)  

        arr = remove_elements(arr, remove_indices)

        bin_num = bin_num + 1

    return Bins, bin_num - 1 


def disperse_func_dip(args_tuple):
    omega, k, epsilon_k, dip_sep, l = args_tuple
    return omega - epsilon_k - self_energy_dip(k, omega, dip_sep, l)



def disperse_func_avg(args_tuple):
    omega, k, epsilon_k, l = args_tuple
    return omega - epsilon_k - self_energy_avg(k, omega, l)


#burg_sep is a float that indicates the magnitude of the 
#dipole separation in units of the burgers vector,
#it is only used to name the output file appropiately 

#angle is also used for file naming purposes, and indicates the 
#the angle of k wrt to the kx axes

#part indicates the current div number for file naming purposes
def dispersion_relation_dip(k, dip_sep, l, burg_sep, angle, part):
    Integrand_object = Integrand_class()
    ek = Integrand_object.Energy(k) 
    omega_max = ek + 0.05 
    omega_min = ek - 0.05 
    omega = omega_min

    delta_omega = 7e-6 
    pool_list = [] 

    while omega <= omega_max:
        list_i = ()
        list_i = list_i + (omega,)
        list_i = list_i + (k,)
        list_i = list_i + (ek,)
        list_i = list_i + (dip_sep,)
        list_i = list_i + (l,)
        pool_list.append(list_i)

        omega = omega + delta_omega

    with Pool() as pool:
        all_results = pool.map(disperse_func_dip, pool_list)

    E_vec = [] 
    self_E_arr = [] 

    for task, result in zip(pool_list, all_results): 
        #energy val
        E = task[0]
        E_vec.append(E)
        self_E_arr.append(result)

    sorted_indices, sorted_values = sort_by_absolute_value(self_E_arr)

    E_vec_sorted = [] 

    for i in range(len(E_vec)):
        E_vec_sorted.append(E_vec[sorted_indices[i]]) 

    k_mag = np.sqrt(np.dot(k,k)) 

    outF = open("Roots_Si_" + str(burg_sep) + "_burg_" + str(angle) + "_angle_Part_" + str(part) + "_3D_l_72.txt", "a")

    #first energy is stored in bin 1 along with its corresponding
    #E - ek - SelfEnergy(k,E)
    Energy_to_Index = {} 
    for j in range(300):
        Energy_to_Index[E_vec[sorted_indices[j]]] = j

    Bins, num_bins = Bining_func(E_vec_sorted[0:300], 0.00001)

    #The Energy bins hashes Energy -> [bin #, E - ek - SelfEnergy(k,E)]
    Energy_Bins = {} 

    for key, value in Bins.items(): 
        index = Energy_to_Index[key]
        Energy_Bins[key] = [value, sorted_values[index]] 

    roots = []
    selfE_roots_arr = []
    for n in range(1, num_bins + 1):
        E_root = 0
        selfE_root = sys.maxsize 
        Assigned_root = False
        for key , value in Energy_Bins.items():
            if abs(value[1]) > 3e-4:
                continue
            if value[0] == n:
                if value[1] < selfE_root:
                    selfE_root = value[1]
                    E_root = key
                    Assigned_root = True

        if Assigned_root == True:
            roots.append(E_root) 
            selfE_roots_arr.append(selfE_root)

    outF.write("k: " + str(k_mag) + " E: ")
    for j in range(len(roots)):
        outF.write(str(roots[j]) + "," + str(selfE_roots_arr[j]))
        outF.write(" ")

    outF.write("\n")
    outF.flush()


def dispersion_relation_avg(k, l, angle, part):
    Integrand_object = Integrand_class()
    ek = Integrand_object.Energy(k) 
    omega_max = ek + 0.05 
    omega_min = ek - 0.05 
    omega = omega_min

    delta_omega = 7e-7
    pool_list = [] 

    while omega <= omega_max:
        list_i = ()
        list_i = list_i + (omega,)
        list_i = list_i + (k,)
        list_i = list_i + (ek,)
        list_i = list_i + (l,)
        pool_list.append(list_i)

        omega = omega + delta_omega

    with Pool() as pool:
        all_results = pool.map(disperse_func_avg, pool_list)

    E_vec = [] 
    self_E_arr = [] 

    for task, result in zip(pool_list, all_results): 
        #energy val
        E = task[0]
        E_vec.append(E)
        self_E_arr.append(result)

    sorted_indices, sorted_values = sort_by_absolute_value(self_E_arr)

    E_vec_sorted = [] 

    for i in range(len(E_vec)):
        E_vec_sorted.append(E_vec[sorted_indices[i]]) 

    k_mag = np.sqrt(np.dot(k,k)) 

    outF = open("Roots_Si_" + "_Average_" + str(angle) + "_angle_Part_" + str(part) + "_3D_l_72.txt", "a")

    #first energy is stored in bin 1 along with its corresponding
    #E - ek - SelfEnergy(k,E)
    Energy_to_Index = {} 
    for j in range(600):
        Energy_to_Index[E_vec[sorted_indices[j]]] = j

    Bins, num_bins = Bining_func(E_vec_sorted[0:600], 0.00001)

    #The Energy bins hashes Energy -> [bin #, E - ek - SelfEnergy(k,E)]
    Energy_Bins = {} 

    for key, value in Bins.items(): 
        index = Energy_to_Index[key]
        Energy_Bins[key] = [value, sorted_values[index]] 

    roots = []
    selfE_roots_arr = []
    for n in range(1, num_bins + 1):
        E_root = 0
        selfE_root = sys.maxsize 
        Assigned_root = False
        for key , value in Energy_Bins.items():
            if abs(value[1]) > 3e-4:
                continue
            if value[0] == n:
                if value[1] < selfE_root:
                    selfE_root = value[1]
                    E_root = key
                    Assigned_root = True

        if Assigned_root == True:
            roots.append(E_root) 
            selfE_roots_arr.append(selfE_root)

    outF.write("k: " + str(k_mag) + " E: ")
    for j in range(len(roots)):
        outF.write(str(roots[j]) + "," + str(selfE_roots_arr[j]))
        outF.write(" ")

    outF.write("\n")
    outF.flush()


def main(): 
    size_k = 300#3000
    k_max = size_k*14e-4 # = 2.1
    deltak = 14e-4
    kin = 0.0 
    k_mag_arr = [] 

    while kin <= k_max: 
        k_mag_arr.append(kin)
        kin = kin + deltak 

    #num of division of k_arr
    num_subdiv = int(sys.argv[1])
    #current sub-division number
    div_num = int(sys.argv[2])
    #boolean if
    #dip = 0 --> averaged value is taken
    #dip = 1 --> a fixed dipole separation is taken
    dip = int(sys.argv[3])
    #if dip = 1 this is the dipole separation magnitude
    #in units of the burgers vector 
    burg_sep = float(sys.argv[4])

    #angle to compute 
    angle = int(sys.argv[5])

    k_angle = np.zeros(3 , dtype=np.float64)

    if angle == 45:
        k_angle = np.array([1.0/np.sqrt(3), 1.0/np.sqrt(3), 1.0/np.sqrt(3)])

    elif angle == 90: 
        k_angle = np.array([0.0,1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)])
        
    elif angle == 0:
        k_angle = np.array([1.0/np.sqrt(2.0), 0.0, 1.0/np.sqrt(2.0)])

    k_start = (size_k/num_subdiv)*(div_num - 1) 
    k_end = (size_k/num_subdiv)*div_num

    #set of k values to use for this run 
    k_comp = k_mag_arr[int(k_start):int(k_end)]

    part = div_num
    l = 72

    Integrand_object = Integrand_class()

    if dip == 1:
        dip_sep = burg_sep*Integrand_object.burg*np.array([0.0, 1.0])
        for k_mag in k_comp:
            k = k_mag*k_angle
            dispersion_relation_dip(k, dip_sep, l, burg_sep, angle, part) 

    elif dip == 0: 
        for k_mag in k_comp:
            k = k_mag*k_angle
            dispersion_relation_avg(k, l, angle, part)


if __name__ == "__main__":
    main()
