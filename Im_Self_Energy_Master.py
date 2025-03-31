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

        return result.imag

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

        return result.imag




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



def calculate_Im_SelfE_2D(args_tuple):

    part, dip, burg_sep,  angle =  args_tuple

    k_angle = np.zeros(3 , dtype=np.float64)

    #2D
    if angle == 45:
        k_angle = np.array([1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0])

    elif angle == 90:
        k_angle = np.array([0.0, 1.0, 0.0])

    elif angle == 0: 
        k_angle = np.array([1.0, 0.0, 0.0])

    l = 72

    Integrand_object = Integrand_class()

    outF = 0

    if dip == 1:
        outF = open("Im_SelfE_Si_" + str(burg_sep) + "_burg_" + str(angle) + "_angle_" + "part_" + str(part) + "_2D_l_72.txt", "w")
    
    elif dip == 0:
        outF = open("Im_SelfE_Si_Averaged_part_" + str(part) + "_2D_l_72.txt", "w")


    dip_sep = burg_sep*Integrand_object.burg*np.array([0.0, 1.0])

    
    inF = 0

    if dip == 1:
        inF = open("Roots_Si_" + str(burg_sep) + "_burg_" + str(angle) + "_angle_Part_" + str(part) + "_l_72.txt", "r")
    elif dip == 0: 
        inF = open("Roots_Si_" + "_Average_" + "45" + "_angle_Part_" + str(part) + "_l_72.txt", "r") 

    myline = inF.readline()

    while myline:
        if myline.strip():  
            line = myline.strip()
            parse_str = line.split()
            k_mag = float(parse_str[1])
            for j in range(3, len(parse_str)):
                E_and_Self = parse_str[j].split(",")
                omega = float(E_and_Self[0])
                k = k_mag*k_angle
                ImSelfE = 0 
                if dip == 1: 
                    ImSelfE = self_energy_dip(k, omega, dip_sep, l)
                elif dip == 0: 
                    ImSelfE = self_energy_avg(k, omega, l) 

                outF.write("k: " + str(k_mag) + " ")
                outF.write("E: " + str(omega) + " ")
                outF.write("ImSelfE: " + str(ImSelfE) + "\n")

        myline = inF.readline()




def calculate_Im_SelfE_3D(args_tuple):

    part, dip, burg_sep,  angle =  args_tuple

    k_angle = np.zeros(3 , dtype=np.float64)

    #3D
    if angle == 45:
        k_angle = np.array([1.0/np.sqrt(3), 1.0/np.sqrt(3), 1.0/np.sqrt(3)])

    elif angle == 90: 
        k_angle = np.array([0.0,1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)])
        
    elif angle == 0:
        k_angle = np.array([1.0/np.sqrt(2.0), 0.0, 1.0/np.sqrt(2.0)])

    l = 72

    Integrand_object = Integrand_class()


    outF = 0

    if dip == 1:
        outF = open("Im_SelfE_Si_" + str(burg_sep) + "_burg_" + str(angle) + "_angle_" + "part_" + str(part) + "_3D_l_72.txt", "w")
    
    elif dip == 0:
        outF = open("Im_SelfE_Si_Averaged_" + str(angle) + "_angle_Part_" + str(part) + "_3D_l_72.txt", "w")


    dip_sep = burg_sep*Integrand_object.burg*np.array([0.0, 1.0])

    
    inF = 0

    if dip == 1:
        inF = open("Roots_Si_" + str(burg_sep) + "_burg_" + str(angle) + "_angle_Part_" + str(part) + "_3D_l_72.txt", "r")
    elif dip == 0: 
        inF = open("Roots_Si_" + "_Average_" + str(angle) + "_angle_Part_" + str(part) + "_3D_l_72.txt", "r") 

    myline = inF.readline()

    while myline:
        if myline.strip():  
            line = myline.strip()
            parse_str = line.split()
            k_mag = float(parse_str[1])
            for j in range(3, len(parse_str)):
                E_and_Self = parse_str[j].split(",")
                omega = float(E_and_Self[0])
                k = k_mag*k_angle
                ImSelfE = 0 
                if dip == 1: 
                    ImSelfE = self_energy_dip(k, omega, dip_sep, l)
                elif dip == 0: 
                    ImSelfE = self_energy_avg(k, omega, l) 

                outF.write("k: " + str(k_mag) + " ")
                outF.write("E: " + str(omega) + " ")
                outF.write("ImSelfE: " + str(ImSelfE) + "\n")

        myline = inF.readline()





def main(): 
    #boolean if
    #dip = 0 --> averaged value is taken
    #dip = 1 --> a fixed dipole separation is taken
    dip = int(sys.argv[1])
    #if dip = 1 this is the dipole separation magnitude
    #in units of the burgers vector 
    burg_sep = float(sys.argv[2])

    #angle to compute 
    angle = int(sys.argv[3])

    pool_list = []

    if dip == 1:
        for part in range(1,3):
            list_i = ()
            list_i = list_i + (part,)
            list_i = list_i + (dip,)
            list_i = list_i + (burg_sep,)
            list_i = list_i + (angle,)
            pool_list.append(list_i) 

        with Pool() as pool:
            all_results = pool.map(calculate_Im_SelfE_3D, pool_list) 
    elif dip == 0: 
        for part in range(1,16):
            list_i = ()
            list_i = list_i + (part,)
            list_i = list_i + (dip,)
            list_i = list_i + (burg_sep,)
            list_i = list_i + (angle,)
            pool_list.append(list_i) 

        with Pool() as pool:
            all_results = pool.map(calculate_Im_SelfE_3D, pool_list) 



if __name__ == "__main__":
    main()
