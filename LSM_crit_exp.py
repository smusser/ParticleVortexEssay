#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import errno
import sys
import time

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=50):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()



#### WE WILL FIRST MAKE SURE TO GET THE RIGHT DEFINITIONS ####

nt = 2**10       # number of temperature points
beta_array  = np.random.normal(1.6, 0.4, nt) # cluster points possible Tc
beta_array = beta_array[(beta_array > 0.01) & (beta_array < 2.5)]
nt = np.size(beta_array)

system_size = 4            # this is the number of points on the cubic lattice
eqSteps = 5000      # number of MC sweeps for equilibration
mcSteps = 5000       # number of MC sweeps for calculation
theta_n = 100         # discretization of theta
lattice_n = 3		# only -1,0,+1 because other values energetically disfavored

possible_theta_vals = 2.0*np.pi*np.array(range(theta_n))/(theta_n - 1.0) - np.pi
possible_n_vals = np.array(range(lattice_n)) - lattice_n/2



# define the lattice derivative
def lattice_deriv(system_size, x, n):
	return np.roll(x, system_size-1, axis=(n-1)) - x



# define the curl of the integer vector
def lattice_curl_mag(system_size, n1, n2, n3):
	x_comp = lattice_deriv(system_size, n3, 2) - lattice_deriv(system_size, n2, 3)
	y_comp = lattice_deriv(system_size, n1, 3) - lattice_deriv(system_size, n3, 1)
	z_comp = lattice_deriv(system_size, n2, 1) - lattice_deriv(system_size, n1, 2)

	return x_comp**2 + y_comp**2 + z_comp**2



# this is the exponent, computed quickly because of using numpy arrays
def G(system_size, theta, n1, n2, n3, e2, beta_prime):

	term1 = 0.5*beta_prime*lattice_curl_mag(system_size, n1, n2, n3)

	term2 = e2/(8*np.pi**2)*(lattice_deriv(system_size, theta, 1) - 2*np.pi*n1)**2
	term3 = e2/(8*np.pi**2)*(lattice_deriv(system_size, theta, 2) - 2*np.pi*n2)**2
	term4 = e2/(8*np.pi**2)*(lattice_deriv(system_size, theta, 3) - 2*np.pi*n3)**2

	return np.sum(term1 + term2 + term3 + term4)



# these are the terms in G that depend on the position <i,j,k>
# this is useful for fast computation of Delta_G
def local_G(system_size, pos, theta, n1, n2, n3, e2, beta_prime):
	i = pos[0]
	j = pos[1]
	k = pos[2]

	theta_terms_part1 = (theta[i, j, k] - theta[(i-1)%system_size, j, k] - 2*np.pi*n1[(i-1)%system_size, j, k])**2 + (theta[i, j, k] - theta[i, (j-1)%system_size, k] - 2*np.pi*n2[i, (j-1)%system_size, k])**2 + (theta[i, j, k] - theta[i, j, (k-1)%system_size] - 2*np.pi*n3[i, j, (k-1)%system_size])**2

	theta_terms_part2 = (theta[(i+1)%system_size, j, k] - theta[i, j, k] - 2*np.pi*n1[i, j, k])**2 + (theta[i, (j+1)%system_size, k] - theta[i, j, k] - 2*np.pi*n2[i, j, k])**2 + (theta[i, j, (k+1)%system_size] - theta[i, j, k] - 2*np.pi*n3[i, j, k])**2


	curl_terms_part1 = ((n3[i, (j+1)%system_size, k] - n3[i, j, k]) - (n2[i, j, (k+1)%system_size] - n2[i, j, k]))**2 + ((n1[i, j, (k+1)%system_size] - n1[i, j, k]) - (n3[(i+1)%system_size, j, k] - n3[i, j, k]))**2 + ((n2[(i+1)%system_size, j, k] - n2[i, j, k]) - (n1[i, (j+1)%system_size, k] - n1[i, j, k]))**2 

	curl_terms_part2_1 = ((n1[(i-1)%system_size, j, (k+1)%system_size] - n1[(i-1)%system_size, j, k]) - (n3[i, j, k] - n3[(i-1)%system_size, j, k]))**2 + ((n2[i, j, k] - n2[(i-1)%system_size, j, k]) - (n1[(i-1)%system_size, (j+1)%system_size, k] - n1[(i-1)%system_size, j, k]))**2 

	curl_terms_part2_2 = ((n3[i, j, k] - n3[i, (j-1)%system_size, k]) - (n2[i, (j-1)%system_size, (k+1)%system_size] - n2[i, (j-1)%system_size, k]))**2 + ((n2[(i+1)%system_size, (j-1)%system_size, k] - n2[i, (j-1)%system_size, k]) - (n1[i, j, k] - n1[i, (j-1)%system_size, k]))**2

	curl_terms_part2_3 = ((n3[i, (j+1)%system_size, (k-1)%system_size] - n3[i, j, (k-1)%system_size]) - (n2[i, j, k] - n2[i, j, (k-1)%system_size]))**2 + ((n1[i, j, k] - n1[i, j, (k-1)%system_size]) - (n3[(i+1)%system_size, j, (k-1)%system_size] - n3[i, j, (k-1)%system_size]))**2


	return 0.5*beta_prime*(curl_terms_part1 + curl_terms_part2_1+ curl_terms_part2_2 + curl_terms_part2_3) + e2/(8*np.pi**2)*(theta_terms_part1 + theta_terms_part2)



# This is the Metropolis move
# We pick possible values of n and theta uniformly 
def mcmove(system_size, theta, n1, n2, n3, e2, beta_prime):
	for i in range(system_size):
		for j in range(system_size):
			for k in range(system_size):

				a = np.random.randint(0, system_size)
				b = np.random.randint(0, system_size)
				c = np.random.randint(0, system_size)
				pos = (a,b,c)
				M = np.zeros((system_size, system_size, system_size))
				M[a, b, c] = 1.0


				uniform_pick_theta = np.random.randint(0, theta_n)
				uniform_pick_n1 = np.random.randint(0, lattice_n)
				uniform_pick_n2 = np.random.randint(0, lattice_n)
				uniform_pick_n3 = np.random.randint(0, lattice_n)

				theta_final_site = possible_theta_vals[uniform_pick_theta]
				n1_final_site = possible_n_vals[uniform_pick_n1]
				n2_final_site = possible_n_vals[uniform_pick_n2]
				n3_final_site = possible_n_vals[uniform_pick_n3]

				theta_final = theta*(1.0 - M) + theta_final_site*M
				n1_final = n1*(1.0 - M) + n1_final_site*M
				n2_final = n2*(1.0 - M) + n2_final_site*M
				n3_final = n3*(1.0 - M) + n3_final_site*M


				site_intital_energy = local_G(system_size, pos, theta, n1, n2, n3, e2, beta_prime)
				site_final_energy = local_G(system_size, pos, theta_final, n1_final, n2_final, n3_final, e2, beta_prime)
				Delta_G = site_final_energy - site_intital_energy

				if Delta_G < 0:
					theta = theta_final
					n1 = n1_final
					n2 = n2_final
					n3 = n3_final
				elif rand() < np.exp(-Delta_G):
					theta = theta_final
					n1 = n1_final
					n2 = n2_final
					n3 = n3_final

	return theta, n1, n2, n3



### HERE IS WHERE WE COMPUTE THE SPECIFIC HEAT AT EACH GIVEN TEMPERATURE VALUE ####

SpecificHeat = np.zeros(nt)
Energy = np.zeros(nt)
Betap = np.zeros(nt)
Order = np.zeros(nt)


print_progress(0, len(beta_array)-1)
for m in range(len(beta_array)):
	print_progress(m, len(beta_array)-1)

	beta_prime = beta_array[m]

	GT = np.zeros(mcSteps)
	BetapT = np.zeros(mcSteps)
	OrderT = np.zeros(mcSteps, dtype=np.complex128)


	# INITIALIZE SOME RANDOM VALUES FOR OUR PARAMETERS
	theta_vals = 2.0*np.pi*np.random.randint(theta_n, size=(system_size, system_size, system_size))/(theta_n-1.0) - np.pi
	nvals_1 = np.random.randint(lattice_n, size=(system_size, system_size, system_size)) - lattice_n/2
	nvals_2 = np.random.randint(lattice_n, size=(system_size, system_size, system_size)) - lattice_n/2
	nvals_3 = np.random.randint(lattice_n, size=(system_size, system_size, system_size)) - lattice_n/2
    

	#real_eq_time = int(eqSteps**(1.33-abs(T-0.66)))
	for i in range(eqSteps):         # equilibrate
		theta_vals, nvals_1, nvals_2, nvals_3 = mcmove(system_size, theta_vals, nvals_1, nvals_2, nvals_3, 5.0, beta_prime)  # Monte Carlo moves

	for i in range(mcSteps):
		theta_vals, nvals_1, nvals_2, nvals_3 = mcmove(system_size, theta_vals, nvals_1, nvals_2, nvals_3, 5.0, beta_prime)           

		GT[i] = G(system_size, theta_vals, nvals_1, nvals_2, nvals_3, 5.0, beta_prime)
		BetapT[i] = 0.5*beta_prime*np.sum(lattice_curl_mag(system_size, nvals_1, nvals_2, nvals_3))
		OrderT[i] = np.mean(np.exp(1j*theta_vals), dtype=np.complex128)


	SpecificHeat[m] = np.mean(GT*GT) - np.mean(GT)*np.mean(GT)
	Energy[m] = np.mean(GT)
	Betap[m] = np.mean(BetapT)
	Order[m] = np.sqrt(np.real( np.mean(OrderT, dtype=np.complex128)*np.conjugate(np.mean(OrderT, dtype=np.complex128)) ))


	target1 = open('LSM_SpecificHeat.txt', 'a')
	target1.write(str(1/beta_prime)+','+str(SpecificHeat[m]))
	target1.write('\n')
	target1.close()

	target4 = open('LSM_Energy.txt', 'a')
	target4.write(str(1/beta_prime)+','+str(Energy[m]))
	target4.write('\n')
	target4.close()

	target5 = open('LSM_Betap.txt', 'a')
	target5.write(str(1/beta_prime)+','+str(Betap[m]))
	target5.write('\n')
	target5.close()

	target8 = open('LSM_Order.txt', 'a')
	target8.write(str(1/beta_prime)+','+str(Order[m]))
	target8.write('\n')
	target8.close()
