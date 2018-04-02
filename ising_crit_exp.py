#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt


### FUNCTIONS ##########################################################

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



# this function implements the Metropolis moves on our lattice
def lattice_moves(lattice, K, side_length):
	for i in range(side_length):
		for j in range(side_length):

			# pick the random site and get its spin
			a = np.random.randint(side_length)
			b = np.random.randint(side_length)

			s = lattice[a, b]


			# now find the energy change from flipping the spin
			neighbor_sum = lattice[(a+1)%side_length, b] + lattice[(a-1)%side_length, b] + lattice[a, (b+1)%side_length] + lattice[a, (b-1)%side_length]

			Delta_F = 2 * K * s * neighbor_sum


			# implement the Monte Carlo procedure
			u = np.random.random()

			if Delta_F < 0:

				lattice[a,b] = -s

			elif u < np.exp(-Delta_F):

				lattice[a,b] = -s

	return lattice



# this function computes the 'energy' of the lattice (really beta*energy)
# note that K is defined as beta*J 
def lattice_energy(lattice, K, side_length):

	neighbor_grid = np.roll(lattice, 1, axis=0) + np.roll(lattice, side_length-1, axis=0) + np.roll(lattice, 1, axis=1) + np.roll(lattice, side_length-1, axis=1)

	energy_grid = -K * lattice * neighbor_grid/4.0 # four avoids overcounting

	return np.sum(energy_grid)



# this function computes the magnetization of the lattice
def lattice_magnetization(lattice):

	return np.sum(lattice)



#### COMPUTATION ##########################################################

# specify parameters
side_length = 2**3  # size of lattice where we compute critical exponents
number_temperature = 2**7  # number of temperature points to test
equilibration_time = 2**9  # equilibration time (MAKE TEMP DEPENDENT)
calculation_time = 2**10  # number of times we sample representative state


# distribute temperature points around critical point
# points near critical point more important, so normally distributed
T_range = np.random.normal(2.27, 0.5, number_temperature)
T_range = T_range[(T_range>1.07) & (T_range<3.47)]
number_temperature = np.size(T_range)


# initialize arrays for quantities we wish to measure
SpecificHeat = np.zeros(number_temperature)
Magnetization = np.zeros(number_temperature)


count = 0 # define a count so we can show a progress bar
print_progress(count, number_temperature) # progress bar


for T in T_range:


	# initialization step for our lattice at the given temperature
	lattice = 2*np.random.randint(2, size=(side_length, side_length))-1


	# now we will equilibrate the lattice
	# we deal with critical slowing down by increasing equilibration time
	# as we approach the critical point
	real_eq_time = int(equilibration_time**(2.0-abs(T-2.27)))
	for m in range(real_eq_time):

		lattice = lattice_moves(lattice, 1/T, side_length)



	# having equilibrated the lattice, we now move on to finding
	# expectation values
	# initialize the arrays we will be measuring
	F_vals = np.zeros(calculation_time)
	M_vals = np.zeros(calculation_time)

	# now we calculate expectation values
	# advancing lattice corresponds to finding another representative state
	for m in range(calculation_time):

		lattice = lattice_moves(lattice, 1/T, side_length)

		# measure quantities
		F = lattice_energy(lattice, 1/T, side_length)
		M = lattice_magnetization(lattice)

		# add back in quantities
		F_vals[m] = F
		M_vals[m] = M


	# compute specific heat and magnetization from samples
	SpecificHeat[count] = np.mean(F_vals*F_vals) - np.mean(F_vals)*np.mean(F_vals)
	Magnetization[count] = np.mean(M_vals)

	count += 1 # increment count
	print_progress(count, number_temperature) # update progress bar



# write the specific heat and magnetization to files for later fitting
target1 = open('ising_SpecificHeat.txt', 'a')
target1.truncate()
for m in range(number_temperature):
	target1.write(str(T_range[m])+','+str(SpecificHeat[m]))
	target1.write('\n')
target1.close()

target2 = open('ising_Magnetization.txt', 'a')
target2.truncate()
for m in range(number_temperature):
	target2.write(str(T_range[m])+','+str(Magnetization[m]))
	target2.write('\n')
target2.close()

