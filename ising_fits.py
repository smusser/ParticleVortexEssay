#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



##### READ IN DATA (SPECIFIC HEAT) ##############################################

# read in the specific heat values from the saved file
T = []
SpecificHeat = []

target1 = open('ising_SpecificHeat.txt', 'r')
for i in range(200):
	st = target1.readline()

	if len(st) > 0:
		number1 = st.find(',')+1

		T.append(float(st[:number1-1]))
		SpecificHeat.append(float(st[number1:]))

# now sort the values appropriately
T, SpecificHeat = (list(t) for t in zip(*sorted(zip(T, SpecificHeat))))

# now redefine the lists as arrays
T = np.array(T)
SpecificHeat = np.array(SpecificHeat)



##### READ IN DATA (magnetization) ##############################################

# read in the magnetization values from the saved file
T1 = []
Magnetization = []

target1 = open('ising_Magnetization.txt', 'r')
for i in range(200):
	st = target1.readline()

	if len(st) > 0:
		number1 = st.find(',')+1

		T1.append(float(st[:number1-1]))
		Magnetization.append(float(st[number1:]))

# now sort the values appropriately
T1, Magnetization = (list(t) for t in zip(*sorted(zip(T1, Magnetization))))

# now redefine the lists as arrays (normalize magnetization correctly)
T1 = np.array(T1)
Magnetization = np.array(Magnetization)/64.0



##### FITTING (SPECIFIC HEAT) #################################################


# neglect the outlying values from our fit
T_mid = T[20:105]
SpecificHeat_mid = SpecificHeat[20:105]
T_plots = np.linspace(np.min(T_mid), np.max(T_mid), 100)


# define the function we want to fit
def guess(T, A, Tc, D, B):
	t = (T-Tc)/Tc
	return -A*np.log(abs(t)) - 0.5*D*A*np.sign(t) + B


# fit using scipy's built in function
popt, pcov = curve_fit(guess, T_mid, SpecificHeat_mid, p0=[5.0, 2.27, 1.0, 2.0])
perr = np.sqrt(np.diag(pcov))


# plot the fit and the data (with a primer plot to remove weirdness)
fig0, ax0 = plt.subplots(1,1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax0.plot(np.zeros(10), np.zeros(10))
plt.close()
plt.clf()


# this is the real plot
fig, ax = plt.subplots(1,1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

ax.set_title('Specific heat vs. temperature for $8\\times 8$ Ising Model')
ax.set_ylabel('Specific Heat, $C_V$ ($J/k_B^2$)')
ax.set_xlabel('Temperature, $T$ ($J/k_B$)')

ax.plot(T, SpecificHeat, 'k.', label='Data')
ax.plot(T_mid, SpecificHeat_mid, 'b+', label='Data in Fit')
ax.plot(T_plots, guess(T_plots, *popt), 'b-', label='Fit')


textfit = '$C_V(T) = B -A\ln\left|t \\right| -\\frac{1}{2}DA\mathrm{sgn}\left(t\\right)$ \n' \
	'$t = \\frac{T-T_c}{T_c}$ \n' \
	'$---------$ \n' \
	'$A = %.1f \pm %.1f \ J/k_B^2 $ \n' \
	'$B = %.1f \pm %.1f \ J/k_B^2$ \n' \
	'$D = %.1f \pm %.1f$ \n' \
	'$T_c = %.3f \pm %.3f \ J/k_B$' \
	% (popt[0], perr[0], popt[3], perr[3], popt[2], perr[2], popt[1], perr[1])
ax.text(0.03, .92, textfit, transform=ax.transAxes, fontsize=11,
verticalalignment='top')

plt.legend(loc='upper right')

plt.savefig('Alpha_fit.png', dpi=400)
plt.close()



##### FITTING (MAGNETIZATION) #################################################

# neglect the outlying values from our fit
# this will involve culling values that have overly low magnetization
T_fit = []
Magnetization_fit = []

for i in range(len(T)):

	if (abs(Magnetization[i]) > 0.3) and (1.5 < T[i] < 2.30):

		T_fit.append(T[i])
		Magnetization_fit.append(Magnetization[i])

T_fit = np.array(T_fit)
Magnetization_fit = np.array(Magnetization_fit)
T_mag_plots = np.linspace(np.min(T_fit), np.max(T_fit), 100)


# define the function we want to fit
def guess_mag(T, A, Tc, beta):
	t = (T-Tc)/Tc
	return A+beta*np.log(abs(t))


# fit using scipy's built in function
popt_mag, pcov_mag = curve_fit(guess_mag, T_fit, np.log(abs(Magnetization_fit)), p0=[1.0, 2.27, 0.125])
perr_mag = np.sqrt(np.diag(pcov_mag))


# now plot the data and the fit
fig1, ax1 = plt.subplots(1,1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

ax1.set_title('Magnetization vs. temperature for $8\\times 8$ Ising Model')
ax1.set_ylabel('Magnetization, $|\langle M \\rangle |$')
ax1.set_xlabel('Temperature, $T$ ($J/k_B$)')

ax1.plot(T, abs(Magnetization), 'k.', label='Data')
ax1.plot(T_fit, abs(Magnetization_fit), 'r+', label='Data in Fit')
ax1.plot(T_mag_plots, np.exp(guess_mag(T_mag_plots, *popt_mag)), 'r-', label='Fit')

textfit = '$|\langle M \\rangle |(T) = A\left|\\frac{T-T_c}{T_c} \\right|^{\\beta}$ \n' \
	'$A = %.2f \pm %.2f $ \n' \
	'$T_c = %.3f \pm %.3f \ J/k_B$ \n' \
	'$\\beta = %.2f \pm %.2f$ ' \
	% (np.exp(popt_mag[0]), perr_mag[0]*np.exp(popt_mag[0]), popt_mag[1], perr_mag[1], popt_mag[2], perr_mag[2])
ax1.text(0.03, 0.70, textfit, transform=ax1.transAxes, fontsize=11,
verticalalignment='top')


plt.legend(loc='upper right')

plt.savefig('Beta_fit.png', dpi=400)

