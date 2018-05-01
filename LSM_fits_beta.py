import os
import errno
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



### IMPORT ALL THE VALUE FOR T, Quantities ################################
# read in the specific heat values from the saved file
T1 = []
SpecificHeat = []

target1 = open('LSM_SpecificHeat.txt', 'r')
for i in range(10000):
	st = target1.readline()

	if len(st) > 0:
		number1 = st.find(',')+1

		T1.append(float(st[:number1-1]))
		SpecificHeat.append(float(st[number1:]))

# now sort the values appropriately
T1, SpecificHeat = (list(t) for t in zip(*sorted(zip(T1, SpecificHeat))))
T1 = 1.0/np.array(T1)
SpecificHeat = np.array(SpecificHeat)



T4 = []
Betap = []

target4 = open('LSM_Betap.txt', 'r')
for i in range(10000):
	st = target4.readline()

	if len(st) > 0:
		number4 = st.find(',')+1

		T4.append(float(st[:number4-1]))
		Betap.append(float(st[number4:]))

# now sort the values appropriately
T4, Betap = (list(t) for t in zip(*sorted(zip(T4, Betap))))
T4 = 1.0/np.array(T4)
Betap = np.array(Betap)



T5 = []
Energy = []

target5 = open('LSM_Energy.txt', 'r')
for i in range(10000):
	st = target5.readline()

	if len(st) > 0:
		number5 = st.find(',')+1

		T5.append(float(st[:number5-1]))
		Energy.append(float(st[number5:]))

# now sort the values appropriately
T5, Energy = (list(t) for t in zip(*sorted(zip(T5, Energy))))
T5 = 1.0/np.array(T5)
Energy = np.array(Energy)



T8 = []
Order = []

target8 = open('LSM_Order.txt', 'r')
for i in range(10000):
	st = target8.readline()

	if len(st) > 0:
		number8 = st.find(',')+1

		T8.append(float(st[:number8-1]))
		Order.append(float(st[number8:]))

# now sort the values appropriately
T8, Order = (list(t) for t in zip(*sorted(zip(T8, Order))))
T8 = 1.0/np.array(T8)
Order = np.array(Order)


### NOW BIN THE DATA TO GET GOOD ERRORBARS ###################################

### BETA FIT #################################################################
bins = np.linspace(1.56, 2.25, 16)
T_plots = np.linspace(1.6, 2.3, 100)
digitized = np.digitize(T8, bins)

bin_means = [Order[digitized == i].mean() for i in range(1, len(bins))]
bin_err = [Order[digitized == i].std()/np.sqrt(len(Order[digitized == i])) for i in range(1, len(bins))]


# define the function we want to fit
def guess(T, A, Tc, beta):
	t = (T-Tc)/Tc
	return A*(abs(t)**(beta))


# fit using scipy's built in function
popt, pcov = curve_fit(guess, bins[1:], bin_means, sigma=bin_err, p0=[0.5, 1.6, 0.3])
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

ax.set_title('Magnitude of order parameter vs. temperature for $4\\times 4\\times 4$ LSM')
ax.set_ylabel('Magnitude of order parameter, $\left|\left\langle e^{i\sigma} \\right\\rangle\\right|$')
ax.set_xlabel('Temperature, $T = \\beta^{\prime}$ ($J/k_B$)')
ax.set_xlim([0.5, 2.5])
ax.set_ylim([0, 0.5])

ax.plot(T8, Order, 'k.', label='Data')
ax.errorbar(bins[1:], bin_means, yerr=bin_err, fmt='o', color='r', label='Data in Fit')
ax.plot(T_plots, guess(T_plots, *popt), 'b-', label='Fit')


textfit = '$|\langle e^{i\sigma}\\rangle|(T) = A\left|\\frac{T-T_c}{T_c}\\right|^{\\beta}$ \n' \
	'$A = %.2f \pm %.2f $ \n' \
	'$T_c = %.2f \pm %.2f \ J/k_B$ \n' \
	'$\\beta = %.2f \pm %.2f \ $' \
	% (popt[0], perr[0], popt[1], perr[1], popt[2], perr[2])
ax.text(0.03, 0.75, textfit, transform=ax.transAxes, fontsize=11,
verticalalignment='top', horizontalalignment='left')

plt.legend(loc='upper left')

plt.savefig('Beta_fit_LSM.png', dpi=400)
plt.close()


### ENERGY PLOTS ####################################################

# this is the real plot
fig, ax = plt.subplots(1,1)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

ax.set_title('Action vs. temperature for $4\\times 4\\times 4$ LSM')
ax.set_ylabel('Action, $\langle S_E\\rangle$')
ax.set_xlabel('Temperature, $T = \\beta^{\prime}$ ($J/k_B$)')
ax.set_xlim([0.0, 2.5])
ax.set_ylim([0, 120])

ax.plot(T5[::3], Energy[::3], 'k.', label='$\langle S_E\\rangle$')
ax.plot(T4[::3], Energy[::3]-Betap[::3], 'b.', label='$\left\langle \sum_{j=1}^{N^3} \\frac{e^2}{8\pi^2}|\\vec{\\nabla}\sigma_j - 2\pi \\vec{n}_j|^2\\right\\rangle$')
ax.plot(T4[::3], Betap[::3], 'r.', label='$\left\langle \sum_{j=1}^{N^3} \\frac{\\beta^{\prime}}{2}|\\vec{\\nabla}\\times \\vec{n}_j|^2\\right\\rangle$')


plt.legend(loc='upper right')

plt.savefig('Energy.png', dpi=400)
plt.close()
