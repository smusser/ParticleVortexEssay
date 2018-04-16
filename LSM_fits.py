import os
import errno
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



### IMPORT ALL THE VALUE FOR T, SpecificHeat ################################
# read in the specific heat values from the saved file
T = []
SpecificHeat = []

target1 = open('LSM_SpecificHeat.txt', 'r')
for i in range(1000):
	st = target1.readline()

	if len(st) > 0:
		number1 = st.find(',')+1

		T.append(float(st[:number1-1]))
		SpecificHeat.append(float(st[number1:]))

# now sort the values appropriately
T, SpecificHeat = (list(t) for t in zip(*sorted(zip(T, SpecificHeat))))
T = 1.0/np.array(T)
SpecificHeat = np.array(SpecificHeat)

### NOW BIN THE DATA TO GET GOOD ERRORBARS ###################################

bins = np.linspace(1.25, 1.8, 20)
T_plots = np.linspace(1.25, 1.8, 100)
digitized = np.digitize(T, bins)

bin_means = [SpecificHeat[digitized == i].mean() for i in range(1, len(bins))]
bin_err = [SpecificHeat[digitized == i].std()/np.sqrt(len(SpecificHeat[digitized == i])) for i in range(1, len(bins))]


# define the function we want to fit
def guess(T, A, Tc, D, B):
	t = (T-Tc)/Tc
	return -A*np.log(abs(t)) - 0.5*D*A*np.sign(t) + B


# fit using scipy's built in function
popt, pcov = curve_fit(guess, bins[1:], bin_means, sigma=bin_err, p0=[100.0, 1.5, 1.0, 2.0])
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

ax.set_title('Specific heat vs. temperature for $5\\times 5\\times 5$ LSM')
ax.set_ylabel('Specific Heat, $C_V$ ($J/k_B^2$)')
ax.set_xlabel('Temperature, $T$ ($J/k_B$)')
ax.set_xlim([1.1, 2.4])
ax.set_ylim([0, 900])

ax.plot(T, SpecificHeat, 'k.', label='Data')
ax.errorbar(bins[1:], bin_means, yerr=bin_err, fmt='o', color='r', label='Data in Fit')
ax.plot(T_plots, guess(T_plots, *popt), 'b-', label='Fit')


textfit = '$C_V(T) = B -A\ln\left|t \\right| -\\frac{1}{2}DA\mathrm{sgn}\left(t\\right)$ \n' \
	'$t = \\frac{T-T_c}{T_c}$ \n' \
	'$---------$ \n' \
	'$A = %.0f \pm %.0f \ J/k_B^2 $ \n' \
	'$B = %.0f \pm %.0f \ J/k_B^2$ \n' \
	'$D = %.1f \pm %.1f$ \n' \
	'$T_c = %.2f \pm %.2f \ J/k_B$' \
	% (np.round(popt[0], -1), np.round(perr[0], -1), np.round(popt[3], -1), np.round(perr[3], -1), popt[2], perr[2], np.round(popt[1],2), np.round(perr[1],2))
ax.text(0.97, .97, textfit, transform=ax.transAxes, fontsize=11,
verticalalignment='top', horizontalalignment='right')

plt.legend(loc='upper left')

plt.savefig('Alpha_fit_LSM.png', dpi=400)
plt.show()
plt.close()
