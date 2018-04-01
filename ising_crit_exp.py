import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


'''
# specify parameters
side_length = 2**3
equilibration_time = 2**13
calculation_time = 2**10
T_range = np.linspace(1.4, 3.14, 100)
'''

def lattice_moves(lattice, K):
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




def lattice_energy(lattice, K):

	neighbor_grid = np.roll(lattice, 1, axis=0) + np.roll(lattice, side_length-1, axis=0) + np.roll(lattice, 1, axis=1) + np.roll(lattice, side_length-1, axis=1)

	energy_grid = -K * lattice * neighbor_grid/4.0 # four avoids overcounting

	return np.sum(energy_grid)




side_length = 2**8
T = 1.4

lattice = 2*np.random.randint(2, size=(side_length, side_length))-1

for m in range(2**10+1):

	lattice = lattice_moves(lattice, 1/T)

	if m % 2**8 == 0:

		fig, ax = plt.subplots(1,1)
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')

		ax.set_title('Ising Model, $T=$ '+str(T)+', step number $=$ '+str(m))

		im = ax.imshow(lattice, vmin=-1.0, vmax=1.0, interpolation='none', cmap=plt.cm.seismic, origin='lower')
		divider = make_axes_locatable(ax)
		cax = divider.append_axes('right', size='2%', pad=0.05)
		cbar = plt.colorbar(im, cax=cax, ticks=[-1, 1])
		cbar.set_label('Spin at site', labelpad=-10)

		plt.tight_layout()
		#plt.subplots_adjust(bottom=0.0, top=1.0, wspace=0.75)

		plt.savefig('Ising_sim_'+str(T)+'_'+str(m)+'.png', dpi=400)

		if m == 0:

			fig, ax = plt.subplots(1,1)
			plt.rc('text', usetex=True)
			plt.rc('font', family='serif')

			ax.set_title('Ising Model, $T=$ '+str(T)+', step number $=$ '+str(m))

			im = ax.imshow(lattice, vmin=-1.0, vmax=1.0, interpolation='none', cmap=plt.cm.seismic, origin='lower')
			divider = make_axes_locatable(ax)
			cax = divider.append_axes('right', size='2%', pad=0.05)
			cbar = plt.colorbar(im, cax=cax, ticks=[-1, 1])
			cbar.set_label('Spin at site', labelpad=-10)

			plt.tight_layout()
			#plt.subplots_adjust(bottom=0.0, top=1.0, wspace=0.75)

			plt.savefig('Ising_sim_'+str(T)+'_'+str(m)+'.png', dpi=400)



'''
SpecificHeat = []
SpecificHeatErr = []

count = 0
for T in T_range:

	K_temp = 1/T

	# initialization step
	lattice = 2*np.random.randint(2, size=(side_length, side_length))-1

	F_pts = []
	F_pts_sq = []

	for m in range(equilibration_time):

		lattice = lattice_moves(lattice, K_temp)

	for m in range(calculation_time):

		lattice = lattice_moves(lattice, K_temp)

		F = lattice_energy(lattice, K_temp)
		F_pts.append(F)
		F_pts_sq.append(F**2)

	F_pts = np.array(F_pts)
	F_pts_sq = np.array(F_pts_sq)

	SpecificHeat.append(np.mean(F_pts_sq) - np.mean(F_pts)*np.mean(F_pts))

	sigma_sq = np.std(F_pts_sq)/np.sqrt(len(F_pts_sq))
	sigma = np.std(F_pts)/np.sqrt(len(F_pts))

	SpecificHeatErr.append(sigma_sq + 2*np.mean(F_pts)*sigma)

	count += 1
	print count


fig, ax = plt.subplots(1,1)
ax.errorbar(T_range, np.array(SpecificHeat), yerr=np.array(SpecificHeatErr), fmt='o')
plt.show()
'''
















