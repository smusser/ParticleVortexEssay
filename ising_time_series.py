import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# set the number of sites per side, the temperature, and the equilibration time
side_length = 2**8
T = 1.4
equilibration_time = 2**10+1


# define a function that implements side_length^2 number of  Monte Carlo
# moves on lattice
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



# randomly initialize the lattice
lattice = 2*np.random.randint(2, size=(side_length, side_length))-1



# advance the lattice using our function
for m in range(2**10+1):

	lattice = lattice_moves(lattice, 1/T)


	# plot the current state at specified intervals
	# this code seems complicated, but it is just to make our plots look nice
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

			plt.savefig('Ising_sim_'+str(T)+'_'+str(m)+'.png', dpi=400)


