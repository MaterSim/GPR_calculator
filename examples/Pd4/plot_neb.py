"""
Recompute the energy and get neb plot
mpirun -np 8 python plot_neb.py
"""
from ase.io import read
from gpr_calc.NEB import neb_plot_path
from gpr_calc.gaussianprocess import GP
from gpr_calc.calculator import GPR
from mpi4py import MPI

# Set MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

N_images = 7
traj = read('neb.traj', index=':')
N_max = int(len(traj)/N_images)

# Set the GP model
gp_model = GP.set_GPR(None, None, json_file='Pd4-RBF-gpr.json')

# Set the GP calculator
calc = GPR(base=None, ff=gp_model)
calc.update = False

data = []

for step in range(0, N_max+1, 20):
    images = traj[step*N_images:(step+1)*N_images]
    engs = []
    for i, image in enumerate(images):
        image.calc = calc
        eng = image.get_potential_energy()
        engs.append(eng)
    data.append((images, engs, f'NEB_iter_{step}'))

# Plot the NEB path
if rank == 0:
    neb_plot_path(data, figname='neb-process.png')
