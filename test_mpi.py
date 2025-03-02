from gpr_calc.gaussianprocess import GaussianProcess as GPR
from mpi4py import MPI
from time import time
from ase.io import read

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

t0 = time()
#gpr = GPR.load('database/h2s-RBF-gpr.json', N_max=10)
gpr = GPR.load('database/pd4-RBF.json', N_max=10)
gpr.fit(opt=False)
gpr.set_K_inv()
#gpr.validate_data(return_std=True)
if rank == 0: print(f'Time: {time() - t0:.2f}s')

struc = read('database/POSCAR_initial_pd4', format='vasp')
for i in range(3):
    E, F, S, E_std, F_std = gpr.predict_structure(struc, stress=False, return_std=True)

if rank == 0:
    print(f'E: {E:.6f} eV')
    print(f'E_std: {E_std:.6f} eV')
    print(f'F: {F[-3:]}')
    print(f'F_std: {F_std[-3:]}')
    print(f'Time: {time() - t0:.2f}s')
