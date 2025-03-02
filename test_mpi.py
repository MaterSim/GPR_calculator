from gpr_calc.gaussianprocess import GaussianProcess as GPR
from mpi4py import MPI
from time import time
from ase.io import read

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

t0 = time()
gpr = GPR.load('database/h2s-RBF-gpr.json', N_max=100)
gpr.fit(opt=False)
#gpr.validate_data(return_std=True)

struc = read('database/POSCAR_initial_h2s', format='vasp')
res = gpr.predict_structure(struc, stress=False, return_std=True)

if rank == 0:
    print(res)
    print(f'Time: {time() - t0:.2f}s')
