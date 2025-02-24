from gpr_calc.gaussianprocess import GaussianProcess as GPR
from mpi4py import MPI
from time import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

t0 = time()
gpr = GPR.load('database/hs2-RBF-gpr.json', N_max=100)
gpr.fit()

if rank == 0:
    print(f'Time: {time() - t0:.2f}s')
