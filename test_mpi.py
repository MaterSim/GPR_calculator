from gpr_calc.gaussianprocess import GaussianProcess as GPR
from mpi4py import MPI
from time import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

t0 = time()

#try:
gpr = GPR.load('database/hs2-RBF-gpr.json')
#gpr.fit(opt=True)
print(gpr)
print(f'Time: {time() - t0:.2f}s')
#except Exception as e:
#    print(f"Rank {rank} encountered error: {str(e)}")
#    comm.Abort()