"""
Recompute the energy and get neb plot
mpirun -np 8 python plot_neb.py
"""
from gpr_calc.NEB import plot_progress
from gpr_calc.gaussianprocess import GP
from gpr_calc.calculator import GPR

# Set the GP calculator
gp = GP.set_GPR(None, None, json_file='pd4-RBF-gpr.json')
calc = GPR(base=None, ff=gp)
calc.update = False
calc.allow_base = False

plot_progress('neb.traj', calc, N_images=7, start=20, interval=30)
