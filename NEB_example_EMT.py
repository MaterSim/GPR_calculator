from gpr_calc.GPRANEB import GP_NEB, plot_neb_path
from gpr_calc.calculator import GPR
from ase.optimize import BFGS
from ase.mep import NEB
from ase.calculators.emt import EMT
from time import time
from mpi4py import MPI

# Set MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Set parameters
t0 = time()
initial = 'database/initial.traj'
final = 'database/final.traj'
num_images = 5
fmax = 0.05

# Set NEB_GPR
neb_gp = GP_NEB(initial, final, num_images=num_images)
images0 = neb_gp.generate_images(IDPP = False)

# Set NEB_base
if rank == 0:
    for image in images0: image.calc = EMT()
    neb = NEB(images0, parallel=False)
    opt = BFGS(neb)
    opt.run(fmax=fmax)
    eng = [image.get_potential_energy() for image in images0]
    data1 = (images0, eng, 'EMT')

# Test gpr calculator
for kernel in ['RBF']: #, 'Dot']:
    images = neb_gp.generate_images(IDPP = False)

    print("\nCreate the initial GPR model")
    neb_gp.set_GPR(kernel=kernel, 
                   noise_e=0.03/len(image), 
                   noise_f=0.1)
    neb_gp.train_GPR(images)
    print(neb_gp.model)

    # Set hybrid calculator
    for image in images:
        image.calc = GPR(base_calculator=neb_gp.useCalc,
                         ff=neb_gp.model,
                         freq=10, #update frequency
                         tag=f'test-{kernel}',
                         return_std=True)
        image.calc.verbose = True

    neb = NEB(images, parallel=False)
    opt = BFGS(neb) # add callback function to update the F_std threshold
    opt.run(fmax=fmax, steps=50)
    eng = [image.get_potential_energy() for image in images]
    data2 = (images, eng, "GPR-"+kernel)
    plot_neb_path(data1, data2, figname=kernel+'.png')

    if rank == 0:
        print("\nTotal number of base calls", neb_gp.model.count_use_base)
        print("Total number of surrogate calls", neb_gp.model.count_use_surrogate)
        print("Total number of gpr_fit calls", neb_gp.model.count_fits)
        print(neb_gp.model)
        print(data1[1], data2[1])

print(f"Total time in rank-{rank}: {time()-t0}")
