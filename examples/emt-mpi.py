from ase.calculators.emt import EMT
from gpr_calc.gaussianprocess import GP
from gpr_calc.calculator import GPR
from gpr_calc.NEB import neb_calc, init_images, neb_plot_path
from mpi4py import MPI

# Set MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Set parameters
init, final = 'database/initial.traj', 'database/final.traj'
num_images = 5
fmax = 0.05

# Run NEB with EMT calculator
images = init_images(init, final, num_images)
if rank == 0:
    images, engs, steps = neb_calc(images, EMT(), fmax=fmax)
    data = [(images, engs, f'EMT ({(steps+1)*(len(images)-2)+2})')]

# Run NEB with gpr calculators
for etol in [0.02, 0.1, 0.2]:
    images = init_images(init, final, num_images)

    # initialize GPR model
    gp_model = GP.set_GPR(images,
                          base=EMT(),
                          noise_e=etol/len(images[0]),
                          noise_f=0.1)
    # Set GPR calculator
    calc = GPR(base=EMT(), ff=gp_model)

    # Run NEB calculation
    images, engs, _ = neb_calc(images, calc, fmax=fmax)

    if rank == 0:
        N_calls = gp_model.count_use_base
        data.append((images, engs, f'GPR-{etol:.2f} ({N_calls})'))

if rank == 0:
    neb_plot_path(data, figname='NEB-test.png')
