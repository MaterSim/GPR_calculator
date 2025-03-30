from ase.calculators.emt import EMT
from gpr_calc.NEB import neb_calc, neb_generate_images, neb_plot_path
from gpr_calc.gaussianprocess import GaussianProcess as GP
from gpr_calc.calculator import GPR
from mpi4py import MPI

# Set MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Set parameters
initial, final = 'database/initial.traj', 'database/final.traj'
num_images = 5
fmax = 0.05

# Run NEB with EMT calculator
images = neb_generate_images(initial, final, num_images)
if rank == 0:
    images, energies, steps = neb_calc(images, EMT(), fmax=fmax, steps=100)
    data = [(images, energies, f'EMT ({(steps+1)*(len(images)-2)+2})')]

# Run NEB with gpr calculator in different etols
for etol in [0.02, 0.1, 0.2]:
    images = neb_generate_images(initial, final, num_images)

    # initialize GPR model
    gp_model = GP.set_GPR(images,
                          base_calculator=EMT(),
                          kernel='RBF',
                          noise_e=etol/len(images[0]),
                          noise_f=0.1)
    # Set GPR calculator
    calc = GPR(base_calculator=EMT(), ff=gp_model)

    # Run NEB calculation
    images, energies, _ = neb_calc(images, calc, fmax=fmax, steps=100)

    if rank == 0:
        print(gp_model)
        N_calls = gp_model.count_use_base
        data.append((images, energies, f'GPR-{etol:.2f} ({N_calls})'))

if rank == 0:
    neb_plot_path(data, figname='NEB-test.png')
