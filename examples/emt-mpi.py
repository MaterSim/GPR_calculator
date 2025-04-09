from ase.calculators.emt import EMT
from gpr_calc.gaussianprocess import GP
from gpr_calc.calculator import GPR
from gpr_calc.NEB import neb_calc, get_images, plot_path
from mpi4py import MPI

# Set MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Set parameters
init, final = 'database/initial.traj', 'database/final.traj'
num_images = 5
fmax = 0.05

# Run NEB with gpr calculators
data = []
for (etol, ftol) in zip([0.05, 0.1], [0.05, 0.1]):
    images = get_images(init, final, num_images)

    # initialize GPR model
    gp = GP.set_GPR(images,
                    base=EMT(),
                    noise_e=etol/len(images[0]),
                    noise_f=ftol)
    # Set GPR calculator
    calc = GPR(base=EMT(), ff=gp, save=False)

    # Run NEB calculation
    neb = neb_calc(images, calc, fmax=fmax)
    if rank == 0:
        N1, N2 = gp.use_base, gp.use_surrogate
        data.append((neb.images, neb.energies, f'GPR-{ftol:.2f} ({N1}/{N2})'))
        print(gp, '\n\n')

if rank == 0:
    plot_path(data, figname='NEB-test.png')
