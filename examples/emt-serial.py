from ase.calculators.emt import EMT
from gpr_calc.NEB import neb_calc, neb_generate_images, neb_plot_path
from gpr_calc.gaussianprocess import GaussianProcess as GP
from gpr_calc.calculator import GPR

# Set parameters
initial, final = 'database/initial.traj', 'database/final.traj'
num_images = 5
fmax = 0.05

# Run NEB with EMT calculator
images = neb_generate_images(initial, final, num_images)
images, energies = neb_calc(images, EMT(), fmax=fmax, steps=100)
data = [(images, energies, 'EMT')]

# Run NEB with gpr calculator in different etols
for etol in [0.02, 0.1, 0.2]:
    images = neb_generate_images(initial, final, num_images)

    # initialize GPR model
    gp_model = GP.set_GPR(images, EMT(),
                          kernel='RBF',
                          noise_e=etol/len(images[0]),
                          noise_f=0.1)
    # Set GPR calculator
    calc = GPR(base_calculator=EMT(), ff=gp_model)

    # Run NEB calculation
    images, energies = neb_calc(images, calc, fmax=fmax, steps=100)
    print(gp_model)
    data.append((images, energies, f'GPR-{etol:.2f}'))

neb_plot_path(data, figname='NEB-test.png')
