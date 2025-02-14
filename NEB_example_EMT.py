from gpr_calc.GPRANEB import GP_NEB
from gpr_calc.calculator import GPR
from ase.optimize import BFGS
from ase.mep import NEB
from ase.calculators.emt import EMT

initial_state = 'database/initial.traj'
final_state = 'database/final.traj'
num_images = 5
fmax = 0.05

print("\nInit the model")
neb_gp = GP_NEB(initial_state,
                final_state,
                num_images=num_images)

print("\nGet the initial images")
images = neb_gp.generate_images(IDPP = False)

# Set Base calculator and Run NEB
for image in images:
    image.calc = EMT()
neb = NEB(images)
opt = BFGS(neb)
opt.run(fmax=fmax)
neb_gp.plot_neb_path(images, figname='Ref.png')

# Test gpr calculator
for kernel in ['RBF']: #, 'Dot']:
    images = neb_gp.generate_images(IDPP = False)

    print("\nCreate the initial GPR model")
    neb_gp.set_GPR(kernel=kernel, noise_e=fmax/10)
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

    print("\nRun actual NEB")
    neb = NEB(images)
    opt = BFGS(neb) # add callback function to update the F_std threshold
    opt.run(fmax=fmax)

    # Plot results
    neb_gp.plot_neb_path(images, figname=kernel+'.png')

    print(neb_gp.model)
    print("\nTotal number of base calls", neb_gp.model.count_use_base)
    print("Total number of surrogate calls", neb_gp.model.count_use_surrogate)
    print("Total number of gpr_fit calls", neb_gp.model.count_fits)
