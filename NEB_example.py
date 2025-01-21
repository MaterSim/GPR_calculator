from gpr_calc.GPRANEB import GP_NEB
from gpr_calc.calculator import GPR
from ase.optimize import BFGS
from ase.mep import NEB
from ase.calculators.emt import EMT
import os
from ase.calculators.vasp import Vasp


base = Vasp(label='mylabel', 
            txt='vasp.out',
            #setups={},
            xc = 'PBE',
            prec = "accurate",
            kspacing = 0.2,
            kgamma = True,
            lcharg = False,
            lwave = False,
            ediff = 1e-3,
            npar = 8, 
            )

os.system("module load vasp/6.4.3")
os.environ["ASE_VASP_COMMAND"] = "mpirun -np 32 vasp_std"
os.environ["VASP_PP_PATH"] = "/users/qzhu8/pkgs/VASP6.4/pps"


initial_state = 'database/initial.traj'
final_state = 'database/final.traj'
num_images = 5
fmax = 0.05

print("\nInit the model")
neb_gp = GP_NEB(initial_state, 
                final_state, 
                num_images=num_images,
                useCalc=base,
                pbc=True)

print("\nGet the initial images")
images = neb_gp.generate_images(IDPP = False)
    
# Set Base calculator and Run NEB
#for image in images: image.calc = neb_gp.useCalc
#neb = NEB(images)
#opt = BFGS(neb) 
#opt.run(fmax=fmax)
#neb_gp.plot_neb_path(images, figname='Ref.png')

# Test gpr calculator
for kernel in ['Dot', 'RBF']:
    images = neb_gp.generate_images(IDPP = False)

    print("\nCreate the initial GPR model")
    #neb_gp.set_GPR(kernel=kernel, noise_e=0.002)
    neb_gp.set_GPR(kernel=kernel, noise_e=fmax/10)
    neb_gp.train_GPR(images)
    print(neb_gp.model)

    # Set hybrid calculator
    for image in images:
        image.calc = GPR(base_calculator=neb_gp.useCalc,
                         ff=neb_gp.model,
                         return_std=True)

    print("\nRun actual NEB")
    neb = NEB(images)
    opt = BFGS(neb)
    opt.run(fmax=fmax, logfile='ase.log')

    # Plot results
    neb_gp.plot_neb_path(images, figname=kernel+'.png')

    print(neb_gp.model)
    print("Total number of base calls", neb_gp.model.count_use_base)
    print("Total number of gpr_fit calls", neb_gp.model.count_fits)
