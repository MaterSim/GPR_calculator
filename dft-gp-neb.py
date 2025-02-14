from ase.calculators.vasp import Vasp
from gpr_calc.GPRANEB import GP_NEB
from gpr_calc.calculator import GPR
from ase.mep import NEB
from ase.io import read, write
from ase.optimize import FIRE, BFGS
from ase.constraints import FixAtoms
import os

base = Vasp(txt = 'vasp.out',
            prec = 'Accurate',
            algo = 'Fast',
            xc = 'pbe',
            icharg = 2,
            encut = 400,
            ediff = 1.0e-4,
            ismear = 1,
            sigma = 0.1,
            ediffg = -0.03,
            ibrion = -1,
            isym = 0,
            idipol = 3,
            ldipol = True,
            lwave = False,
            lcharg = False,
            lreal = 'Auto',
            npar = 2,
            kpts = [2, 2, 1],
            )

# Prepare the environment
os.system("module load vasp/6.4.3")
os.environ["ASE_VASP_COMMAND"] = "mpirun -np 32 vasp_std"
os.environ["VASP_PP_PATH"] = "/projects/mmi/potcarFiles/VASP6.4"

# Read the initial and final images
initial = read("database/POSCAR_initial")
final = read("database/POSCAR_final")
nimages = 5
fmax = 0.05

# Set the GP model
neb_gp = GP_NEB("database/POSCAR_initial",
                "database/POSCAR_final",
                num_images = nimages,
                useCalc = base,
                pbc = False)
neb_gp.set_GPR(kernel='RBF', noise_e = fmax/10)

# Create the images
images = [initial]
for i in range(nimages):
    image = initial.copy()
    images.append(image)
images.append(final)
neb = NEB(images)#, k=0.5)
neb.interpolate(method = 'idpp', mic = True)

# Initialize the calculator
neb_gp.train_GPR(images)
print(neb_gp.model)

for i, image in enumerate(images):
    image.calc = GPR(base_calculator = base,
                     ff = neb_gp.model,
                     freq = 20,
                     tag = 'O-diff-dot',
                     return_std = True)
    image.calc.parameters.base_calculator.set(directory = f"neb_calc_{i}")

qn = BFGS(neb, trajectory='neb.traj')
qn.run(fmax=0.05, steps=2) #trajectory='neb.traj')
