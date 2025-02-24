from ase.calculators.vasp import Vasp
from ase.mep import NEB
from ase.io import read, write
from ase.optimize import BFGS
from ase.constraints import FixAtoms
import os

initial = read("POSCAR_initial")
final = read("POSCAR_final")
nimages = 5
#images = make_neb([initial, final], nimages=5)

images = [initial]

for i in range(nimages):
    image = initial.copy()
    images.append(image)

images.append(final)

#neb = NEB(images)
#neb.interpolate()

for i, image in enumerate(images):
    calc = Vasp(
            txt='vasp.out',
            xc='pbe',
            icharg=2,
            encut=400,
            ediff=1.0e-4,
            ismear=1,
            sigma=0.1,
            ediffg=-0.03,
            ibrion=-1,
            #potim=0.1,
            algo='Fast',
            nsw=0,
            isif=2,
            isym=0,
            idipol=3,
            ldipol=True,
            lwave=False,
            lcharg=False,
            lreal='Auto',
            prec='Accurate',
            #nelmin=4,
            npar=2,
            kpts=[2, 2, 1],
            directory=f"neb_calc_{i}"
            )

    image.calc = calc #set_calculator(calculator)


os.system("module load vasp/6.4.3")
os.environ["ASE_VASP_COMMAND"] = "mpirun -np 32 vasp_std"
os.environ["VASP_PP_PATH"] = "/projects/mmi/potcarFiles/VASP6.4"

neb = NEB(images)#, k=0.5)
neb.interpolate(method='idpp', mic=True)
qn = BFGS(neb, trajectory='neb.traj')
qn.run(fmax=0.05) #trajectory='neb.traj')

#energies = [image.get_potential_energy() for image in images]
#print("Energies along the path:", energies)

# print the images in a file
