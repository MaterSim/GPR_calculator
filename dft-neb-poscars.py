from gpr_calc.GPRANEB import GP_NEB
from gpr_calc.calculator import GPR
from ase.optimize import BFGS
from ase.mep import NEB
from ase.calculators.emt import EMT
import os
from ase.calculators.vasp import Vasp
from ase.io import read

initial = read('POSCAR_initial')
final = read('POSCAR_final')
nimages = 5
fmax = 0.05

images = [initial]

for i in range(1, nimages + 1):
    # read in the POSCAR files
    image = read(f'POSCAR_{i}')
    images.append(image)

images.append(final)

for i, image in enumerate(images):
    image.calc = Vasp(
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

os.system("module load vasp/6.4.3")
os.environ["ASE_VASP_COMMAND"] = "mpirun vasp_std"
os.environ["VASP_PP_PATH"] = "/projects/mmi/potcarFiles/VASP6.4"

neb = NEB(images, k=0.5) #, climb=True)
opt = BFGS(neb)
opt.run(fmax=fmax) #, logfile='ase.log')
