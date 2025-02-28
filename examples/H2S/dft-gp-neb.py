import os
from mpi4py import MPI
from gpr_calc.GPRANEB import GP_NEB
from gpr_calc.calculator import GPR
from gpr_calc.gaussianprocess import GaussianProcess as GP
from ase.mep import NEB
from ase.optimize import FIRE, BFGS
from ase.calculators.vasp import Vasp
import socket 

# Set MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
tag = 'h2s-RBF'

# Set VASP parameters and Environment
ncpu = 80
vasp_args = {"txt": 'vasp.out',
             "prec": 'Accurate',
             "encut": 400,
             "algo": 'Fast',
             "xc": 'pbe',
             "icharg": 2,
             "ediff": 1.0e-4,
             "ediffg": -0.03,
             "ismear": 1,
             "sigma": 0.1,
             "ibrion": -1,
             "isym": 0,
             "idipol": 3,
             "ldipol": True,
             "lwave": False,
             "lcharg": False,
             "lreal": 'Auto',
             "npar": 2, #4,
             "kpts": [2, 2, 1],
            }
os.system("module load vasp/6.4.3")
os.environ["ASE_VASP_COMMAND"] = (
    "mpirun "
    "--bind-to core "
    "--map-by rankfile:file=../rankfile.txt "  # New syntax for rankfile
    #"--report-bindings "
    f"-np {ncpu} vasp_std")
os.environ["VASP_PP_PATH"] = "/projects/mmi/potcarFiles/VASP6.4"

# Create rankfile for process binding
if rank == 0:
    hostname = socket.gethostname()
    with open('rankfile.txt', 'w') as f:
        for i in range(ncpu):
            cpu_id = i + size  # Start from core 4
            f.write(f'rank {i}={hostname} slot={cpu_id}\n')

# Set NEB Parameters
nimages = 5
fmax = 0.1

# Set the GP model
neb_gp = GP_NEB("POSCAR_initial",
                "POSCAR_final",
                num_images = nimages,
                useCalc = Vasp(**vasp_args),
                pbc = False)

# Create the images
images = [neb_gp.initial_state]
for i in range(neb_gp.num_images):
    image = neb_gp.initial_state.copy()
    images.append(image)
images.append(neb_gp.final_state)
neb = NEB(images, parallel=False)#, k=0.5)
neb.interpolate(method='idpp', mic=True)

# Initialize the calculator
if os.path.exists(tag+'-gpr.json'):
    neb_gp.model = GP.load(tag + '-gpr.json')
    neb_gp.model.fit()
else:
    neb_gp.set_GPR(kernel='RBF', noise_e=0.0015, noise_f=0.1)
    neb_gp.train_GPR(images)

if rank == 0: print(neb_gp.model)

for i, image in enumerate(neb.images):
    base = Vasp(**vasp_args, directory=f"neb_calc_{i}")
    image.calc = GPR(base_calculator = base,
                     ff = neb_gp.model,
                     freq = 20,
                     tag = tag,
                     return_std = True)

qn = FIRE(neb, trajectory='neb.traj')
qn.run(fmax=0.05, steps=1000) #trajectory='neb.traj')
