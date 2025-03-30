from ase.calculators.vasp import Vasp
from gpr_calc.NEB import neb_calc, neb_generate_images, neb_plot_path
from gpr_calc.calculator import GPR
from gpr_calc.gaussianprocess import GaussianProcess as GP
from mpi4py import MPI
import os
import socket
import psutil

# Set MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
cpu_count = psutil.cpu_count(logical=False)

# Create rankfile for process binding
if rank == 0:
    hostname = socket.gethostname()
    with open('rankfile.txt', 'w') as f:
        for i in range(ncpu):
            cpu_id = i + size  # Start from core 4
            f.write(f'rank {i}={hostname} slot={cpu_id}\n')

# Set NEB Parameters
tag = 'h2s-RBF'
images = neb_generate_images('POSCAR_initial', 'POSCAR_final', 5,
                             IDPP=True, mic=True)

# Set VASP calculator
ncpu = cpu_count - size
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
base_calc = Vasp(**vasp_args)

# Set the GP model
if os.path.exists(tag + '-gpr.json'):
    gp_model = GP.load(tag + '-gpr.json')
    gp_model.fit()
else:
    gp_model = GP.set_GPR(images, base_calc,
                          kernel='RBF',
                          noise_e=0.05/len(images[0]),
                          noise_f=0.10)

# Set the calculator
calc = GPR(base_calculator=Vasp(**vasp_args),
           ff=gp_model,
           freq=10,
           tag=tag)

# Run NEB calculation
images, engs_gpr, _ = neb_calc(images, calc, nsteps=1000, fmax=0.05, method='FIRE')
if rank == 0:
    # Get the VASP reference
    images, engs_ref = neb_calc(images, base_calc, nsteps=0)
    label = f'GPR ({gp_model.count_use_base}/{gp_model.count_use_surrogate})'
    data = [(images, engs_ref, 'VASP'), (images, engs_gpr, label)]
    neb_plot_path(data, figname='neb.png')
