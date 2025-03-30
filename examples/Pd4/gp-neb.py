from gpr_calc.NEB import neb_calc, init_images, neb_plot_path, get_vasp_calculator
from gpr_calc.calculator import GPR
from gpr_calc.gaussianprocess import GP
from mpi4py import MPI
import os
import socket
import psutil

# Set MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
cpu_count = psutil.cpu_count(logical=False)
ncpu = cpu_count - size

# Create rankfile for process binding
if rank == 0:
    hostname = socket.gethostname()
    with open('rankfile.txt', 'w') as f:
        for i in range(ncpu):
            cpu_id = i + size
            f.write(f'rank {i}={hostname} slot={cpu_id}\n')

# Set VASP calculator
os.system("module load vasp/6.4.3")
os.environ["ASE_VASP_COMMAND"] = (
    "mpirun --bind-to core --map-by rankfile:file=../rankfile.txt "
    f"-np {ncpu} vasp_std")
os.environ["VASP_PP_PATH"] = "/projects/mmi/potcarFiles/VASP6.4"
kpts = [2, 2, 1]

# Set NEB Images
init, final, tag = 'POSCAR_initial', 'POSCAR_final', 'Pd4-RBF'
images = init_images(init, final, 7, IDPP=True, mic=True)

# Set the GP model
base_calc = get_vasp_calculator(kpts=kpts)
noise_e, noise_f = 0.03/len(images[0]), 0.10
gp_model = GP.set_GPR(images, base_calc,
                      noise_e=noise_e,
                      noise_f=noise_f,
                      json_file=tag+'-gpr.json')

# Set the GP calculator
for i, image in enumerate(images):
    base_calc = get_vasp_calculator(kpts=kpts, directory=f"calc_{i}")
    image.calc = GPR(base=base_calc, ff=gp_model, freq=10, tag=tag)

# Run NEB calculation
images, engs_gpr, _ = neb_calc(images, steps=1000, algo='FIRE')

# Plot the NEB path
if rank == 0:
    # Get the VASP reference
    for i, image in enumerate(images):
        image.calc = get_vasp_calculator(kpts=kpts, directory=f"calc_{i}")
    images, engs_ref, _ = neb_calc(images, steps=0)
    label = f'GPR ({gp_model.count_use_base}/{gp_model.count_use_surrogate})'
    data = [(images, engs_ref, 'VASP'), (images, engs_gpr, label)]
    neb_plot_path(data, figname='neb.png')
