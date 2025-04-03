import os, socket, psutil
from mpi4py import MPI
from ase.io import read
from gpr_calc.NEB import neb_calc, init_images, plot_path, get_vasp
from gpr_calc.calculator import GPR
from gpr_calc.gaussianprocess import GP

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
            f.write(f'rank {i}={hostname} slot={i+size}\n')

# Set VASP calculator
cmd = "mpirun --bind-to core --map-by rankfile:file=../rankfile.txt "
os.system("module load vasp/6.4.3")
os.environ["ASE_VASP_COMMAND"] = cmd + f"-np {ncpu} vasp_std"
os.environ["VASP_PP_PATH"] = "/projects/mmi/potcarFiles/VASP6.4"

# Modify the parameters here
init, final, numImages= 'POSCAR_initial', 'POSCAR_final', 7
noise_e, noise_f, tag = 0.05, 0.08, 'h2s-RBF'
#noise_e, noise_f, tag = 0.025, 0.08, 'h2s-RBF'
kpts = [2, 2, 1]

# Initialize the NEB images
if os.path.exists('neb.traj'):
    images = read('neb.traj', index=':')[-numImages:]
else:
    images = init_images(init, final, numImages, IDPP=True, mic=True)

# Set the GP calculators
base_calc = get_vasp(kpts=kpts)
noise_e = max([0.0004, noise_e/len(images[0])]) # Ensure noise_e is not too small
gp = GP.set_GPR(images, base_calc, noise_e=noise_e, noise_f=noise_f,
                json_file=tag+'-gpr.json', overwrite=True)
for i, image in enumerate(images):
    base_calc = get_vasp(kpts=kpts, directory=f"calc_{i}")
    image.calc = GPR(base=base_calc, ff=gp, freq=10, tag=tag)
    # Only invoke update_gpr on the first image
    image.calc.update_gpr = (i == 1)

# Run NEB calculation
for i, climb in enumerate([False, True, False]):
    neb, refs = neb_calc(images, steps=50, algo='FIRE',
                         fmax=noise_f, trajectory='neb.traj',
                         climb=climb, use_ref=True)

    images = neb.images
    # Plot the NEB path
    if rank == 0:
        print('NEB residuals:', neb.residuals)
        label = f'GPR ({gp.count_use_base}/{gp.count_use_surrogate})'
        data = [(images, refs, 'VASP'), (images, neb.energies, label)]
        plot_path(data, title='H2S on Pd(111)', figname=f'neb-{i}.png')
    
    if neb.converged:
        break
