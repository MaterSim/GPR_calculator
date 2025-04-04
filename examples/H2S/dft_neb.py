import os, psutil
from ase.io import read
from gpr_calc.NEB import neb_calc, init_images, plot_path, get_vasp

# Set MPI
ncpu = psutil.cpu_count(logical=False)

# Set VASP calculator
os.system("module load vasp/6.4.3")
os.environ["ASE_VASP_COMMAND"] = f"mpirun -np {ncpu} vasp_std"
os.environ["VASP_PP_PATH"] = "/projects/mmi/potcarFiles/VASP6.4"

# Modify the parameters here
init, final, numImages= 'POSCAR_initial', 'POSCAR_final', 7
kpts = [3, 3, 1]
traj = 'dft_neb.traj'

# Initialize the NEB images
if os.path.exists(traj):
    images = read(traj, index=':')[-numImages:]
else:
    images = init_images(init, final, numImages, IDPP=True, mic=True)

# Set the GP calculators
base_calc = get_vasp(kpts=kpts)
for i, image in enumerate(images):
    image.calc = get_vasp(kpts=kpts, directory=f"DFT/calc_{i}")

# Run NEB calculation
neb = neb_calc(images, steps=200, algo='FIRE', fmax=0.075,
               trajectory=traj, climb=True)

# Plot the NEB path
print('NEB residuals:', neb.residuals)
data = [(images, neb.energies, 'VASP')]
plot_path(data, title='H2S on Pd(100)', figname=f'dft_neb.png')
