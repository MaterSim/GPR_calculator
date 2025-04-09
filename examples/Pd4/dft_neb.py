import os, psutil
from gpr_calc.NEB import neb_calc, get_images, plot_path
from gpr_calc.utilities import get_vasp

# Set VASP calculator
ncpu = psutil.cpu_count(logical=False)
os.system("module load vasp/6.4.3")
os.environ["ASE_VASP_COMMAND"] = f"mpirun -np {ncpu} vasp_std"
os.environ["VASP_PP_PATH"] = "/projects/mmi/potcarFiles/VASP6.4"

# Modify the parameters here
init, final, numImages= 'POSCAR_initial', 'POSCAR_final', 7
kpts = [2, 2, 1]
traj, title = 'dft_neb.traj', 'Pd4 on MgO(100)'

# Initialize the NEB images
images = get_images(init, final, numImages, traj=traj,
                    IDPP=True, mic=True)

# Set the GP calculators
base_calc = get_vasp(kpts=kpts)
for i, image in enumerate(images):
    image.calc = get_vasp(kpts=kpts, directory=f"DFT/calc_{i}")

# Run NEB calculation
neb = neb_calc(images, steps=200, algo='FIRE', fmax=0.075,
               traj=traj, climb=True)

# Plot the NEB path
print('NEB residuals:', neb.residuals)
data = [(images, neb.energies, 'VASP')]
plot_path(data, title=title, figname=f'dft_neb.png')
