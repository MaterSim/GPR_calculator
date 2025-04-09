import os
from gpr_calc.NEB import neb_calc, get_images, plot_path
from gpr_calc.calculator import GPR
from gpr_calc.gaussianprocess import GP
from gpr_calc.utilities import set_mpi, get_vasp

# Modify the parameters here
init, final, numImages= 'POSCAR_initial', 'POSCAR_final', 7
noise_e, noise_f, kpts = 0.03, 0.075, [3, 3, 1]
tag, traj, title = 'H2S', 'gp_neb.traj', 'H2S on Pd(100)'

# Set MPI and VASP calculator
rank, ncpu = set_mpi()
cmd = "mpirun --bind-to core --map-by rankfile:file=../../rankfile.txt "
os.system("module load vasp/6.4.3")
os.environ["ASE_VASP_COMMAND"] = cmd + f"-np {ncpu} vasp_std"
os.environ["VASP_PP_PATH"] = "/projects/mmi/potcarFiles/VASP6.4"

# Initialize the NEB images
images = get_images(init, final, numImages, traj=traj,
                    IDPP=True, mic=True)

# Set the GP calculators
base_calc = get_vasp(kpts=kpts)
noise_e = max([0.0004, noise_e/len(images[0])]) # Ensure noise_e is not too small
gp = GP.set_GPR(images, base_calc, noise_e=noise_e, noise_f=noise_f,
                json_file=tag+'-gpr.json', overwrite=True)
for i, image in enumerate(images):
    base_calc = get_vasp(kpts=kpts, directory=f"GP/calc_{i}")
    image.calc = GPR(base=base_calc, ff=gp, freq=10, tag=tag)
    # Only invoke update_gpr on the first image
    image.calc.update_gpr = (i == 1)

# Run NEB calculation
for i, climb in enumerate([False, True, False]):
    neb, refs = neb_calc(images, steps=50, algo='FIRE',
                         fmax=noise_f, trajectory=traj,
                         climb=climb, use_ref=True)

    images = neb.images
    # Plot the NEB path
    if rank == 0:
        print('NEB residuals:', neb.residuals)
        label = f'GPR ({gp.count_use_base}/{gp.count_use_surrogate})'
        data = [(images, refs, 'VASP'), (images, neb.energies, label)]
        plot_path(data, title=title, figname=f'gp_neb_{i}.png')

    if neb.converged:
        break
