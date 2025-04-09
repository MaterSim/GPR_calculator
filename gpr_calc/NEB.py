"""
NEB related functions
"""
from ase.mep import NEB
from ase.geometry import find_mic
from ase.io.trajectory import Trajectory
from ase.io import read

def neb_calc(images, calculator=None, algo='BFGS',
             fmax=0.05, steps=100, k=0.1,
             climb=False, traj=None,
             use_ref=False):
    """
    NEB calculation with ASE's NEB module
    The function will return the images and energies of the NEB calculation

    Args:
        images: list of initial images for NEB calculation
        calculator: calculator for the NEB calculation
        algo: algorithm for the NEB calculation (BFGS, FIRE, etc.)
        fmax: maximum force
        steps: maximum number of steps
        k: spring constant (optional)
        climb: climb option (optional)
        traj: trajectory file name (optional)
        use_ref: if True, return the reference energies

    Returns:
        neb: NEB object with the results of the calculation
    """
    from ase.optimize import BFGS, FIRE
    from copy import copy

    # Set NEB calculation
    neb = NEB(images, k=k, parallel=False)
    neb.climb = climb
    # Set the calculator for the images
    if calculator is not None:
        for i, image in enumerate(images):
            image.calc = copy(calculator)
            # only allow the last image to be updated
            if calculator.name == 'gpr':
                if i == 1:
                    image.calc.update_gpr = True
                else:
                    image.calc.update_gpr = False

    # Set the optimizer
    if algo == 'BFGS':
        opt = BFGS(neb, trajectory=traj, append_trajectory=True)
    elif algo == 'FIRE':
        if trajectory is not None:
            traj1 = Trajectory(traj1, 'a')
            opt = FIRE(neb, trajectory=traj1)
        else:
            opt = FIRE(neb)
    else:
        raise ValueError('Invalid algorithm for NEB calculation')
    opt.run(fmax=fmax, steps=steps)
    neb.nsteps = opt.nsteps + 1
    neb.converged = opt.converged()

    for i, image in enumerate(images):
        # Use the reference energy for the first and last images
        if image.calc.name == 'gpr':
            if i in [0, len(images)-1]:
                neb.energies[i] = image.calc.parameters.ff.train_y['energy'][i]*len(image)
            else:
                image.calc.freeze()
                neb.energies[i] = image.get_potential_energy()
                image.calc.unfreeze()
        else:
            neb.energies[i] = image.get_potential_energy()

    if use_ref:
        ref_engs = []
        for i, image in enumerate(images):
            if i in [0, len(images)-1]:
                ref_engs.append(neb.energies[i])
            else:
                # Reset the calculator to get the reference energy
                image.calc.results = image.calc.result = {}
                if 'forces' in image.arrays: del image.arrays['forces']
                image.calc.force_base = True
                ref_engs.append(image.get_potential_energy())
                image.calc.force_base = False
        return neb, ref_engs
    else:
        return neb

def get_images(init, final, num_images=5, vaccum=0.0,
               traj=None, IDPP=False, mic=False,
               apply_constraint=False):
    """
    Generate initial images from ASE's NEB module
    The number of images generated is self.num_images - 2

    Args:
        init: initial structure file
        final: final structure file
        num_images: number of images
        vaccum: vacuum size in angstrom
        traj: trajectory file name
        IDPP: use the improved dimer
        mic: use the minimum image convention
        apply_constraint: apply constraint to the images

    Returns:
        images: list of initial images for NEB calculation
    """
    if traj is not None and os.path.exists(traj):
        images = read(traj, index=':')[-num_images:]
        return images

    initial, final = read(init), read(final)
    num_images = num_images

    # Set the PBC condition (mostly for surfaces)
    if initial.pbc[-1] and vaccum > 0:
        def set_pbc(atoms, vacuum=vaccum):
            atoms.cell[2, 2] += vacuum
            atoms.center()
            atoms.pbc = [True, True, True]
            return atoms
        initial, final = set_pbc(initial), set_pbc(final)

    # Make the list of images
    images = [initial] + [initial.copy() for i in range(num_images-2)] + [final]

    # Set intermediate images
    neb = NEB(images, parallel=False)
    if IDPP:
        neb.interpolate(method='idpp', mic=mic, apply_constraint=apply_constraint)
    else:
        neb.interpolate(apply_constraint=apply_constraint, mic=mic)

    return images

def plot_path(data, unit='eV', fontsize=15, figname='neb_path.png',
                  title='NEB Path', max_yticks=8, x_scale=False):
    """
    Function to plot the NEB path

    Args:
        data: nested list [(imgs1, engs1, label2), (img2, engs, label2)]
        unit: unit of energy
        fontsize: font size of the plot
        figname: name of the figure file
        title: title of the plot
        max_yticks: maximum number of yticks
        x_scale: scale for the x-axis (default is False)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from scipy.interpolate import make_interp_spline

    plt.figure(figsize=(8, 6))
    for d in data:
        (images, Y, label) = d
        tmp = np.array([image.positions for image in images])
        X = np.zeros(len(images))
        for i in range(len(tmp)-1):
            # Find the minimum image convention
            d = tmp[i+1] - tmp[i]
            d = find_mic(d, images[0].get_cell(), images[0].pbc)[0]
            X[i+1] = np.linalg.norm(d)

        # Normalize the distance
        X = np.cumsum(X)
        if x_scale: X /= X[-1]

        X_smooth = np.linspace(min(X), max(X), 30)
        spline = make_interp_spline(X, Y, k=3, bc_type=([(1, 0.0)], [(1, 0.0)]))
        Y_smooth = spline(X_smooth)
        line, = plt.plot(X, Y, 'o')  # Get the line object
        plt.plot(X_smooth, Y_smooth, ls='--', label=label, color=line.get_color())

    x1, x2 = plt.xlim()
    plt.xlim(x1, x2 * 1.1)
    plt.gca().yaxis.set_major_locator(MaxNLocator(max_yticks))
    plt.xlabel('Reaction Coordinates', fontsize=fontsize)
    plt.ylabel(f'Energy ({unit})', fontsize=fontsize)
    plt.title(title, fontsize=fontsize*1.1)
    plt.legend(fontsize=fontsize, frameon=False, loc=1)
    plt.xticks(fontsize=fontsize*0.9)
    plt.yticks(fontsize=fontsize*0.9)
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.close()


def plot_progress(trajectory, calc, N_images, start=0, interval=50,
                  figname='neb-process.png'):
    """
    Parse NEB convergence results from trajectory.
    We compute the energy/distances and check if they are below the threshold.

    Args:
        trajectory: ASE trajectory file
        calc: calculator for the NEB calculation
        N_images: number of images in the NEB calculation
        start: starting step for the NEB calculation
        interval: interval for the NEB calculation
        figname: name of the figure file

    """
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()
    traj = read(trajectory, index=':')
    N_max = int(len(traj) / N_images)

    data = []
    for step in range(start, N_max, interval):
        if rank == 0: print(f'Processing step {step} of {N_max}')
        images = traj[step*N_images:(step+1)*N_images]
        engs = []

        for i, image in enumerate(images):
            # Use the reference energy for the first and last images
            if i in [0, len(images)-1]:
                eng = calc.parameters.ff.train_y['energy'][i]*len(image)
            else:
                image.calc = calc
                eng = image.get_potential_energy()
            engs.append(eng)
        data.append((images, engs, f'NEB_iter_{step}'))

    # Plot the NEB path
    if rank == 0:
        plot_path(data, figname=figname)
