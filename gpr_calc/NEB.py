"""
NEB related functions
"""
from ase.mep import NEB

def neb_calc(images, calculator=None, algo='BFGS',
             fmax=0.05, steps=100, k=0.1, trajectory=None):
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
        trajectory: trajectory file name (optional)

    Returns:
        images: list of images after NEB calculation
        eng: list of energies of the images
    """
    from ase.optimize import BFGS, FIRE
    from copy import copy

    # Set NEB calculation
    neb = NEB(images, k=k, parallel=False)

    # Set the calculator for the images
    if calculator is not None:
        for image in images:
            image.calc = copy(calculator)

    # Set the optimizer
    if algo == 'BFGS':
        opt = BFGS(neb, trajectory=trajectory)
    elif algo == 'FIRE':
        opt = FIRE(neb, trajectory=trajectory)
    else:
        raise ValueError('Invalid algorithm for NEB calculation')
    opt.run(fmax=fmax, steps=steps)
    eng = [image.get_potential_energy() for image in images]

    # Return the images and energie
    return images, eng, opt.nsteps

def init_images(init, final, num_images=5, vaccum=0.0, 
                IDPP=False, mic=False, apply_constraint=False):
    """
    Generate initial images from ASE's NEB module
    The number of images generated is self.num_images - 2

    Args:
        init: initial structure file
        final: final structure file
        num_images: number of images
        vaccum: vacuum size in angstrom
        IDPP: use the improved dimer
        mic: use the minimum image convention
        apply_constraint: apply constraint to the images

    Returns:
        images: list of initial images for NEB calculation
    """
    from ase.io import read

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

def neb_plot_path(data, unit='eV', figname='neb_path.png'):
    """
    Function to plot the NEB path

    Args:
        data: nested list [(imgs1, engs1, label2), (img2, engs, label2)]
        unit: unit of energy
        figname: name of the figure file
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import make_interp_spline

    colors = ['r', 'b', 'g', 'm'][:len(data)]
    plt.figure()
    for d, color in zip(data, colors):
        (images, Y, label) = d
        tmp = np.array([image.positions for image in images])
        X = np.cumsum([np.linalg.norm(tmp[i] - tmp[i+1]) for i in range(len(tmp)-1)])
        X = [0] + X.tolist()  # Add the initial point at distance 0
        X = np.array(X)
        X /= X[-1]  # Normalize the distance
        X_smooth = np.linspace(min(X), max(X), 30)
        spline = make_interp_spline(X, Y, k=3)
        Y_smooth = spline(X_smooth)
        plt.plot(X, Y, 'o')
        plt.plot(X_smooth, Y_smooth, ls='--', color=color, label=label)

    plt.xlabel('Normalized Reaction Coordinates')
    plt.ylabel(f'Energy ({unit})')
    plt.title('Simulated NEB Path')
    plt.legend()
    plt.grid(True)
    plt.savefig(figname)
    plt.close()


def get_vasp_calculator(**kwargs):
    """
    Set up and return a VASP calculator with specified parameters.

    Args:
        **kwargs: Additional VASP parameters to override defaults
        
    Returns:
        ase.calculators.vasp.Vasp: Configured VASP calculator
    """
    from ase.calculators.vasp import Vasp

    vasp_args = {
        "txt": 'vasp.out',
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
        "npar": 2,
        "kpts": [2, 2, 1],
    }
    vasp_args.update(kwargs)
    
    return Vasp(**vasp_args)