import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes#, PropertyNotImplementedError
from ase.neighborlist import NeighborList
from ase.constraints import full_3x3_to_voigt_6_stress
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class GPR(Calculator):
    implemented_properties = ['energy', 'forces', 'stress', 'var_e', 'var_f']
    nolabel = True

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.results = {}
        self.update = True
        self.verbose = False

    def freeze(self):
        self.update = False

    def unfreeze(self):
        self.update = True

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        self._calculate(atoms, properties, system_changes)
        e_tol = 1.2 * self.parameters.ff.noise_e
        f_tol = 1.2 * self.parameters.ff.noise_f
        label = self.parameters.tag
        E_std, F_std = self.results['var_e'], self.results['var_f'].max()
        E = self.results['energy']
        Fmax = np.abs(self.results['forces']).max()
        if E_std > e_tol or F_std > max([f_tol, Fmax/10]):
            self.parameters.ff.count_use_base += 1
            atoms.calc = self.parameters.base_calculator
            eng = atoms.get_potential_energy()
            forces = atoms.get_forces()
            data = (atoms, eng, forces)
            f_max = np.abs(forces).max()
            if rank == 0:
                print(f"From base model, E: {E_std:.3f}/{E:.3f}/{eng:.3f}, F: {F_std:.3f}/{Fmax:.3f}/{f_max:.3f}")
            self.parameters.ff.add_structure(data)

            # update model
            if self.update and self.parameters.ff.N_queue > self.parameters.freq:
                print("====================== Update the model ===============", self.parameters.ff.N_queue)
                self.parameters.ff.fit(opt=True, show=False)
                self.parameters.ff.save(f'{label}-gpr.json', f'{label}-gpr.db')
                if rank ==0:
                    self.parameters.ff.validate_data(show=True)
                    print(self.parameters.ff)
                    if self.parameters.ff.error['energy_mae'] > 0.1 or \
                        self.parameters.ff.error['forces_mae'] > 0.3:
                        print("ERROR: The error is too large, check the data.")
                        print(self.parameters.ff.error)
                        print("The program stops here!\n")
                        import sys; sys.exit()

            self.results['energy'] = eng
            self.results['forces'] = forces
            atoms.calc = self
        else:
            self.parameters.ff.count_use_surrogate += 1
            #print(F_std > max([f_tol, Fmax/10]), F_std, f_tol, Fmax/10)
            if rank == 0:
                print(f"From surrogate,  E: {E_std:.3f}/{e_tol:.3f}/{E:.3f}, F: {F_std:.3f}/{f_tol:.3f}/{Fmax:.3f}")

    def _calculate(self, atoms, properties, system_changes):

        Calculator.calculate(self, atoms, properties, system_changes)
        if hasattr(self.parameters, 'stress'):
            stress = self.parameters.stress
        else:
            stress = False
        if hasattr(self.parameters, 'f_tol'):
            f_tol = self.parameters.f_tol
        else:
            f_tol = 1e-12

        if hasattr(self.parameters, 'return_std'):
            return_std=self.parameters.return_std
        else:
            return_std=False

        if return_std:
            #print(atoms)
            res = self.parameters.ff.predict_structure(atoms, stress, True, f_tol=f_tol)
            self.results['var_e'] = res[3] #/20
            self.results['var_f'] = res[4]

        else:
            res = self.parameters.ff.predict_structure(atoms, stress, False, f_tol=f_tol)

        self.results['energy'] = res[0]
        self.results['free_energy'] = res[0]
        self.results['forces'] = res[1]
        if stress:
            self.results['stress'] = res[2].sum(axis=0) #*eV2GPa
        else:
            self.results['stress'] = None

    def get_var_e(self, total=False):
        if total:
            return self.results["var_e"] * len(self.results["forces"]) # eV
        else:
            return self.results["var_e"] # eV/atom

    def get_var_f(self):
        return self.results["var_f"]

    def get_e(self, peratom=True):
        if peratom:
            return self.results["energy"] / len(self.results["forces"])
        else:
            return self.results["energy"]


class LJ():
    """
    Pairwise LJ model (mostly copied from `ase.calculators.lj`)
    https://gitlab.com/ase/ase/-/blob/master/ase/calculators/lj.py

    Args:
        atoms: ASE atoms object
        parameters: dictionary to store the LJ parameters

    Returns:
        energy, force, stress
    """
    def __init__(self, parameters=None):
        # Set up default descriptors parameters
        keywords = ['rc', 'sigma', 'epsilon']
        _parameters = {
                       'name': 'LJ',
                       'rc': 5.0,
                       'sigma': 1.0,
                       'epsilon': 1.0,
                      }

        if parameters is not None:
            _parameters.update(parameters)

        self.load_from_dict(_parameters)

    def __str__(self):
        return "LJ(eps: {:.3f}, sigma: {:.3f}, cutoff: {:.3f})".format(\
        self.epsilon, self.sigma, self.rc)

    def load_from_dict(self, dict0):
        self._parameters = dict0
        self.name = self._parameters["name"]
        self.epsilon = self._parameters["epsilon"]
        self.sigma = self._parameters["sigma"]
        self.rc = self._parameters["rc"]


    def save_dict(self):
        """
        save the model as a dictionary in json
        """
        return self._parameters

    def calculate(self, atoms):
        """
        Compute the E/F/S

        Args:
            atom: ASE atoms object
        """

        sigma, epsilon, rc = self.sigma, self.epsilon, self.rc

        natoms = len(atoms)
        positions = atoms.positions
        cell = atoms.cell

        e0 = 4 * epsilon * ((sigma / rc) ** 12 - (sigma / rc) ** 6)

        energies = np.zeros(natoms)
        forces = np.zeros((natoms, 3))
        stresses = np.zeros((natoms, 3, 3))

        nl = NeighborList([rc / 2] * natoms, self_interaction=False)
        nl.update(atoms)

        for ii in range(natoms):
            neighbors, offsets = nl.get_neighbors(ii)
            cells = np.dot(offsets, cell)

            # pointing *towards* neighbours
            distance_vectors = positions[neighbors] + cells - positions[ii]

            r2 = (distance_vectors ** 2).sum(1)
            c6 = (sigma ** 2 / r2) ** 3
            c6[r2 > rc ** 2] = 0.0
            c12 = c6 ** 2

            pairwise_energies = 4 * epsilon * (c12 - c6) - e0 * (c6 != 0.0)
            energies[ii] += 0.5 * pairwise_energies.sum()  # atomic energies

            pairwise_forces = (-24 * epsilon * (2 * c12 - c6) / r2)[
                :, np.newaxis
            ] * distance_vectors

            forces[ii] += pairwise_forces.sum(axis=0)
            stresses[ii] += 0.5 * np.dot(
                pairwise_forces.T, distance_vectors
            )  # equivalent to outer product

            # add j < i contributions
            for jj, atom_j in enumerate(neighbors):
                energies[atom_j] += 0.5 * pairwise_energies[jj]
                forces[atom_j] += -pairwise_forces[jj]  # f_ji = - f_ij
                stresses[atom_j] += 0.5 * np.outer(
                    pairwise_forces[jj], distance_vectors[jj]
                )

        # whether or not output stress
        if atoms.number_of_lattice_vectors == 3:
            stresses = full_3x3_to_voigt_6_stress(stresses)
            stress = stresses / atoms.get_volume()
        else:
            stress = None

        energy = energies.sum()
        #print(energy)
        return energy, forces, stress

def get_pyscf_calc(atoms, basis='gth-szv-molopt-sr', pseudo='gth-pade', xc='lda,vwn'):

    from pyscf.pbc.tools import pyscf_ase
    import pyscf.pbc.gto as pbcgto
    import pyscf.pbc.dft as pbcdft

    cell = pbcgto.Cell()
    cell.a = atoms.cell
    cell.basis = basis
    cell.pseudo = pseudo
    cell.verbose = 0
    mf_class = pbcdft.RKS
    mf_class = lambda cell: pbcdft.KRKS(cell, kpts=cell.make_kpts([1, 1, 1]))
    mf_dict = { 'xc' : xc}

    return pyscf_ase.PySCF(molcell=cell, mf_class=mf_class, mf_dict=mf_dict)

if __name__ == '__main__':
    import pyscf
    from ase import Atoms
    from ase.io import read
    from ase.optimize import LBFGS

    from ase.lattice.cubic import Diamond
    si=Diamond(symbol='C', latticeconstant=3.5668)
    si.set_calculator(get_pyscf_calc(si))
    print(si.get_potential_energy())

    #initial = read('database/initial.traj')
    #final = read('database/final.traj')
    #initial.set_calculator(get_pyscf_calc(atoms))
    #final.calc = get_pyscf_calc(final)
    #print(initial.get_potential_energy())
    #print(final.get_potential_energy())

    #dyn_react = LBFGS(initial)
    #dyn_react.run(fmax=0.05)
    #dyn_prod = LBFGS(final)
    #dyn_prod.run(fmax=0.05)

    #images = ]
