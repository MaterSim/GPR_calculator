import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize

from pyxtal.database.element import Element
from .utilities import new_pt, convert_train_data, list_to_tuple, tuple_to_list, metric_values
from .SO3 import SO3
from .kernels.Dot_mb import Dot_mb
from .kernels.RBF_mb import RBF_mb

import json
from ase.db import connect
import os
from copy import deepcopy
from mpi4py import MPI
import logging

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class GaussianProcess():
    """
    Gaussian Process Regressor class to fit the interatomic potential
    from the reference energy/forces.

    Main APIs:
        - fit(): fit the GPR model
        - predict_structure(struc): predict the energy/forces/stress for a structure
        - add_structure((struc, energy, forces)): add a new structure to training
        - sparsify(): reduce training data by removing unimportant configurations

    Main attributes:
        - kernel: the kernel function
        - descriptor: the descriptor function
        - base_potential: the base potential before GPR
        - noise_e: the energy noise
        - f_coef: the coefficient of force noise relative to energy
        - K: the covariance matrix
        - _K_inv: the inverse of the covariance matrix

    Args:
        kernel (callable): compute the covariance matrix
        descriptor (callable): compute the structure to descriptor
        base_potential (callable): compute the base potential before GPR
        f_coef (float): the coefficient of force noise relative to energy
        noise_e (list): define the energy noise (init, lower, upper)
    """

    def __init__(self, kernel, descriptor,
                 base_potential=None,
                 noise_e=0.005, #[5e-3, 2e-3, 1e-1],
                 noise_f=0.1, #[1e-1, 1e-2, 2e-1],
                 f_coef=10,
                 log_file="gpr.log"):

        # Setup logging
        self.log_file = log_file
        logging.getLogger().handlers.clear()
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s| %(message)s',
                            filename=self.log_file)
        self.logging = logging

        self.comm = comm
        self.rank = rank
        self.size = size
        if type(noise_e) is not list:
            self.noise_e = noise_e
            self.noise_f = noise_f
            self.noise_bounds = None
        else:
            self.noise_e = noise_e[0]
            self.noise_f = noise_f[0]
            self.noise_bounds = noise_e[1:]
        self.f_coef = f_coef
        # print("Debug", self.noise_e, self.noise_bounds)
        self.error = None

        self.descriptor = descriptor
        self.kernel = kernel
        self.base_potential = base_potential

        self.x = None
        self.train_x = None
        self.train_y = None
        self.train_db = None
        self.alpha_ = None
        self.N_energy = 0
        self.N_forces = 0
        self.N_energy_queue = 0
        self.N_forces_queue = 0
        self.N_queue = 0

        # For the track of function calls
        self.count_fits = 0
        self.count_use_base = 0
        self.count_use_surrogate = 0

        # For the track of parameters
        if self.rank == 0: self.logging.info(self)

    def __str__(self):
        s = f"------Gaussian Process Regression ({self.rank}/{self.size})------\n"
        s += "Kernel: {:s}".format(str(self.kernel))
        if hasattr(self, "train_x"):
            s += " {:d} energy ({:.5f})".format(self.N_energy, self.noise_e)
            s += " {:d} forces ({:.5f})\n".format(self.N_forces, self.noise_f)

        if self.count_use_base > 0:
            s += "Total number of base calls: {:d}\n".format(self.count_use_base)
            s += "Total number of surrogate calls: {:d}\n".format(self.count_use_surrogate)
            s += "Total number of gpr_fit calls: {:d}\n".format(self.count_fits)
        return s

    def todict(self):
        """
        Added for ASE compatibility
        """
        return {
            #'param1': self.param1,
            #'param2': self.param2,
            # Add other parameters and attributes as needed
        }

    def __repr__(self):
        return str(self)

    def set_K_inv(self):
        if self._K_inv is None:
            L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
            self._K_inv = L_inv.dot(L_inv.T)

    def log_marginal_likelihood(self, params, eval_gradient=False, clone_kernel=False):
        """
        Compute the log marginal likelihood and its gradient

        Args:
            params: hyperparameters
            eval_gradient: bool, evaluate the gradient or not
            clone_kernel: bool, clone the kernel or not

        Returns:
            log marginal likelihood and its gradient
        """
        if self.noise_bounds is None:
            noise_e = self.noise_e
            noise_f = self.noise_f
            kernel_params = params
        else:
            noise_e = params[-1]
            noise_f = self.f_coef * noise_e
            kernel_params = params[:-1]

        if clone_kernel:
            kernel = self.kernel.update(kernel_params)
        else:
            kernel = self.kernel
            kernel.update(kernel_params)

        if eval_gradient:
            K, K_gradient = kernel.k_total_with_grad(self.train_x)
        else:
            K = kernel.k_total(self.train_x)

        noise = np.eye(len(K))
        if len(self.train_x['energy']) > 0:
            NE = len(self.train_x['energy'][-1])
        else:
            NE = 0

        noise[:NE, :NE] *= noise_e**2
        noise[NE:, NE:] *= noise_f**2
        K += noise
        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(params)) if eval_gradient else -np.inf

        y_train = self.y_train
        alpha = cho_solve((L, True), y_train)

        # log marginal likelihood
        ll_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
        ll_dims -= np.log(np.diag(L)).sum()
        ll_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        MLL = ll_dims.sum(-1)  #sum over dimensions

        if eval_gradient:  # compare Equation 5.9 from GPML
            base = np.zeros([len(K), len(K), 1]) #for energy and force noise
            base[:NE,:NE, 0] += 2 * noise_e * np.eye(NE)
            #base[NE:,NE:, 0] += 2 * self.f_coef**2 * noise_e * np.eye(len(K)-NE)
            base[NE:,NE:, 0] += 2 * noise_f * np.eye(len(K)-NE)
            K_gradient = np.concatenate((K_gradient, base), axis=2)
            tmp = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
            tmp -= cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]
            llg_dims = 0.5 * np.einsum("ijl,jik->kl", tmp, K_gradient)
            llg = llg_dims.sum(-1)
            if self.noise_bounds is None: llg = llg[:-1]
            # print(f"eval_grad_Loss: {-MLL}, {llg}, {str(kernel)}")
            return MLL, llg
        else:
            return MLL

    def optimize(self, fun, theta0, bounds):
        """
        Optimize the hyperparameters of the GPR model from scipy.minimize

        Args:
            fun: the function to minimize
            theta0: initial guess of hyperparameters
            bounds: bounds of hyperparameters
        """
        # print(f"Optimizing function rank-{self.rank}", theta0, bounds)
        opt_res = minimize(fun, theta0,
                           method="L-BFGS-B",
                           bounds=bounds,
                           jac=True,
                           options={'maxiter': 10, 'ftol': 1e-2})
        return opt_res.x, opt_res.fun

    def fit(self, TrainData=None, show=True, opt=True):
        """
        Fit the GPR model with optional MPI support

        Args:
            TrainData: a dictionary of energy/force/db data
            show: bool, print the information or not
            opt: bool, optimize the hyperparameters or not
        """
        if TrainData is None:
            self.update_y_train()
        else:
            self.set_train_pts(TrainData)

        if self.rank == 0 and show:
            print(self)

        def obj_func(params, eval_gradient=True):
            if eval_gradient:
                lml, grad = self.log_marginal_likelihood(
                    params, eval_gradient=True, clone_kernel=False)

                # Reduce results across ranks
                lml = self.comm.allreduce(lml, op=MPI.SUM) / self.size
                grad = self.comm.allreduce(grad, op=MPI.SUM) / self.size

                if show:
                    strs = "Loss: {:12.3f} ".format(-lml)
                    for para in params:
                        strs += "{:6.3f} ".format(para)
                    if self.rank == 0:
                        print(strs)
                        self.logging.info(strs)
                    #from scipy.optimize import approx_fprime
                    #print("from ", grad, lml)
                    #print("scipy", approx_fprime(params, self.log_marginal_likelihood, 1e-3))
                    #print("scipy", approx_fprime(params, self.log_marginal_likelihood, 1e-4))
                    #import sys; sys.exit()
                return (-lml, -grad)
            else:
                lml = self.log_marginal_likelihood(params, clone_kernel=False)
                lml = self.comm.allreduce(lml, op=MPI.SUM) / self.size
                return -lml

        hyper_params = self.kernel.parameters()
        hyper_bounds = self.kernel.bounds
        if self.noise_bounds is not None:
            hyper_params += [self.noise_e]
            hyper_bounds += [self.noise_bounds]

        if opt:
            if self.rank == 0: print(f"\nUpdate GP model => {self.N_queue}")
            params, _ = self.optimize(obj_func, hyper_params, hyper_bounds)
            # print(f"Optimized hyperparameters rank-{self.rank}", params, fun)
            params = self.comm.bcast(params, root=0)

            if self.noise_bounds is not None:
                self.kernel.update(params[:-1])
                self.noise_e = params[-1]
                self.noise_f = self.f_coef * params[-1]
            else:
                self.kernel.update(params)

        K = self.kernel.k_total(self.train_x)#; print(K)

        if self.rank == 0:
            # add noise matrix (assuming force/energy has a coupling)
            noise = np.eye(len(K))
            NE = len(self.train_x['energy'][-1])
            noise[:NE, :NE] *= self.noise_e**2
            noise[NE:, NE:] *= self.noise_f**2
            K += noise

            # self.logging.info("Starting Cholesky Decomp on rank 0")
            try:
                self.L_ = cholesky(K, lower=True)  # Line 2
                self._K_inv = None
            except np.linalg.LinAlgError as exc:
                exc.args = ("The kernel, %s, is not returning a "
                            "positive definite matrix. Try gradually "
                            "increasing the 'alpha' parameter of your "
                            "GaussianProcessRegressor estimator."
                            % self.kernel,) + exc.args
                raise
            self.alpha_ = cho_solve((self.L_, True), self.y_train)
            self.logging.info("Cholesky Decomp is Complete on rank 0")
        else:
            self.L_ = None
            self.alpha_ = None
            self._K_inv = None

        # Broadcast L_ and alpha_ to all ranks
        self.L_ = self.comm.bcast(self.L_, root=0)
        self.alpha_ = self.comm.bcast(self.alpha_, root=0)
        self._K_inv = self.comm.bcast(self._K_inv, root=0)

        # Synchronize the ranks
        self.comm.barrier()

        # reset the queue to 0
        self.N_energy_queue = 0
        self.N_forces_queue = 0
        self.N_queue = 0
        self.count_fits += 1
        self.set_K_inv()

    def predict(self, X, stress=False, total_E=False, return_std=False, return_cov=False):
        """
        Internal predict function for the GPR model

        Args:
            X: a dictionary of energy/force data
            stress: bool, return stress or not
            total_E: bool, return total energy or not
            return_std: bool, return variance or not
            return_cov: bool, return covariance or not
        """
        train_x = self.get_train_x()
        if stress:
            K_trans, _ = self.kernel.k_total_with_stress(X, train_x, same=False)
            #pred1 = K_trans1.dot(self.alpha_)
        else:
            K_trans = self.kernel.k_total(X, train_x)

        #print('debug', K_trans.shape, self.alpha_.shape)
        pred = K_trans.dot(self.alpha_)
        y_mean = pred[:, 0]

        Npts = 0
        if 'energy' in X:
            if isinstance(X["energy"], tuple): #big array
                Npts += len(X["energy"][-1])
            else:
                Npts += len(X["energy"])
        if 'force' in X:
            if isinstance(X["force"], tuple): #big array
                Npts += 3*len(X["force"][-1])
            else:
                Npts += 3*len(X["force"])

        factors = np.ones(Npts)

        if total_E:
            if isinstance(X["energy"], tuple): #big array
                N_atoms = np.array([x for x in X["energy"][-1]])
            else:
                N_atoms = np.array([len(x) for x in X["energy"]])
            factors[:len(N_atoms)] = N_atoms
        y_mean *= factors

        if return_cov:
            v = cho_solve((self.L_, True), K_trans.T)
            y_cov = self.kernel.k_total(X) - K_trans.dot(v)
            return y_mean, y_cov
        elif return_std:
            y_var = self.kernel.diag(X)
            y_var -= np.einsum("ij,ij->i", np.dot(K_trans, self._K_inv), K_trans)

            # If get negative variance, set to 0.
            y_var_negative = y_var < 0
            if np.any(y_var_negative) and self.rank == 0:
                print("Warning: Get negative variance")
            y_var[y_var_negative] = 0.0

            return y_mean, np.sqrt(y_var)*factors
        else:
            return y_mean

    def set_train_pts(self, data, mode="w"):
        """
        Set the training pts for the GPR mode
        two modes ("write" and "append") are allowed

        Args:
            data: a dictionary of energy/force/db data
            mode: "w" or "a+", reset or append the training data
        """
        if mode == "w" or self.train_x is None: #reset
            self.train_x = {'energy': [], 'force': []}
            self.train_y = {'energy': [], 'force': []}
            self.train_db = []

        N_E, N_F = 0, 0
        for d in data["db"]:
            (atoms, energy, force, energy_in, force_in) = d
            if energy_in:
                e_id = deepcopy(N_E + 1)
                N_E += 1
            else:
                e_id = None

            if len(force_in) > 0:
                f_ids = [N_F+i for i in range(len(force_in))]
                N_F += len(force_in)
            else:
                f_ids = []
            self.train_db.append((atoms, energy, force, energy_in, force_in, e_id, f_ids))

        for key in data.keys():
            if key == 'energy':
                if len(data[key])>0:
                    self.add_train_pts_energy(data[key])
            elif key == 'force':
                if len(data[key])>0:
                    self.add_train_pts_force(data[key])

        self.update_y_train()
        self.N_energy += N_E
        self.N_forces += N_F
        self.N_energy_queue += N_E
        self.N_forces_queue += N_F
        self.N_queue += N_E + N_F
        #print(self.train_y['energy'], N_E, N_F, self.N_energy_queue)

    def remove_train_pts(self, e_ids, f_ids):
        """
        Delete the training pts for the GPR model

        Args:
            e_ids: ids to delete in K_EE
            f_ids: ids to delete in K_FF
        """
        data = {"energy":[], "force": [], "db": []}
        energy_data = tuple_to_list(self.train_x['energy'], mode='energy')
        force_data = tuple_to_list(self.train_x['force'])

        N_E, N_F = len(energy_data), len(force_data)
        for id in range(N_E):
            if id not in e_ids:
                (X, ele) = energy_data[id]
                E = self.train_y['energy'][id]
                data['energy'].append((X, E, ele))

        for id in range(N_F):
            if id not in f_ids:
                (X, dxdr, ele) = force_data[id]
                F = self.train_y['force'][id]
                data["force"].append((X, dxdr, F, ele))

        for d in self.train_db:
            (atoms, energy, force, energy_in, force_in, e_id, _f_ids) = d
            if e_id in e_ids:
                energy_in = False
            _force_in = []
            for i, f_id in enumerate(_f_ids):
                if f_id not in f_ids:
                    _force_in.append(force_in[i])
            if energy_in or len(_force_in)>0:
                data['db'].append((atoms, energy, force, energy_in, _force_in))

        self.set_train_pts(data) # reset the train data
        self.fit()

    def compute_base_potential(self, atoms):
        """
        Compute the energy/forces/stress from the base_potential
        """
        return self.base_potential.calculate(atoms)

    def update_y_train(self):
        """
        Convert self.train_y to 1D numpy array
        """
        Npt_E = len(self.train_y["energy"])
        Npt_F = 3*len(self.train_y["force"])
        y_train = np.zeros([Npt_E+Npt_F, 1])
        count = 0
        for i in range(len(y_train)):
            if Npt_E > 0 and i < Npt_E:
                y_train[i,0] = self.train_y["energy"][i]
            else:
                if (i-Npt_E)%3 == 0:
                    #print(i, count, y_train.shape, self.train_y["force"][count])
                    y_train[i:i+3,0] = self.train_y["force"][count]
                    count += 1
        self.y_train=y_train

    def validate_data(self, test_data=None, total_E=False, return_std=False, show=False):
        """
        Validate the given dataset

        Args:
            test_data: a dictionary of energy/force data
            total_E: bool, return total energy or not
            return_std: bool, return variance or not
            show: bool, print the information or not
        """
        if test_data is None: #from train
            test_X_E = {"energy": self.train_x['energy']}
            test_X_F = {"force": self.train_x['force']}
            E = self.y_train[:len(test_X_E['energy'][-1])].flatten()
            F = self.y_train[len(test_X_E['energy'][-1]):].flatten()
        else:
            test_X_E = {"energy": [(data[0], data[2]) for data in test_data['energy']]}
            test_X_F = {"force": [(data[0], data[1], data[3]) for data in test_data['force']]}
            E = np.array([data[1] for data in test_data['energy']])
            F = np.array([data[2] for data in test_data['force']]).flatten()
            #test_X_E = list_to_tuple(test_X_E["energy"], mode="energy")
            #test_X_F = list_to_tuple(test_X_F["force"])

        if total_E:
            for i in range(len(E)):
                E[i] *= len(test_X_E['energy'][i])

        E_Pred, E_std, F_Pred, F_std = None, None, None, None

        if return_std:
            if len(test_X_E['energy']) > 0:
                E_Pred, E_std = self.predict(test_X_E, total_E=total_E, return_std=True)
            if len(test_X_F['force']) > 0:
                F_Pred, F_std = self.predict(test_X_F, return_std=True)
            if show:
                self.update_error(E, E_Pred, F, F_Pred)
            return E, E_Pred, E_std, F, F_Pred, F_std
        else:
            if len(test_X_E['energy']) > 0:
                E_Pred = self.predict(test_X_E, total_E=total_E)
            if len(test_X_F['force']) > 0:
                F_Pred = self.predict(test_X_F)

            if show:
                self.update_error(E, E_Pred, F, F_Pred)
            return E, E_Pred, F, F_Pred

    def update_error(self, E, E_Pred, F, F_Pred):
        """
        Update the training error for the model
        """
        e_r2, e_mae, e_rmse = metric_values(E, E_Pred)
        f_r2, f_mae, f_rmse = metric_values(F, F_Pred)
        self.error = {"energy_r2": e_r2,
                      "energy_mae": e_mae,
                      "energy_rmse": e_rmse,
                      "forces_r2": f_r2,
                      "forces_mae": f_mae,
                      "forces_rmse": f_rmse}
        if self.rank == 0:
            for key in self.error.keys():
                self.logging.info(f"{key:<12s}: {self.error[key]:.4f}")

    def get_train_x(self):
        """
        Get the current training data (excluding the data on the queue)
        """
        if self.N_queue > 0:
            train_x = {}
            (_X, _ELE, _indices) = self.train_x['energy']
            NE = self.N_energy - self.N_energy_queue
            if NE > 0:
                ids = sum(_indices[:NE])
                train_x['energy'] = (_X[:ids], _ELE[:ids], _indices[:NE])
            else:
                train_x['energy'] = (_X, _ELE, _indices)

            NF = self.N_forces - self.N_forces_queue
            (_X, _dXdR, _ELE, _indices) = self.train_x['force']
            if NF > 0:
                ids = sum(_indices[:NF])
                #print("debug", NF, _X.shape, _dXdR.shape, _ELE.shape, _indices, ids)
                train_x['force'] = (_X[:ids], _dXdR[:ids], _ELE[:ids], _indices[:NF])
            else:
                train_x['force'] = (_X, _dXdR, _ELE, _indices)
            return train_x
        else:
            return self.train_x

    def add_train_pts_energy(self, energy_data):
        """
        A function to add the energy data to the training.

        Args:
            Energy_data is a list of tuples, including the following:
            - X: the descriptors for a given structure: (N1, d)
            - E: total energy: scalor
            N1 is the number of atoms in the given structure
        """
        (X, ELE, indices, E) = list_to_tuple(energy_data, include_value=True, mode='energy')
        if len(self.train_x['energy']) == 3:
            (_X, _ELE, _indices) = self.train_x['energy']
            _X = np.concatenate((_X, X), axis=0)
            _indices.extend(indices)
            _ELE = np.concatenate((_ELE, ELE), axis=0)
            self.train_x['energy'] = (_X, _ELE, _indices)
            self.train_y['energy'].extend(E)
        else:
            self.train_x['energy'] = (X, ELE, indices)
            self.train_y['energy'] = E
        #self.update_y_train()

    def add_train_pts_force(self, force_data):
        """
        A function to add the force data to the training.

        Args:
            Force_data: a list of tuples (X, dXdR, F), where
                - X: the descriptors for a given structure: (N2, d)
                - dXdR: the descriptors: (N2, d, 3)
                - F: atomic force: 1*3
            N2 is the number of the centered atoms' neighbors
        """

        # pack the new data
        (X, dXdR, ELE, indices, F) = list_to_tuple(force_data, include_value=True)

        # stack the data to the existing data
        if len(self.train_x['force']) == 4:
            (_X, _dXdR, _ELE, _indices) = self.train_x['force']
            _X = np.concatenate((_X, X), axis=0)
            _indices.extend(indices)
            _ELE = np.concatenate((_ELE, ELE), axis=0)
            _dXdR = np.concatenate((_dXdR, dXdR), axis=0)

            self.train_x['force'] = (_X, _dXdR, _ELE, _indices)
            self.train_y['force'].extend(F)
        else:
            self.train_x['force'] = (X, dXdR, ELE, indices)
            self.train_y['force'] = F


    def save(self, filename, db_filename):
        """
        Save the model to the files

        Args:
            filename: the file to save json information
            db_filename: the file to save structural information
        """
        dict0 = self.save_dict(db_filename)
        with open(filename, "w") as fp:
            json.dump(dict0, fp, indent=4)
        self.export_ase_db(db_filename, permission="w")

        print("save the GP model to", filename, "and database to", db_filename)

    @classmethod
    def load(cls, filename, N_max=None, device='cpu'):
        """
        Load the model from files with MPI support

        Args:
            filename: the file to save json information
            db_filename: the file to save structural information
        """
        with open(filename, "r") as fp: dict0 = json.load(fp)
        instance = cls.load_from_dict(dict0, device=device)
        if instance.rank == 0:
            print(f"load GP model from {filename}")
            instance.logging.info(f"load GP model from {filename}")
        instance.extract_db(dict0["db_filename"], N_max)
        return instance

    def save_dict(self, db_filename):
        """
        Save the model as a dictionary in json
        """
        noise = {"energy": self.noise_e,
                 "force": self.noise_f,
                 "f_coef": self.f_coef,
                 "bounds": self.noise_bounds}

        dict0 = {"noise": noise,
                 "kernel": self.kernel.save_dict(),
                 "descriptor": self.descriptor.save_dict(),
                 "db_filename": db_filename,
                }

        if self.error is not None:
            dict0["error"] = self.error
        if self.base_potential is not None:
            dict0["base_potential"] = self.base_potential.save_dict()
        return dict0



    def export_ase_db(self, db_filename, permission="w"):
        """
        Export the structural information in ase db format, including
            - atoms: ase.Atoms object
            - energy: energy value
            _ forces: forces value
            - energy_in: bool whether the energy is included in the training
            - forces_in: bool whether the forces are included in the training

        Args:
            db_filename: the file to save structural information
            permission: "w" or "a+", reset or append the training data
        """
        if permission=="w" and os.path.exists(db_filename):
            os.remove(db_filename)

        with connect(db_filename, serial=True) as db:
            for _data in self.train_db:
                (struc, energy, force, energy_in, force_in, _, _) = _data
                actual_energy = deepcopy(energy)
                actual_forces = force.copy()
                if self.base_potential is not None:
                    energy_off, force_off, _ = self.compute_base_potential(struc)
                    actual_energy += energy_off
                    actual_forces += force_off

                data = {"energy": energy,
                        "force": force,
                        "energy_in": energy_in,
                        "force_in": force_in,
                       }
                kvp = {"dft_energy": actual_energy/len(force),
                       "dft_fmax": np.max(np.abs(actual_forces.flatten())),
                      }
                struc.set_constraint()
                db.write(struc, data=data, key_value_pairs=kvp)

    def extract_db(self, db_filename, N_max=None):
        """
        Convert the structures to descriptors from a given ASE database
        with MPI support

        Args:
            db_filename: the file to save structural information
            N_max: the maximum number of structures
        """

        # Initialize data structures
        rows_data = None
        n_total = None

        # Only rank 0 reads database
        if self.rank == 0:
            rows_data = []
            with connect(db_filename, serial=True) as db:
                for row in db.select():
                    atoms = db.get_atoms(id=row.id)
                    data = {
                        'atoms': atoms,
                        'energy': row.data.energy,
                        'force': row.data.force.copy(),
                        'energy_in': row.data.energy_in,
                        'force_in': row.data.force_in
                    }
                    rows_data.append(data)

                n_total = len(rows_data)
                if N_max is not None:
                    n_total = min(n_total, N_max)
                    rows_data = rows_data[:n_total]
                print(f"Rank 0: Loaded {n_total} structures")

        # Broadcast the n_total to all ranks
        n_total = self.comm.bcast(n_total, root=0)

        # Distribute rows across ranks
        if self.rank == 0:
            chunk_size = (n_total + self.size - 1) // self.size
            chunks = [rows_data[i:i + chunk_size] for i in range(0, n_total, chunk_size)]
            while len(chunks) < self.size:
                chunks.append([])
        else:
            chunks = None

        # Scatter the row_chunks to all ranks
        my_chunk = self.comm.scatter(chunks, root=0)
        # print(f"Rank {self.rank}: {len(my_chunk)} structures")

        # Process the data
        local_pts = {"energy": [], "force": [], "db": []}
        for data in my_chunk:
            atoms = data['atoms']
            energy = data['energy']
            force = data['force']
            energy_in = data['energy_in']
            force_in = data['force_in']

            # Calculate the descriptor
            d = self.descriptor.calculate(atoms)
            ele = [Element(ele).z for ele in d['elements']]
            ele = np.array(ele)

            if energy_in:
                local_pts["energy"].append((d['x'], energy/len(atoms), ele))

            for id in force_in:
                ids = np.argwhere(d['seq'][:,1]==id).flatten()
                _i = d['seq'][ids, 0]
                local_pts["force"].append((d['x'][_i,:], d['dxdr'][ids], force[id], ele[_i]))

            local_pts["db"].append((atoms, energy, force, energy_in, force_in))

        # print(f"Rank {self.rank}: Processed {len(local_pts['db'])} structures")
        all_pts = self.comm.gather(local_pts, root=0)

        # Initialize pts_to_add for all ranks
        pts_to_add = {"energy": [], "force": [], "db": []}

        # Combine results on rank 0
        if self.rank == 0:
            # Combine all gathered results
            for pts in all_pts:
                if pts["db"]:  # Only extend if there's data
                    pts_to_add["energy"].extend(pts["energy"])
                    pts_to_add["force"].extend(pts["force"])
                    pts_to_add["db"].extend(pts["db"])
            # print(f"Rank 0: Combined {len(pts_to_add['db'])} structures")

        # Broadcast the combined results to all ranks
        pts_to_add = self.comm.bcast(pts_to_add, root=0)

        # Add the structures to the training data
        self.set_train_pts(pts_to_add, "w")

    def _get_fixed_atoms(self, struc):
        """Helper to get indices of fixed atoms"""
        from ase.constraints import FixAtoms
        fix_ids = []
        if len(struc.constraints) > 0:
            for c in struc.constraints:
                if isinstance(c, FixAtoms):
                    fix_ids = c.get_indices()
                    break
        return fix_ids

    def predict_structure(self, struc, stress=True, return_std=False, f_tol=1e-8):
        """
        Make prediction for a given structure.
        This is a main API for the GPR model

        Args:
            struc: ase.Atoms object
            stress bool, return stress or not
            return_std bool, return variance or not
            f_tol float, precision to compute force
        """

        # Calculate the descriptor
        d = self.descriptor.calculate(struc, use_mpi=True)
        ele = [Element(ele).z for ele in d['elements']]
        ele = np.array(ele)
        data = {"energy": list_to_tuple([(d['x'], ele)], mode='energy')}
        #print(f"[Debug]-predict_structure in {self.rank}", d['x'].shape)
        force_data = np.empty(len(struc), dtype=object)
        x_shape = d['x'].shape[1]

        for i in range(len(struc)):
            ids = np.argwhere(d['seq'][:,1]==i).flatten()
            _i = d['seq'][ids, 0]
            _x, _dxdr, ele0 = d['x'][_i,:], d['dxdr'][ids], ele[_i]

            if stress:
                _rdxdr = d['rdxdr'][ids].reshape(len(ids), x_shape, 9)[:, :, [0, 4, 8, 1, 2, 5]]
                force_data[i] = (_x, np.concatenate((_dxdr, _rdxdr), axis=2), ele0)
            else:
                force_data[i] = (_x, _dxdr, ele0)
        data["force"] = force_data
        #print(f"[Debug]-predict_structure in {self.rank}", list_to_tuple(force_data)[0].shape)

        fix_ids = self._get_fixed_atoms(struc)
        train_x = self.get_train_x()
        if stress:
            K_trans, K_trans1 = self.kernel.k_total_with_stress(data, train_x, f_tol)
        else:
            K_trans = self.kernel.k_total(data, train_x, f_tol)

        pred = K_trans.dot(self.alpha_)
        #print('debug rank-alpha', self.rank, self.alpha_[:5, 0])
        #print('debug rank-K_trans', self.rank, '\n', K_trans[:4, :4])
        #print('debug rank-pred', self.rank, pred[:5, 0])
        y_mean = pred[:, 0]
        y_mean[0] *= len(struc) #total energy
        E = y_mean[0]
        F = y_mean[1:].reshape([len(struc), 3])
        if len(fix_ids) > 0: F[fix_ids] = 0.0

        if stress:
            S = K_trans1.dot(self.alpha_)[:,0].reshape([len(struc), 6])
        else:
            S = None

        # substract the energy/force offsets due to the base_potential
        if self.base_potential is not None:
            energy_off, force_off, stress_off = self.compute_base_potential(struc)
            E += energy_off
            F += force_off
            if stress:
                S += stress_off
        if return_std:
            y_var = self.kernel.diag(data)
            y_var -= np.einsum("ij,ij->i", np.dot(K_trans, self._K_inv), K_trans)
            y_var_negative = y_var < 0
            y_var[y_var_negative] = 0.0
            y_var = np.sqrt(y_var)
            E_std = y_var[0]  # eV/atom
            F_std = y_var[1:].reshape([len(struc), 3])
            if len(fix_ids) > 0: F_std[fix_ids] = 0.0
            # print(f"[Debug]-predict_structure in {self.rank}", E_std, fix_ids)

            return E, F, S, E_std, F_std
        else:
            return E, F, S


    def add_structure(self, data, N_max=10, tol_e_var=1.2, tol_f_var=1.2, add_force=True):
        """
        Add the training points from a given structure base on the followings:
            1, compute the (E, F, E_var, F_var) based on the current model
            2, if E_var is greater than tol_e_var, add energy data
            3, if F_var is greater than tol_f_var, add force data
        This is a main API for the GPR model

        Args:
            data: a tuple of (atoms, energy, force)
            N_max: the maximum number of force points to add
            tol_e_var: the threshold for energy variance
            tol_f_var: the threshold for force variance
            add_force: bool, add force data or not
        """
        tol_e_var *= self.noise_e
        tol_f_var *= self.noise_f

        pts_to_add = {"energy": [], "force": [], "db": []}
        (atoms, energy, force) = data

        # substract the energy/force offsets due to the base_potential
        if self.base_potential is not None:
            energy_off, force_off, _ = self.compute_base_potential(atoms)
        else:
            energy_off, force_off = 0, np.zeros((len(atoms), 3))

        energy -= energy_off
        force -= force_off
        my_data = convert_train_data([(atoms, energy, force)], self.descriptor)

        if self.alpha_ is not None:
            E, E1, E_std, F, F1, F_std = self.validate_data(my_data, return_std=True)
            E_std = E_std[0]
            F_std = F_std.reshape((len(atoms), 3))
        else:
            # If the model is not trained, set the variance to be large
            E = E1 = [energy / len(atoms)] # eV/atom
            F = F1 = force.flatten()
            E_std = 2 * tol_e_var
            F_std = 2 * tol_f_var * np.ones((len(atoms), 3))

        # QZ: to debug
        #if self.rank == 0:
        #    print(f"Debug: E_std: {E_std}, dE: {E[0]-E1[0]}, E: {E[0]}, E1: {E1[0]}")
        if E_std > tol_e_var or abs(E[0] - E1[0]) > 1.2 * tol_e_var:
            pts_to_add["energy"] = my_data["energy"]
            N_energy = 1
            energy_in = True
        else:
            N_energy = 0
            energy_in = False

        force_in = []
        if add_force:
            xs_added = []
            for f_id in range(len(atoms)):
                include = False
                if np.max(F_std[f_id]) > tol_f_var or np.max(abs(F[f_id] - F1[f_id])) > 1.5 * tol_f_var:
                    X = my_data["energy"][0][0][f_id]
                    _ele = my_data["energy"][0][2][f_id]
                    if len(xs_added) == 0:
                        include = True
                    else:
                        if new_pt((X, _ele), xs_added):
                            include = True
                if include:
                    force_in.append(f_id)
                    xs_added.append((X, _ele))
                    pts_to_add["force"].append(my_data["force"][f_id])
                if len(force_in) == N_max:
                    break

        N_forces = len(force_in)
        N_pts = N_energy + N_forces
        if N_pts > 0:
            pts_to_add["db"].append((atoms, energy, force, energy_in, force_in))
            self.set_train_pts(pts_to_add, mode="a+")
            #print("{:d} energy and {:d} forces will be added".format(N_energy, N_forces))
        errors = (E[0]+energy_off, E1[0]+energy_off, E_std,
                  F+force_off.flatten(), F1+force_off.flatten(), F_std)
        return pts_to_add, N_pts, errors

    def sparsify(self, e_tol=1e-10, f_tol=1e-10):
        """
        Sparsify the covariance matrix by removing unimportant
        configurations from the training database
        This is a main API for the GPR model
        """
        K = self.kernel.k_total(self.train_x)
        N_e = len(self.train_x["energy"][-1])
        N_f = len(self.train_x["force"][-1])

        pts_e = CUR(K[:N_e,:N_e], e_tol)
        pts = CUR(K[N_e:,N_e:], f_tol)
        pts_f = []
        if N_f > 1:
            for i in range(N_f):
                if len(pts[pts==i*3])==1 and len(pts[pts==(i*3+1)])==1 and len(pts[pts==(i*3+2)])==1:
                    pts_f.append(i)
        print("{:d} energy and {:d} forces will be removed".format(len(pts_e), len(pts_f)))
        if len(pts_e) + len(pts_f) > 0:
            self.remove_train_pts(pts_e, pts_f)

    @classmethod
    def set_GPR(cls, images, base_calculator,
                kernel='Dot', zeta=2.0, noise_e=0.002, noise_f=0.1,
                lmax=4, nmax=3, rcut=5.0, json=None):
        """
        Setup and train GPR model from images

        Args:
            images: list of images
            base_calculator: ase calculator
            kernel: kernel type (Dot or RBF)
            zeta: zeta value for the kernel
            noise_e: noise for energy
            noise_f: noise for forces
            lmax: lmax for the descriptor
            nmax: nmax for the descriptor
            rcut: cutoff radius for the descriptor
            json: json file to load the model
        """
        if json is not None and os.path.exists(json):
            instance = cls()
            instance.load(json)
        else:
            instance = cls(kernel=None, descriptor=None, base_potential=None)
            if kernel == 'Dot':
                instance.kernel = Dot_mb(para=[2, 2.0], zeta=zeta)
            else:
                instance.kernel = RBF_mb(para=[1.0, 0.1], zeta=zeta)
            instance.descriptor = SO3(nmax=nmax, lmax=lmax, rcut=rcut)
            instance.noise_e = noise_e
            instance.noise_f = noise_f

        # Train the initial model
        instance.train_images(images, base_calculator)
        return instance

    def train_images(self, images, base_calculator):
        """
        Function to train the GPR model from the images

        Args:
            model: gpr object
            images: list of images
            base_calculator: ase calculator e.g Vasp

        """
        for i, image in enumerate(images):
            if self.rank == 0:
                new_env = os.environ.copy()
                for var in ["OMPI_COMM_WORLD_SIZE",
                            "OMPI_COMM_WORLD_RANK",
                            "PMI_RANK",
                            "PMI_SIZE"]:
                    new_env.pop(var, None)

                # Set the calculator and calculate the energy and forces
                image.calc = base_calculator
                # For vasp calculator, set the directory for each image
                if hasattr(image.calc, 'set'):
                    image.calc.set(directory = f"neb_calc_{i}")
                eng = image.get_potential_energy()
                forces = image.get_forces()
                print(f"Calculate E/F for image {i}: {eng:.6f}")

            else:
                eng = 0.0
                forces = None

            # Reset calculator
            image.calc = None

            self.comm.Barrier()
            eng = self.comm.bcast(eng, root=0)
            forces = self.comm.bcast(forces, root=0)
            self.add_structure((image.copy(), eng, forces))

        self.fit()
        self.validate_data()
        self.set_K_inv()

    @classmethod
    def load_from_dict(cls, dict0, device='cpu'):
        """
        Load the model from dictionary

        Args:
            dict0: a dictionary of the model
            device: the device to run the model
        """

        instance = cls(kernel=None, descriptor=None, base_potential=None)
        #keys = ['kernel', 'descriptor', 'Noise']
        if dict0["kernel"]["name"] == "RBF_mb":
            instance.kernel = RBF_mb()
        elif dict0["kernel"]["name"] == "Dot_mb":
            instance.kernel = Dot_mb()
        else:
            msg = "unknow kernel {:s}".format(dict0["kernel"]["name"])
            raise NotImplementedError(msg)
        instance.kernel.load_from_dict(dict0["kernel"])

        if dict0["descriptor"]["_type"] == "SO3":
            instance.descriptor = SO3()
            instance.descriptor.load_from_dict(dict0["descriptor"])
        else:
            msg = "unknow descriptors {:s}".format(dict0["descriptor"]["name"])
            raise NotImplementedError(msg)

        if "base_potential" in dict0.keys():
            if dict0["base_potential"]["name"] == "LJ":
                from .calculator import LJ
                instance.base_potential = LJ()
                instance.base_potential.load_from_dict(dict0["base_potential"])
            else:
                msg = "unknow base potential {:s}".format(dict0["base_potential"]["name"])
                raise NotImplementedError(msg)
        instance.kernel.device = device
        instance.noise_e = dict0["noise"]["energy"]
        instance.noise_f = dict0["noise"]["force"]
        instance.f_coef = dict0["noise"]["f_coef"]
        instance.noise_bounds = dict0["noise"]["bounds"]
        # instance.noise_f = instance.f_coef * instance.noise_e

        return instance



def CUR(K, l_tol=1e-10):
    """
    This is a code to perform CUR decomposition for the covariance matrix:
    Appendix D in Jinnouchi, et al, Phys. Rev. B, 014105 (2019)

    Args:
        K: N*N covariance matrix
        N_e: number of
    """
    L, U = np.linalg.eigh(K)
    N_low = len(L[L<l_tol])
    omega = np.zeros(len(L))
    for j in range(len(L)):
        for eta in range(len(L)):
            if L[eta] < l_tol:
                omega[j] += U[j,eta]*U[j,eta]
    ids = np.argsort(-1*omega)
    return ids[:N_low]
