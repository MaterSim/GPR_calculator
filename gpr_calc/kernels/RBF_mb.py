import numpy as np
from mpi4py import MPI
from ..utilities import list_to_tuple
from .base import build_covariance, get_mask, K_ee_RBF
from .rbf_kernel import kee_C, kff_C, kef_C

class RBF_mb():
    r"""
    .. math::
        k(x_i, x_j) = \sigma ^2 * \\exp\\left(- \\frac{d(x_i, x_j)^2}{2l^2} \\right)

    Args:
        para: [sigma, l]
        bounds: [[sigma_min, sigma_max], [l_min, l_max]]
        zeta: power term
        ncpu: number of cpu cores
        device: cpu or cuda
    """

    def __init__(self,
                 para=[1., 1.],
                 bounds=[[1e-2, 5e+1], [1e-1, 1e+1]],
                 zeta=2,
                 ncpu=1,
                 device='cpu'):

        self.name = 'RBF_mb'
        self.bounds = bounds
        self.update(para)
        self.zeta = zeta
        self.device = device
        self.ncpu = ncpu

    def __str__(self):
        return "{:.5f}**2 *RBF(length={:.5f})".format(self.sigma, self.l)

    def load_from_dict(self, dict0):
        self.sigma = dict0["sigma"]
        self.l = dict0["l"]
        self.zeta = dict0["zeta"]
        self.bounds = dict0["bounds"]
        self.name = dict0["name"]

    def save_dict(self):
        """
        save the model as a dictionary in json
        """
        dict = {"name": self.name,
                "sigma": self.sigma,
                "l": self.l,
                "zeta": self.zeta,
                "bounds": self.bounds
               }
        return dict

    def parameters(self):
        return [self.sigma, self.l]

    def update(self, para):
        self.sigma, self.l = para[0], para[1]

    def diag(self, data):
        """
        Returns the diagonal of the kernel k(X, X)
        """
        sigma2, l2, zeta = self.sigma**2, self.l**2, self.zeta
        C_ee, C_ff = None, None
        # Get MPI info
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if "energy" in data:
            eng_data = data["energy"]
            if isinstance(eng_data, list):
                eng_data = list_to_tuple(eng_data, mode="energy")
            NE = eng_data[-1]
            C_ee = np.zeros(len(NE))
            count = 0
            for i, ne in enumerate(NE):
                x1, ele1 = eng_data[0][count:count+ne], eng_data[1][count:count+ne]
                mask = get_mask(ele1, ele1)
                C_ee[i] = K_ee_RBF(x1, x1, sigma2, l2, zeta, mask=mask)
                count += ne

        if "force" in data:
            NF = len(data["force"])
            force_data = data["force"]
            # Process force data
            if isinstance(force_data, (list, np.ndarray)):
                force_data = list_to_tuple(force_data, stress=False)
            x1, dx1dr, ele1, x1_indices = force_data

            # Calculate chunk size for each rank
            chunk_size = (NF + size - 1) // size
            start = rank * chunk_size
            end = min(start + chunk_size, NF)

            # Initialize local C_ff array
            local_C_ff = np.zeros(3*(end-start)) if start < NF else np.zeros(0)

            # Calculate local C_ff
            for i in range(start, end):
                if i < NF:
                    start1 = sum(x1_indices[:i])
                    end1 = sum(x1_indices[:i+1])
                    dat = (x1[start1:end1], dx1dr[start1:end1], ele1[start1:end1], [x1_indices[i]])
                    tmp = kff_C(dat, dat, self.sigma, self.l, self.zeta)
                    local_idx = i - start
                    local_C_ff[local_idx*3:(local_idx+1)*3] = np.diag(tmp)

            # Gather results to rank 0
            all_C_ff = comm.gather(local_C_ff, root=0)

            # Combine results on rank 0
            if rank == 0:
                C_ff = np.zeros(3 * NF)
                offset = 0
                for chunk in all_C_ff:
                    size = len(chunk)
                    if size > 0:
                        C_ff[offset:offset+len(chunk)] = chunk
                        offset += size
            # Broadcast the result to all ranks
            C_ff = comm.bcast(C_ff, root=0)

        #if rank == 0: print('debug', C_ff[:3])
        if C_ff is None:
            return C_ee
        elif C_ee is None:
            return C_ff
        else:
            return np.hstack((C_ee, C_ff))

    def k_total(self, data1, data2=None, f_tol=1e-10):
        """
        Compute the covairance for train data
        {"energy": [x, ele, indices], "force": [x, dx1dr, ele, indices]}
        Used for energy/force prediction

        Args:
            data1: dictionary of training data
            data2: dictionary of training data
            f_tol: tolerance for the force-force kernel
        """
        C_ee, C_ef, C_fe, C_ff = None, None, None, None

        same = False
        if data2 is None:
            data2 = data1
            same = True

        # Energy-energy terms
        if 'energy' in data1 and 'energy' in data2:
            C_ee = self._compute_K_ee(data1['energy'], data2['energy'])

        # Energy-force terms with MPI parallelization
        if 'energy' in data1 and 'force' in data2:
            C_ef = self._compute_K_ef(data1['energy'], data2['force'])

        if 'force' in data1 and 'energy' in data2:
            if not same:
                C_fe = self._compute_K_ef(data2['energy'], data1['force'], transpose=True)
            else:
                C_fe = C_ef.T if C_ef is not None else None

        # Force-force terms with MPI parallelization
        if 'force' in data1 and 'force' in data2:
            C_ff = self._compute_K_ff(data1['force'], data2['force'], tol=f_tol)
        #print("Debug-Cff", C_ee.shape, C_ef.shape, C_fe.shape, C_ff.shape)
        return build_covariance(C_ee, C_ef, C_fe, C_ff)

    def k_total_with_grad(self, data1, f_tol=1e-10):
        """
        Compute the covairance and gradient for training data

        Args:
            data1: dictionary of training (energy/force) data
            f_tol: tolerance for the force-force kernel
        """
        eng_data = data1['energy']
        force_data = data1['force']

        # Energy-energy terms
        C_ee, C_ee_s, C_ee_l = self._compute_K_ee(eng_data, eng_data, grad=True)

        # Energy-force terms
        C_ef, C_ef_s, C_ef_l = self._compute_K_ef(eng_data, force_data, grad=True)

        # Force-energy terms
        if C_ef is not None:
            C_fe, C_fe_s, C_fe_l = C_ef.T, C_ef_s.T, C_ef_l.T
        else:
            C_fe = C_fe_s = C_fe_l = None

        # Force-force terms
        C_ff, C_ff_s, C_ff_l = self._compute_K_ff(force_data, force_data, grad=True, tol=f_tol)

        # Build final matrices
        C = build_covariance(C_ee, C_ef, C_fe, C_ff, None, None)
        C_s = build_covariance(C_ee_s, C_ef_s, C_fe_s, C_ff_s, None, None)
        C_l = build_covariance(C_ee_l, C_ef_l, C_fe_l, C_ff_l, None, None)

        return C, np.dstack((C_s, C_l))

    def k_total_with_stress(self, data1, data2, tol=1e-10):
        """
        Compute the covairance
        Used for energy/force/stress prediction
        Obsolete function, needs to be updated later if needed
        """
        sigma, l, zeta = self.sigma, self.l, self.zeta
        C_ee, C_ef, C_fe, C_ff = None, None, None, None
        for key1 in data1.keys():
            d1 = data1[key1]
            for key2 in data2.keys():
                d2 = data2[key2]
                if len(d1)>0 and len(d2)>0:
                    if key1 == 'energy' and key2 == 'energy':
                        C_ee = kee_C(d1, d2, sigma, l, zeta)
                    elif key1 == 'energy' and key2 == 'force':
                        C_ef = kef_C(d1, d2, sigma, l, zeta)
                    elif key1 == 'force' and key2 == 'energy':
                        C_fe, C_se = kef_C(d2, d1, sigma, l, zeta, stress=True, transpose=True)
                    elif key1 == 'force' and key2 == 'force':
                        C_ff, C_sf = kff_C(d1, d2, sigma, l, zeta, stress=True, tol=tol)
        C = build_covariance(C_ee, C_ef, C_fe, C_ff)
        C1 = build_covariance(None, None, C_se, C_sf)
        return C, C1

    # Detailed K_ee, K_ff, K_ef functions

    def _compute_K_ee(self, eng_data1, eng_data2, grad=False, use_mpi=True):
        """
        Compute the energy-energy kernel with MPI parallelization
        Args:
            eng_data1: tuple of energy data
            eng_data2: tuple of energy data
            grad: whether to compute the gradient of the kernel
            use_mpi: whether use mpi or not
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        sigma, l, zeta = self.sigma, self.l, self.zeta
        # Process energy data
        if isinstance(eng_data1, list):
            eng_data1 = list_to_tuple(eng_data1, mode="energy")

        n_energies = len(eng_data1[-1])
        if n_energies == 0:
            if grad:
                return None, None, None
            else:
                return None

        if use_mpi and size > 1:
            # Calculate workload distribution for energy points
            chunk_size = (n_energies + size - 1) // size
            start = rank * chunk_size
            end = min(start + chunk_size, n_energies)

            # Get local data slice for energy points
            start1 = sum(eng_data1[2][:start])
            end1 = sum(eng_data1[2][:end])
            local_x = eng_data1[0][start1:end1]
            local_ele = eng_data1[1][start1:end1]
            local_indices = eng_data1[2][start:end]
            local_data = (local_x, local_ele, local_indices)

            # Compute local portion of energy-energy kernel
            if start < n_energies:
                if grad:
                    local_ee, local_ee_s, local_ee_l = kee_C(local_data, eng_data2,
                                                             sigma, l, zeta, grad=True)
                else:
                    local_ee = kee_C(local_data, eng_data2, sigma, l, zeta)
            else:
                local_ee = local_ee_s = local_ee_l = None

            # Gather results to rank 0
            #print("Debug-rank", start, end, rank, local_ee.shape)
            all_ee = comm.gather(local_ee, root=0)
            if grad:
                all_ee_s = comm.gather(local_ee_s, root=0)
                all_ee_l = comm.gather(local_ee_l, root=0)

            if rank == 0:
                C_ee = np.vstack([ee for ee in all_ee if ee is not None])
                if grad:
                    C_ee_s = np.vstack([ee_s for ee_s in all_ee_s if ee_s is not None])
                    C_ee_l = np.vstack([ee_l for ee_l in all_ee_l if ee_l is not None])
            else:
                C_ee = C_ee_s = C_ee_l = None

            # Broadcast the result to all ranks
            C_ee = comm.bcast(C_ee, root=0)
            # print("Debug-rank", rank, C_ee.shape)
            if grad:
                C_ee_s = comm.bcast(C_ee_s, root=0)
                C_ee_l = comm.bcast(C_ee_l, root=0)
        else:
            if grad:
                C_ee, C_ee_s, C_ee_l = kee_C(eng_data1, eng_data2, sigma, l, zeta, grad=True)
            else:
                C_ee = kee_C(eng_data1, eng_data2, sigma, l, zeta)

        if grad:
            return C_ee, C_ee_s, C_ee_l
        else:
            return C_ee


    def _compute_K_ef(self, eng_data, force_data, grad=False, transpose=False, use_mpi=True):
        """
        Compute the energy-force kernel with MPI parallelization

        Args:
            eng_data: tuple of energy data
            force_data: tuple of force data
            grad: whether to compute the gradient of the kernel
            transpose: whether to transpose the force data
            use_mpi: whether use mpi or not
        """
        sigma, l, zeta = self.sigma, self.l, self.zeta
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Process energy data
        if isinstance(eng_data, list):
            eng_data = list_to_tuple(eng_data, mode="energy")

        if isinstance(force_data, (list, np.ndarray)):
            force_data = list_to_tuple(force_data, stress=False)

        # Determine dimensions of the data
        n_energies = len(eng_data[-1])
        n_forces = len(force_data[-1])
        #print("Debug-Kef-nenergies/nforces", n_energies, n_forces)

        if n_energies == 0 or n_forces == 0:
            if grad:
                return None, None, None
            else:
                return None

        if use_mpi and size > 1:

            # Choose which dimension to parallelize based on relative sizes
            if n_energies > n_forces:
                force_split = False
                # Calculate workload distribution for energy points
                chunk_size = (n_energies + size - 1) // size
                start = rank * chunk_size
                end = min(start + chunk_size, n_energies)

                # Get local data slice for energy points
                start1 = sum(eng_data[2][:start])
                end1 = sum(eng_data[2][:end])
                local_x = eng_data[0][start1:end1]
                local_ele = eng_data[1][start1:end1]
                local_indices = eng_data[2][start:end] # QZ: double check this
                local_data = (local_x, local_ele, local_indices)

                # Compute local portion of energy-force kernel
                if start < n_energies:
                    if grad:
                        local_ef, local_ef_s, local_ef_l = kef_C(local_data, force_data,
                                                                 sigma, l, zeta,
                                                                 transpose=transpose, grad=True)
                    else:
                        local_ef = kef_C(local_data, force_data, sigma, l, zeta, transpose=transpose)
                else:
                    local_ef = local_ef_s = local_ef_l = None
                #print("Energy for split", start, end, local_ef.shape)
            else:
                force_split = True
                # Calculate workload distribution for force points
                chunk_size = (n_forces + size - 1) // size
                start = rank * chunk_size
                end = min(start + chunk_size, n_forces)

                # Get local force data slice
                x, dx1dr, ele, indices = force_data
                start1 = sum(indices[:start])
                end1 = sum(indices[:end])
                local_x = x[start1:end1]
                local_dx1dr = dx1dr[start1:end1]
                local_ele = ele[start1:end1]
                local_indices = indices[start:end]
                local_data = (local_x, local_dx1dr, local_ele, local_indices)

                # Compute local portion of energy-force kernel
                if start < n_forces:
                    if grad:
                        local_ef, local_ef_s, local_ef_l = kef_C(eng_data, local_data,
                                                                 sigma, l, zeta,
                                                                 transpose=transpose,
                                                                 grad=True)
                        #local_ef_s = local_ef_s.T
                        #local_ef_l = local_ef_l.T
                    else:
                        local_ef = kef_C(eng_data, local_data, sigma, l, zeta, transpose=transpose)
                        if transpose: local_ef = local_ef.T
                else:
                    local_ef = local_ef_s = local_ef_l = None
                #print("Force for split", start, end, local_ef.shape)

            # Gather results to rank 0
            all_ef = comm.gather(local_ef, root=0)
            if grad:
                all_ef_s = comm.gather(local_ef_s, root=0)
                all_ef_l = comm.gather(local_ef_l, root=0)

            # Combine results on rank 0
            if rank == 0:
                C_ef = np.hstack([ef for ef in all_ef if ef is not None])
                if transpose and force_split: C_ef = C_ef.T
                #print("Debug-rank", rank, C_ef.shape, transpose)
                if grad:
                    C_ef_s = np.hstack([ef_s for ef_s in all_ef_s if ef_s is not None])
                    C_ef_l = np.hstack([ef_l for ef_l in all_ef_l if ef_l is not None])
            else:
                C_ef = C_ef_s = C_ef_l = None

            # Broadcast the result to all ranks
            C_ef = comm.bcast(C_ef, root=0)
            if grad:
                C_ef_s = comm.bcast(C_ef_s, root=0)
                C_ef_l = comm.bcast(C_ef_l, root=0)
        else:
            if grad:
                C_ef, C_ef_s, C_ef_l = kef_C(eng_data, force_data, sigma, l, zeta,
                                             transpose=transpose, grad=True)
            else:
                C_ef = kef_C(eng_data, force_data, sigma, l, zeta,
                             transpose=transpose, grad=False)

        if grad:
            return C_ef, C_ef_s, C_ef_l
        else:
            return C_ef

    def _compute_K_ff(self, force_data1, force_data2, grad=False, diag=False, tol=1e-10):
        """
        Compute the force-force kernel with MPI parallelization

        Args:
            force_data1: tuple of force data
            force_data2: tuple of force data
            grad: whether to compute the gradient of the kernel
            diag: whether to compute the diagonal of the kernel
            tol: tolerance for the force-force kernel
        """
        sigma, l, zeta = self.sigma, self.l, self.zeta
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Process force data
        if isinstance(force_data1, (list, np.ndarray)):
            force_data1 = list_to_tuple(force_data1, stress=False)
        x1, dx1dr, ele1, x1_indices = force_data1

        # Calculate number of forces and divide work
        n_forces = len(x1_indices)
        if n_forces == 0:
            return None

        chunk_size = (n_forces + size - 1) // size
        start = rank * chunk_size
        end = min(start + chunk_size, n_forces)

        # Get local data slice
        start1 = sum(x1_indices[:start])
        end1 = sum(x1_indices[:end])
        x1_local = x1[start1:end1]
        dx1dr_local = dx1dr[start1:end1]
        ele1_local = ele1[start1:end1]
        x1_indices_local = x1_indices[start:end]

        # Compute local portion of force-force kernel
        local_ff_s = local_ff_l = None
        if start < n_forces:
            local_data = (x1_local, dx1dr_local, ele1_local, x1_indices_local)
            if grad:
                local_ff, local_ff_s, local_ff_l = kff_C(local_data, force_data2,
                                                         sigma, l, zeta, tol=tol, grad=True)
            elif diag:
                local_ff = kff_C(local_data, local_data, sigma, l, zeta, tol=tol, diag=True)
                local_ff = np.diag(local_ff)
            else:
                local_ff = kff_C(local_data, force_data2, sigma, l, zeta, diag=diag, tol=tol)
        else:
            local_ff = None

        # Gather results to rank 0
        all_ff = comm.gather(local_ff, root=0)
        if grad:
            all_ff_s = comm.gather(local_ff_s, root=0)
            all_ff_l = comm.gather(local_ff_l, root=0)

        # Combine results on rank 0
        if rank == 0:
            if diag:
                C_ff = np.hstack([ff for ff in all_ff if ff is not None])
            else:
                C_ff = np.vstack([ff for ff in all_ff if ff is not None])
                if grad:
                    C_ff_s = np.vstack([ff_s for ff_s in all_ff_s if ff_s is not None])
                    C_ff_l = np.vstack([ff_l for ff_l in all_ff_l if ff_l is not None])
        else:
            C_ff = C_ff_s = C_ff_l = None

        # Broadcast the result to all ranks
        C_ff = comm.bcast(C_ff, root=0)
        #if diag and rank == 0: print(f"[Debug]-Cff-Diag\n", C_ff[:3], C_ff.shape)
        if grad:
            C_ff_s = comm.bcast(C_ff_s, root=0)
            C_ff_l = comm.bcast(C_ff_l, root=0)
            return C_ff, C_ff_s, C_ff_l
        else:
            return C_ff
