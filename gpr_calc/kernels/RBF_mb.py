import numpy as np
from ..utilities import list_to_tuple
from .base import build_covariance, get_mask
from .rbf_kernel import kee_C, kff_C, kef_C
from mpi4py import MPI

class RBF_mb():
    r"""
    .. math::
        k(x_i, x_j) = \sigma ^2 * \\exp\\left(- \\frac{d(x_i, x_j)^2}{2l^2} \\right)
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
            NE = data['energy'][-1]
            C_ee = np.zeros(len(NE))
            count = 0
            for i, ne in enumerate(NE):
                x1, ele1 = data['energy'][0][count:count+ne], data['energy'][1][count:count+ne]
                mask = get_mask(ele1, ele1)
                C_ee[i] = K_ee(x1, x1, sigma2, l2, zeta, mask=mask)
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

    def _compute_K_ff(self, force_data1, force_data2, same=False, grad=False, diag=False, tol=1e-10):
        """
        Compute the force-force kernel with MPI parallelization

        Args:
            force_data1: tuple of force data
            force_data2: tuple of force data
            same: whether the two force data sets are the same
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
                local_ff, local_ff_s, local_ff_l = kff_C(local_data,
                                                         force_data1,
                                                         sigma, l, zeta, tol=tol, grad=True)
            elif diag:
                local_ff = kff_C(local_data, local_data, sigma, l, zeta, tol=tol, diag=True)
                local_ff = np.diag(local_ff)
            else:
                local_ff = kff_C(local_data,
                                 force_data1 if same else force_data2,
                                 sigma, l, zeta, diag=diag, tol=tol)
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
        if diag and rank == 0: print(f"[Debug]-Cff-Diag\n", C_ff[:3], C_ff.shape)
        if grad:
            C_ff_s = comm.bcast(C_ff_s, root=0)
            C_ff_l = comm.bcast(C_ff_l, root=0)
            return C_ff, C_ff_s, C_ff_l
        else:
            return C_ff

    def _compute_K_ef(self, eng_data, force_data, grad=False, transpose=False, use_mpi=False):
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
        # Process energy data
        if isinstance(eng_data, list):
            eng_data = list_to_tuple(eng_data, mode="energy")

        if isinstance(force_data, (list, np.ndarray)):
            force_data = list_to_tuple(force_data, stress=False)

        # Determine dimensions of the data
        n_energies = len(eng_data[0])
        n_forces = len(force_data[-1]) if transpose else len(force_data[3])  # Use indices array length

        if n_energies == 0 or n_forces == 0:
            if grad:
                return None, None, None
            else:
                return None

        if use_mpi:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            # Choose which dimension to parallelize based on relative sizes
            if n_energies > n_forces:
                # Calculate workload distribution for energy points
                chunk_size = (n_energies + size - 1) // size
                start = rank * chunk_size
                end = min(start + chunk_size, n_energies)

                # Get local data slice for energy points
                local_x = eng_data[0][start:end]
                local_ele = eng_data[1][start:end]
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
            else:
                # Calculate workload distribution for force points
                chunk_size = (n_forces + size - 1) // size
                start = rank * chunk_size
                end = min(start + chunk_size, n_forces)

                # Get local force data slice
                x, dx1dr, ele, indices = force_data
                start_idx = sum(indices[:start])
                end_idx = sum(indices[:end])

                local_x = x[start_idx:end_idx]
                local_dx1dr = dx1dr[start_idx:end_idx]
                local_ele = ele[start_idx:end_idx]
                local_indices = indices[start:end]
                local_data = (local_x, local_dx1dr, local_ele, local_indices)

                # Compute local portion of energy-force kernel
                if start < n_forces:
                    if grad:
                        local_ef, local_ef_s, local_ef_l = kef_C(eng_data, local_data,
                                                                 sigma, l, zeta,
                                                                 transpose=transpose, grad=True)
                    else:
                        local_ef = kef_C(eng_data, local_data, sigma, l, zeta, transpose=transpose)
                else:
                    local_ef = local_ef_s = local_ef_l = None

            # Gather results to rank 0
            all_ef = comm.gather(local_ef, root=0)
            if grad:
                all_ef_s = comm.gather(local_ef_s, root=0)
                all_ef_l = comm.gather(local_ef_l, root=0)

            # Combine results on rank 0
            if rank == 0:
                C_ef = np.vstack([ef for ef in all_ef if ef is not None])
                if grad:
                    C_ef_s = np.vstack([ef_s for ef_s in all_ef_s if ef_s is not None])
                    C_ef_l = np.vstack([ef_l for ef_l in all_ef_l if ef_l is not None])
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

    def k_total(self, data1, data2=None, f_tol=1e-10):
        """
        Compute the covairance for train data
        Used for energy/force prediction

        Args:
            data1: dictionary of training data
            data2: dictionary of training data
            f_tol: tolerance for the force-force kernel
        """
        sigma, l, zeta = self.sigma, self.l, self.zeta
        C_ee, C_ef, C_fe, C_ff = None, None, None, None

        if data2 is None:
            data2 = data1
            same = True
        else:
            same = False

        # Energy-energy terms
        if 'energy' in data1 and 'energy' in data2:
            if len(data1['energy']) > 0 and len(data2['energy']) > 0:
                eng_data1 = data1['energy']
                if isinstance(eng_data1, list): eng_data1 = list_to_tuple(eng_data1, mode="energy")
                C_ee = kee_C(eng_data1, data2['energy'], sigma, l, zeta)
                #print(f"[Debug]-Cee-Rank-{rank}", C_ee[:5, :5])

        # Energy-force terms with MPI parallelization
        if 'energy' in data1 and 'force' in data2:
            C_ef = self._compute_K_ef(data1['energy'], data2['force'], use_mpi=False)

        if 'force' in data1 and 'energy' in data2:
            if len(data1['force']) > 0 and len(data2['energy']) > 0:
                if not same:
                    C_fe = self._compute_K_ef(data2['energy'], data1['force'], transpose=True, use_mpi=False)
                else:
                    C_fe = C_ef.T if C_ef is not None else None

        # Force-force terms with MPI parallelization
        if 'force' in data1 and 'force' in data2:
            C_ff = self._compute_K_ff(data1['force'], data2['force'], same=same, tol=f_tol)
            #print(f"[Debug]-Cff-Rank-{rank}\n", C_ff[:3, :3])
        return build_covariance(C_ee, C_ef, C_fe, C_ff)

    def k_total_with_grad(self, data1, f_tol=1e-10):
        """
        Compute the covairance and gradient for training data

        Args:
            data1: dictionary of training (energy/force) data
            f_tol: tolerance for the force-force kernel
        """
        sigma, l, zeta = self.sigma, self.l, self.zeta
        eng_data = data1['energy']
        force_data = data1['force']

        # Energy-energy terms
        if len(data1['energy']) > 0:
        	C_ee, C_ee_s, C_ee_l = kee_C(eng_data, eng_data, sigma, l, zeta, grad=True)
        else:
            C_ee = C_ee_s = C_ee_l = None

        # Energy-force terms
        if len(data1['energy']) > 0:
        	C_ef, C_ef_s, C_ef_l = self._compute_K_ef(eng_data, force_data, grad=True, use_mpi=False)

        # Force-energy terms
        if C_ef is not None:
            C_fe, C_fe_s, C_fe_l = C_ef.T, C_ef_s.T, C_ef_l.T
        else:
            C_fe = C_fe_s = C_fe_l = None

        # Force-force terms
        C_ff, C_ff_s, C_ff_l = self._compute_K_ff(force_data, force_data, same=True, grad=True, tol=f_tol)

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

# ===================== Standalone functions to compute K_ee, K_ef, K_ff

def K_ee(x1, x2, sigma2, l2, zeta=2, mask=None, eps=1e-8):
    """
    Compute the Kee between two structures
    Args:
        x1: [M, D] 2d array
        x2: [N, D] 2d array
        sigma2: float
        l2: float
        zeta: power term, float
        mask: to set the kernel zero if the chemical species are different
    """
    x1_norm = np.linalg.norm(x1, axis=1) + eps
    x2_norm = np.linalg.norm(x2, axis=1) + eps
    x1x2_dot = x1@x2.T
    d = x1x2_dot/(eps+x1_norm[:,None]*x2_norm[None,:])
    D = d**zeta

    k = sigma2*np.exp(-(0.5/l2)*(1-D))
    if mask is not None: k[mask] = 0

    Kee = k.sum(axis=0)
    m = len(x1)
    n = len(x2)
    return Kee.sum()/(m*n)

def K_ff(x1, x2, dx1dr, dx2dr, sigma2, l2, zeta=2, mask=None, eps=1e-8):
    """
    Compute the Kff between one and many configurations
    x2, dx1dr, dx2dr will be called from the cuda device in the GPU mode
    """
    x1_norm = np.linalg.norm(x1, axis=1) + eps
    x1_norm2 = x1_norm**2
    x1_norm3 = x1_norm**3
    x1x2_dot = x1@x2.T
    x1_x1_norm3 = x1/x1_norm3[:,None]

    x2_norm = np.linalg.norm(x2, axis=1) + eps
    x2_norm2 = x2_norm**2
    tmp30 = np.ones(x2.shape)/x2_norm[:,None]
    tmp33 = np.eye(x2.shape[1])[None,:,:] - x2[:,:,None] * (x2/x2_norm2[:,None])[:,None,:]


    x2_norm3 = x2_norm**3
    x1x2_norm = x1_norm[:,None]*x2_norm[None,:]
    x2_x2_norm3 = x2/x2_norm3[:,None]

    d = x1x2_dot/(eps+x1x2_norm)
    D2 = d**(zeta-2)
    D1 = d*D2
    D = d*D1
    k = sigma2*np.exp(-(0.5/l2)*(1-D))

    if mask is not None:
        k[mask] = 0

    dk_dD = (-0.5/l2)*k
    zd2 = -0.5/l2*zeta*zeta*(D1**2)

    tmp31 = x1[:,None,:] * tmp30[None,:,:]

    #t0 = time()
    tmp11 = x2[None, :, :] * x1_norm[:, None, None]
    tmp12 = x1x2_dot[:,:,None] * (x1/x1_norm[:, None])[:,None,:]
    tmp13 = x1_norm2[:, None, None] * x2_norm[None, :, None]
    dd_dx1 = (tmp11-tmp12)/tmp13

    tmp21 = x1[:, None, :] * x2_norm[None,:,None]
    tmp22 = x1x2_dot[:,:,None] * (x2/x2_norm[:, None])[None,:,:]
    tmp23 = x1_norm[:, None, None] * x2_norm2[None, :, None]
    dd_dx2 = (tmp21-tmp22)/tmp23  # (29, 1435, 24)


    tmp31 = tmp31[:,:,None,:] * x1_x1_norm3[:,None,:,None]
    tmp32 = x1_x1_norm3[:,None,:,None] * x2_x2_norm3[None,:,None,:] * x1x2_dot[:,:,None,None]
    out1 = tmp31-tmp32
    out2 = tmp33[None,:,:,:]/x1x2_norm[:,:,None,None]
    d2d_dx1dx2 = out2 - out1

    dd_dx1_dd_dx2 = dd_dx1[:,:,:,None] * dd_dx2[:,:,None,:]
    dD_dx1_dD_dx2 = zd2[:,:,None,None] * dd_dx1_dd_dx2

    d2D_dx1dx2 = dd_dx1_dd_dx2 * D2[:,:,None,None] * (zeta-1)
    d2D_dx1dx2 += D1[:,:,None,None]*d2d_dx1dx2
    d2D_dx1dx2 *= zeta
    d2k_dx1dx2 = -d2D_dx1dx2 + dD_dx1_dD_dx2 # m, n, d1, d2

    tmp0 = d2k_dx1dx2 * dk_dD[:,:,None,None] #n1, n2, d, d
    _kff1 = (dx1dr[:,None,:,None,:] * tmp0[:,:,:,:,None]).sum(axis=(0,2)) # n1,n2,3
    kff = (_kff1[:,:,:,None] * dx2dr[:,:,None,:]).sum(axis=1)  # n2, 3, 9
    kff = kff.sum(axis=0)
    return kff


