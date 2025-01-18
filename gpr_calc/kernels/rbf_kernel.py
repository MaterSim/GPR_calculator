from cffi import FFI
import numpy as np
from ..utilities import list_to_tuple
from ._rbf_kernel import lib
from mpi4py import MPI

ffi = FFI()
def kee_C(X1, X2, sigma=1.0, l=1.0, zeta=2.0, grad=False):
    """
    Compute the energy-energy relation through RBF kernel.
    Args:
        X1: stack of ([X, ele, indices])
        X2: stack of ([X, ele, indices])
        sigma: 
        l:
        zeta:
        grad: if True, compute gradient w.r.t. hyperparameters
    Returns:
        C: the energy-energy kernel
        C_s: the energy-energy kernel derivative w.r.t. sigma
        C_l: the energy-energy kernel derivative w.r.t. l
    """
    sigma2, l2 = sigma*sigma, l*l

    if isinstance(X1, list):
        X1 = list_to_tuple(X1, mode='energy')

    (x1, ele1, x1_indices) = X1
    (x2, ele2, x2_indices) = X2

    x1_inds, x2_inds = [], []
    for i, ind in enumerate(x1_indices):
        x1_inds.extend([i]*ind)
    for i, ind in enumerate(x2_indices):
        x2_inds.extend([i]*ind)

    m1, m2, m1p, m2p, d = len(x1_indices), len(x2_indices), len(x1), len(x2), x1.shape[1]

    pdat_x1 = ffi.new('double['+str(m1p*d)+']', x1.ravel().tolist())
    pdat_x2 = ffi.new('double['+str(m2p*d)+']', x2.ravel().tolist())
    pdat_ele1 = ffi.new('int['+str(m1p)+']', ele1.tolist())
    pdat_ele2 = ffi.new('int['+str(m2p)+']', ele2.tolist())
    pdat_x1_inds=ffi.new('int['+str(m1p)+']', x1_inds)
    pdat_x2_inds=ffi.new('int['+str(m2p)+']', x2_inds)
    if grad:
        _l3 = 1 / (l * l2)
        pout = ffi.new('double['+str(m1*m2)+']')
        dpout_dl = ffi.new('double['+str(m1*m2)+']')
        lib.rbf_kee_many_with_grad(m1p, m2p, d, m2, zeta, sigma2, l2,
                               pdat_x1, pdat_ele1, pdat_x1_inds,
                               pdat_x2, pdat_ele2, pdat_x2_inds,
                               pout, dpout_dl)
        C = np.frombuffer(ffi.buffer(pout, m1*m2*8), dtype=np.float64)
        C.shape = (m1, m2)
        C /= (np.array(x1_indices)[:,None] * np.array(x2_indices)[None,:])
        C_l = np.frombuffer(ffi.buffer(dpout_dl, m1*m2*8), dtype=np.float64)
        C_l.shape = (m1, m2)
        C_l /= (np.array(x1_indices)[:,None] * np.array(x2_indices)[None,:])
        C_l *= _l3
        C_s = (2/sigma)*C
    else:
        pout=ffi.new('double['+str(m1*m2)+']')
        lib.rbf_kee_many(m1p, m2p, d, m2, zeta, sigma2, l2,
                     pdat_x1, pdat_ele1, pdat_x1_inds,
                     pdat_x2, pdat_ele2, pdat_x2_inds,
                     pout)
        C = np.frombuffer(ffi.buffer(pout, m1*m2*8), dtype=np.float64)
        C.shape = (m1, m2)
        C /= (np.array(x1_indices)[:,None] * np.array(x2_indices)[None,:])
        
    ffi.release(pdat_x1)
    ffi.release(pdat_ele1)
    ffi.release(pdat_x1_inds)
    ffi.release(pdat_x2)
    ffi.release(pdat_ele2)
    ffi.release(pdat_x2_inds)
    ffi.release(pout)
    if grad:
        ffi.release(dpout_dl)

    if grad:
        return C, C_s, C_l
    else:
        return C

def kef_C(X1, X2, sigma=1.0, l=1.0, zeta=2.0, grad=False, stress=False, transpose=False):
    """
    Compute the energy-force relation through RBF kernel.
    Args:
        X1: stack of ([X, ele, indices])
        X2: stack of ([X, dXdR, ele, indices])
        sigma: 
        l:
        zeta:
        grad: if True, compute gradient w.r.t. hyperparameters
        stress: if True, compute energy-stress relation
        transpose: if True, get the kfe
    Returns:
        C: the energy-force kernel
        C_s: the energy-force kernel derivative w.r.t. sigma
        C_l: the energy-force kernel derivative w.r.t. l
    """
    sigma2, l2 = sigma*sigma, l*l

    if isinstance(X1, list):
        X1 = list_to_tuple(X1, mode='energy')

    if isinstance(X2, list):
        X2 = list_to_tuple(X2, stress=stress)

    (x1, ele1, x1_indices) = X1
    (x2, dx2dr, ele2, x2_indices) = X2

    x1_inds, x2_inds = [], []
    for i, ind in enumerate(x1_indices):
        x1_inds.extend([i]*ind)
    for i, ind in enumerate(x2_indices):
        x2_inds.extend([i]*ind)

    m1, m2, m1p, m2p, d = len(x1_indices), len(x2_indices), len(x1), len(x2), x1.shape[1]
    
    pdat_x1 = ffi.new('double['+str(m1p*d)+']', x1.ravel().tolist())
    pdat_x2 = ffi.new('double['+str(m2p*d)+']', x2.ravel().tolist())
    pdat_ele1 = ffi.new('int['+str(m1p)+']', ele1.tolist())
    pdat_ele2 = ffi.new('int['+str(m2p)+']', ele2.tolist())
    pdat_x1_inds=ffi.new('int['+str(m1p)+']', x1_inds)
    pdat_x2_inds=ffi.new('int['+str(m2p)+']', x2_inds)

    if stress:
        pdat_dx2dr=ffi.new('double['+str(m2p*d*9)+']', list(dx2dr.ravel()))
        pout=ffi.new('double['+str(m1*m2*9)+']')
        lib.rbf_kef_many_stress(m1p, m2p, d, m2, zeta, sigma2, l2,
                     pdat_x1, pdat_ele1, pdat_x1_inds,
                     pdat_x2, pdat_dx2dr, pdat_ele2, pdat_x2_inds,
                     pout)
        d2 = 9
    elif grad:
        pdat_dx2dr=ffi.new('double['+str(m2p*d*6)+']', list(dx2dr.ravel()))
        pout=ffi.new('double['+str(m1*m2*6)+']')
        lib.rbf_kef_many_with_grad(m1p, m2p, d, m2, zeta, sigma2, l,
                              pdat_x1, pdat_ele1, pdat_x1_inds,
                              pdat_x2, pdat_dx2dr, pdat_ele2, pdat_x2_inds,
                              pout)
        d2 = 6
    else:
        pdat_dx2dr=ffi.new('double['+str(m2p*d*3)+']', list(dx2dr.ravel()))
        pout=ffi.new('double['+str(m1*m2*3)+']')
        lib.rbf_kef_many(m1p, m2p, d, m2, zeta, sigma2, l2,
                     pdat_x1, pdat_ele1, pdat_x1_inds,
                     pdat_x2, pdat_dx2dr, pdat_ele2, pdat_x2_inds,
                     pout)
        d2 = 3
    
    # convert cdata to np.array
    out = np.frombuffer(ffi.buffer(pout, m1*m2*d2*8), dtype=np.float64)
    out.shape = (m1, m2, d2)
    out /= np.array(x1_indices)[:,None,None]

    C = out[:, :, :3].reshape([m1, m2*3])
    if stress:
        Cs = out[:, :, 3:].reshape([m1, m2*6])
    elif grad:
        C_l = out[:, :, 3:].reshape([m1, m2*3])
        C_s = (2/sigma) * C
    else:
        Cs = np.zeros([m1, m2*6])

    ffi.release(pdat_x1)
    ffi.release(pdat_ele1)
    ffi.release(pdat_x1_inds)
    ffi.release(pdat_x2)
    ffi.release(pdat_dx2dr)
    ffi.release(pdat_ele2)
    ffi.release(pdat_x2_inds)
    ffi.release(pout)

    if transpose:
        C = C.T
        Cs = Cs.T
    if grad:
        return C, C_s, C_l
    elif stress:
        return C, Cs
    else:
        return C

def kff_C(X1, X2, sigma=1.0, l=1.0, zeta=2.0, grad=False, stress=False, tol=1e-12):
    """
    Compute the force-force relation through RBF kernel.
    Args:
        X1: stack of ([X, ele, indices])
        X2: stack of ([X, dXdR, ele, indices])
        sigma: 
        l:
        zeta:
        grad: if True, compute gradient w.r.t. hyperparameters
        stress: if True, compute force-stress relation
    Returns:
        C: the force-force kernel
        C_s: the force-force kernel derivative w.r.t. sigma
        C_l: the force-force kernel derivative w.r.t. l
    """
    comm=MPI.COMM_WORLD 
    rank=comm.Get_rank()

    sigma2, l2 = sigma*sigma, l*l

    if isinstance(X1, list):
        X1 = list_to_tuple(X1, stress=stress)

    (x1, dx1dr, ele1, x1_indices) = X1
    (x2, dx2dr, ele2, x2_indices) = X2

    x1_inds, x2_inds = [], []
    for i, ind in enumerate(x1_indices):
        x1_inds.extend([i]*ind)
    for i, ind in enumerate(x2_indices):
        x2_inds.extend([i]*ind)

    m1, m2, m1p, m2p, d = len(x1_indices), len(x2_indices), len(x1), len(x2), x1.shape[1]

    nprocs = comm.Get_size()
    ish=int(m2p/nprocs)
    irest=m2p-ish*nprocs
    ndim_pr = np.zeros([nprocs],dtype = int)
    n_mpi_pr = np.zeros([nprocs],dtype = int)
    for i in range(irest):
      ndim_pr[i]=ish+1
    for i in range(irest,nprocs):
      ndim_pr[i]=ish

    ind=0
    for i in range(nprocs):
       n_mpi_pr[i]=ind
       ind=ind+ndim_pr[i]
    m2p_start=n_mpi_pr[comm.rank]
    m2p_end=m2p_start+ndim_pr[comm.rank]
    
    pdat_x1=ffi.new('double['+str(m1p*d)+']', x1.ravel().tolist())
    pdat_x2=ffi.new('double['+str(m2p*d)+']', x2.ravel().tolist())
    pdat_ele1=ffi.new('int['+str(m1p)+']', ele1.tolist())
    pdat_ele2=ffi.new('int['+str(m2p)+']', ele2.tolist())
    pdat_x1_inds=ffi.new('int['+str(m1p)+']', x1_inds)
    pdat_x2_inds=ffi.new('int['+str(m2p)+']', x2_inds)
    pdat_dx2dr=ffi.new('double['+str(m2p*d*3)+']', dx2dr.ravel().tolist())

    if stress:
        pdat_dx1dr=ffi.new('double['+str(m1p*d*9)+']', list(dx1dr.ravel()))
        pout=ffi.new('double['+str(m1*9*m2*3)+']')
        lib.rbf_kff_many_stress(m1p, m2p, m2p_start, m2p_end, d, m2, zeta, sigma2, l2, tol, 
                     pdat_x1, pdat_dx1dr, pdat_ele1, pdat_x1_inds,
                     pdat_x2, pdat_dx2dr, pdat_ele2, pdat_x2_inds,
                     pout)
        out = np.frombuffer(ffi.buffer(pout, m1*9*m2*3*8), dtype=np.float64)
        out.shape = (m1, 9, m2*3)
        Cout = np.zeros([m1, 9, m2*3])
        comm.Barrier()
        comm.Allreduce(
            [out, MPI.DOUBLE],
            [Cout, MPI.DOUBLE],
            op = MPI.SUM,
            )

        C = Cout[:, :3, :].reshape([m1*3, m2*3])
        Cs = Cout[:, 3:, :].reshape([m1*6, m2*3])
    
    elif grad:
        pdat_dx1dr = ffi.new('double['+str(m1p*d*3)+']', dx1dr.ravel().tolist())
        pout = ffi.new('double['+str(m1*3*m2*3*2)+']')
        dpout_dl = ffi.new('double['+str(m1*3*m2*3*2)+']')
        lib.rbf_kff_many_with_grad(m1p, m2p, m2p_start, m2p_end, d, m2, zeta, sigma2, l, 
                               pdat_x1, pdat_dx1dr, pdat_ele1, pdat_x1_inds,
                               pdat_x2, pdat_dx2dr, pdat_ele2, pdat_x2_inds,
                               pout, dpout_dl)
        out = np.frombuffer(ffi.buffer(pout, m1*3*m2*3*8), dtype=np.float64)
        out.shape = (m1, 3, m2*3)
        Cout = np.zeros([m1, 3, m2*3])
        comm.Barrier()
        comm.Allreduce(
            [out, MPI.DOUBLE],
            [Cout, MPI.DOUBLE],
            op = MPI.SUM,
            )
        C = Cout[:, :3, :].reshape([m1*3, m2*3])

        dout_dl = np.frombuffer(ffi.buffer(dpout_dl, m1*3*m2*3*8), dtype=np.float64)
        dout_dl.shape = (m1, 3, m2*3)
        Cout = np.zeros([m1, 3, m2*3])
        comm.Barrier()
        comm.Allreduce(
            [dout_dl, MPI.DOUBLE],
            [Cout, MPI.DOUBLE],
            op = MPI.SUM,
            )

        C_l = Cout[:, :3, :].reshape([m1*3, m2*3])
        C_s = (2/sigma)*C

    else:
        pdat_dx1dr=ffi.new('double['+str(m1p*d*3)+']', dx1dr.ravel().tolist())
        pout=ffi.new('double['+str(m1*3*m2*3)+']')
        lib.rbf_kff_many(m1p, m2p, m2p_start, m2p_end, d, m2, zeta, sigma2, l2, tol,
                     pdat_x1, pdat_dx1dr, pdat_ele1, pdat_x1_inds,
                     pdat_x2, pdat_dx2dr, pdat_ele2, pdat_x2_inds,
                     pout)
        out = np.frombuffer(ffi.buffer(pout, m1*3*m2*3*8), dtype=np.float64)
        out.shape = (m1, 3, m2*3)
        Cout = np.zeros([m1, 3, m2*3])
        comm.Barrier()
        comm.Allreduce(
            [out, MPI.DOUBLE],
            [Cout, MPI.DOUBLE],
            op = MPI.SUM,
            )

        C = Cout[:, :3, :].reshape([m1*3, m2*3])

    ffi.release(pdat_x1)
    ffi.release(pdat_dx1dr)
    ffi.release(pdat_ele1)
    ffi.release(pdat_x1_inds)  
    ffi.release(pdat_x2)
    ffi.release(pdat_dx2dr)
    ffi.release(pdat_ele2)
    ffi.release(pdat_x2_inds)  
    ffi.release(pout)
    if grad:
        ffi.release(dpout_dl)
        return C, C_s, C_l
    elif stress:
        return C, Cs
    else:
        return C
