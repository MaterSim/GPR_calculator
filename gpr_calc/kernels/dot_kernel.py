from cffi import FFI
import numpy as np
from ..utilities import list_to_tuple
from ._dot_kernel import lib
from mpi4py import MPI

ffi = FFI()
def kee_C(X1, X2, sigma=1.0, sigma0=1.0, zeta=2.0, grad=False):
    """
    Compute the energy-force kernel between structures and atoms
    Args:
        X1: stacked ([X, ele, indices])
        X2: stacked ([X, dXdR, ele, indices])
    Returns:
        C
    """
    sigma2, sigma02 = sigma**2, sigma0**2
    if isinstance(X1, list):
        X1 = list_to_tuple(X1, mode='energy')
    (x1, ele1, x1_indices) = X1
    (x2, ele2, x2_indices) = X2
    x1_inds = []
    for i, ind in enumerate(x1_indices):
        x1_inds.extend([i]*ind)
    x2_inds = []
    for i, ind in enumerate(x2_indices):
        x2_inds.extend([i]*ind)

    m1, m2, m1p, m2p, d = len(x1_indices), len(x2_indices), len(x1), len(x2), len(x1[0])
    pdat_x1=ffi.new('double['+str(m1p*d)+']', list(x1.ravel()))
    pdat_x2=ffi.new('double['+str(m2p*d)+']', list(x2.ravel()))
    pdat_ele1=ffi.new('int['+str(m1p)+']', list(ele1))
    pdat_ele2=ffi.new('int['+str(m2p)+']', list(ele2))
    pdat_x1_inds=ffi.new('int['+str(m1p)+']', x1_inds)
    pdat_x2_inds=ffi.new('int['+str(m2p)+']', x2_inds)
    pout=ffi.new('double['+str(m1*m2)+']')
 
    lib.dot_kee_many(m1p, m2p, d, m2, zeta, sigma2, sigma02,
                 pdat_x1, pdat_ele1, pdat_x1_inds, 
                 pdat_x2, pdat_ele2, pdat_x2_inds, 
                 pout) 
    # convert cdata to np.array
    C = np.frombuffer(ffi.buffer(pout, m1*m2*8), dtype=np.float64)
    C.shape = (m1, m2)
    C /= (np.array(x1_indices)[:,None]*np.array(x2_indices)[None,:])
    
    ffi.release(pdat_x1)
    ffi.release(pdat_ele1)
    ffi.release(pdat_x1_inds)  
    ffi.release(pdat_x2)
    ffi.release(pdat_ele2)
    ffi.release(pdat_x2_inds)  
    ffi.release(pout)    

    if grad:
        C1 = 2 * C / sigma
        C2 = 0.8 * 2 * sigma2 * sigma0 * np.ones([m1, m2])
        #C2 /= (np.array(x1_indices)[:,None]*np.array(x2_indices)[None,:])
        #print('kee_dot', np.array(x1_indices)[:,None]*np.array(x2_indices)[None,:])
        return C, C1, C2
    else:
        return C


def kef_C(X1, X2, sigma=1.0, sigma0=1.0, zeta=2.0, grad=False, stress=False, transpose=False):
    """
    Compute the energy-force kernel between structures and atoms

    Args:
        X1: stacked ([X, ele, indices])
        X2: stacked ([X, dXdR, ele, indices])
        sigma:
        sigma0:
        zeta:
        grad:
        transpose:

    Returns:
        C
    """
    if isinstance(X1, list):
        X1 = list_to_tuple(X1, mode='energy')

    if isinstance(X2, list):
        X2 = list_to_tuple(X2, stress=stress)


    (x1, ele1, x1_indices) = X1
    (x2, dx2dr, ele2, x2_indices) = X2

    x1_inds = []
    for i, ind in enumerate(x1_indices):
        x1_inds.extend([i]*ind)
    x2_inds = []
    for i, ind in enumerate(x2_indices):
        x2_inds.extend([i]*ind)

    m1, m2, m1p, m2p, d = len(x1_indices), len(x2_indices), len(x1), len(x2), len(x1[0])

    # copy the arrays to memory
    pdat_x1=ffi.new('double['+str(m1p*d)+']', list(x1.ravel()))
    pdat_x2=ffi.new('double['+str(m2p*d)+']', list(x2.ravel()))
    pdat_ele1=ffi.new('int['+str(m1p)+']', list(ele1))
    pdat_ele2=ffi.new('int['+str(m2p)+']', list(ele2))
    pdat_x1_inds=ffi.new('int['+str(m1p)+']', x1_inds)
    pdat_x2_inds=ffi.new('int['+str(m2p)+']', x2_inds)
    if stress:
        pdat_dx2dr=ffi.new('double['+str(m2p*d*9)+']', list(dx2dr.ravel()))
        pout=ffi.new('double['+str(m1*m2*9)+']')
    else:
        pdat_dx2dr=ffi.new('double['+str(m2p*d*3)+']', list(dx2dr.ravel()))
        pout=ffi.new('double['+str(m1*m2*3)+']')

    if stress:
        lib.dot_kef_many_stress(m1p, m2p, d, m2, zeta,
                     pdat_x1, pdat_ele1, pdat_x1_inds, 
                     pdat_x2, pdat_dx2dr, pdat_ele2, pdat_x2_inds, 
                     pout) 
        d2 = 9
    else:
        lib.dot_kef_many(m1p, m2p, d, m2, zeta,
                     pdat_x1, pdat_ele1, pdat_x1_inds, 
                     pdat_x2, pdat_dx2dr, pdat_ele2, pdat_x2_inds, 
                     pout) 
        d2 = 3
    # convert cdata to np.array
    out = np.frombuffer(ffi.buffer(pout, m1*m2*d2*8), dtype=np.float64)
    out.shape = (m1, m2, d2)
    out /= np.array(x1_indices)[:,None,None]
    out *= -sigma*sigma
 
    C = out[:, :, :3].reshape([m1, m2*3])
    if stress:
        Cs = out[:, :, 3:].reshape([m1, m2*6])
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
        C1 = 2*C/sigma
        C2 = np.zeros([m1, m2*3])
        #C2 = 2 * sigma**2 * sigma0 * np.ones([m1, m2 * 3])
        return C, C1, C2
    elif stress:
        return C, Cs
    else:
        return C

def kff_C(X1, X2, sigma=1.0, sigma0=1.0, zeta=2.0, grad=False, stress=False):
    """
    Compute the energy-force kernel between structures and atoms
    Args:
        X1: stacked ([X, dXdR, ele, indices])
        X2: stacked ([X, dXdR, ele, indices])
    Returns:
        C
    """
    comm=MPI.COMM_WORLD 
    rank=comm.Get_rank()

    if isinstance(X1, list):
        X1 = list_to_tuple(X1, stress=stress)

    (x1, dx1dr, ele1, x1_indices) = X1
    (x2, dx2dr, ele2, x2_indices) = X2

    x1_inds = []
    for i, ind in enumerate(x1_indices):
        x1_inds.extend([i]*ind)
    x2_inds = []
    for i, ind in enumerate(x2_indices):
        x2_inds.extend([i]*ind)

    m1, m2, m1p, m2p, d = len(x1_indices), len(x2_indices), len(x1), len(x2), len(x1[0])

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


    # copy the arrays to memory
    pdat_x1=ffi.new('double['+str(m1p*d)+']', list(x1.ravel()))
    pdat_x2=ffi.new('double['+str(m2p*d)+']', list(x2.ravel()))
    pdat_ele1=ffi.new('int['+str(m1p)+']', list(ele1))
    pdat_ele2=ffi.new('int['+str(m2p)+']', list(ele2))
    pdat_x1_inds=ffi.new('int['+str(m1p)+']', x1_inds)
    pdat_x2_inds=ffi.new('int['+str(m2p)+']', x2_inds)
    pdat_dx2dr=ffi.new('double['+str(m2p*d*3)+']', list(dx2dr.ravel()))
    if stress:
        pdat_dx1dr=ffi.new('double['+str(m1p*d*9)+']', list(dx1dr.ravel()))
        pout=ffi.new('double['+str(m1*9*m2*3)+']')
        lib.dot_kff_many_stress(m1p, m2p, m2p_start, m2p_end, d, m2, zeta,
                     pdat_x1, pdat_dx1dr, pdat_ele1, pdat_x1_inds, 
                     pdat_x2, pdat_dx2dr, pdat_ele2, pdat_x2_inds, 
                     pout) 
        d1 = 9

    else:
        pdat_dx1dr=ffi.new('double['+str(m1p*d*3)+']', list(dx1dr.ravel()))
        pout=ffi.new('double['+str(m1*3*m2*3)+']')
        lib.dot_kff_many(m1p, m2p, m2p_start, m2p_end, d, m2, zeta,
                     pdat_x1, pdat_dx1dr, pdat_ele1, pdat_x1_inds, 
                     pdat_x2, pdat_dx2dr, pdat_ele2, pdat_x2_inds, 
                     pout) 
        d1 = 3

    # convert cdata to np.array

    out = np.frombuffer(ffi.buffer(pout, m1*d1*m2*3*8), dtype=np.float64)
    out.shape = (m1, d1, m2*3)
 
    #for i in range(m1*3):
    #    for j in range(m2*3):
    #        C[i, j]=pout[i*m2*3+j] 
    
    Cout = np.zeros([m1, d1, m2*3]) 
    comm.Barrier()
    comm.Allreduce(
       [out, MPI.DOUBLE],
       [Cout, MPI.DOUBLE],
       op = MPI.SUM,
       )
    #comm.Disconnect()
    #import sys; sys.exit()
    ffi.release(pdat_x1)
    ffi.release(pdat_dx1dr)
    ffi.release(pdat_ele1)
    ffi.release(pdat_x1_inds)  
    ffi.release(pdat_x2)
    ffi.release(pdat_dx2dr)
    ffi.release(pdat_ele2)
    ffi.release(pdat_x2_inds)  
    ffi.release(pout)    

    Cout *= (sigma*sigma*zeta)

    C = Cout[:, :3, :].reshape([m1*3, m2*3])
    if stress:
        Cs = Cout[:, 3:, :].reshape([m1*6, m2*3])
 
    if grad:
        C1 = 2*C/sigma
        #C2 = 2 * sigma**2 * sigma0 * np.ones([m1*3, m2*3])
        C2 = np.zeros([m1*3, m2*3])
        return C, C1, C2
    elif stress:
        return C, Cs
    else:
        return C


#X1 = np.load('../X2.npy', allow_pickle=True)
#X2 = np.load('../X2.npy', allow_pickle=True)
#X1_EE = np.load('X1_EE.npy', allow_pickle=True)
#X2_EE = np.load('X2_EE.npy', allow_pickle=True)
#X1_FF = np.load('X1_FF.npy', allow_pickle=True)
#X2_FF = np.load('X2_FF.npy', allow_pickle=True)
#sigma = 18.55544058601137
#sigma0 = 0.01
#
#t0 = time()
#C_EE = kee_C(X1_EE, X2_EE, sigma=sigma, sigma0=sigma0)
#print("KEE:", C_EE.shape)
#print(C_EE)
#C_EF = kef_C(X1_EE, X2_FF, sigma=sigma)
#print("KEF:", C_EF.shape)
#print(C_EF[0, :6])
#C_FE = kef_C(X2_EE, X1_FF, sigma=sigma)
#print("KFE:", C_FE.shape)
#print(C_FE.T[:6, :3])
#C_FF = kff_C(X1_FF, X2_FF, sigma=sigma)
#print("KFF:", C_FF.shape)
#print(C_FF[:6, :3])
#print("Elapsed time: ", time()-t0)
