from cffi import FFI
import numpy as np
from ..utilities import list_to_tuple
from ._rbf_kernel import lib

ffi = FFI()
def kee_C(X1, X2, sigma=1.0, l=1.0, zeta=2.0, grad=False):
    """
    Compute the energy-energy relation through RBF kernel.

    Args:
        X1: stack of ([X, ele, indices])
        X2: stack of ([X, ele, indices])
        sigma: hyperparameter
        l: lengthscale
        zeta: scaling factor
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
    pdat_x1_inds = ffi.new('int['+str(m1p)+']', x1_inds)
    pdat_x2_inds = ffi.new('int['+str(m2p)+']', x2_inds)
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
        sigma: hyperparameter
        l: lengthscale
        zeta: scaling factor
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

    #print("Debug: X2 type: ", type(X2), isinstance(X2, (list, np.ndarray)))
    if isinstance(X2, (list, np.ndarray)):
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
    pdat_x1_inds = ffi.new('int['+str(m1p)+']', x1_inds)
    pdat_x2_inds = ffi.new('int['+str(m2p)+']', x2_inds)

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

def kff_C(X1, X2, sigma=1.0, l=1.0, zeta=2.0, grad=False, stress=False, diag=False, tol=1e-12):
    """
    Compute the force-force relation through RBF kernel.

    Args:
        X1: stack of ([X, dXdR, ele, indices])
        X2: stack of ([X, dXdR, ele, indices])
        sigma: hyperparameter
        l: lengthscale
        zeta: scaling factor
        grad: if True, compute gradient w.r.t. hyperparameters
        stress: if True, compute force-stress relation
        diag: if True, compute diagonal of force-force kernel
        tol: tolerance for numerical stability

    Returns:
        C: the force-force kernel
        C_s: the force-force kernel derivative w.r.t. sigma
        C_l: the force-force kernel derivative w.r.t. l
    """
    sigma2, l2 = sigma*sigma, l*l

    if isinstance(X1, list):
        X1 = list_to_tuple(X1, stress=stress)
    if isinstance(X2, list):
        X2 = list_to_tuple(X2, stress=stress)

    (x1, dx1dr, ele1, x1_indices) = X1
    (x2, dx2dr, ele2, x2_indices) = X2

    # Calculate indices sizes first
    x1_inds = []
    x2_inds = []
    total_inds1 = sum(x1_indices)
    total_inds2 = sum(x2_indices)

    # Pre-allocate arrays of correct size
    x1_inds = np.zeros(total_inds1, dtype=np.int32)
    x2_inds = np.zeros(total_inds2, dtype=np.int32)

    # Fill indices arrays
    idx = 0
    for i, ind in enumerate(x1_indices):
        x1_inds[idx:idx+ind] = i
        idx += ind

    idx = 0
    for i, ind in enumerate(x2_indices):
        x2_inds[idx:idx+ind] = i
        idx += ind
    m1, m2 = len(x1_indices), len(x2_indices)
    m1p, m2p = len(x1), len(x2)
    d = x1.shape[1]

    # Set up CFFI arrays with verified sizes
    pdat_x1 = ffi.new('double[]', x1.ravel().tolist())
    pdat_x2 = ffi.new('double[]', x2.ravel().tolist())
    pdat_ele1 = ffi.new('int[]', ele1.tolist())
    pdat_ele2 = ffi.new('int[]', ele2.tolist())
    pdat_x1_inds = ffi.new('int[]', x1_inds.tolist())
    pdat_x2_inds = ffi.new('int[]', x2_inds.tolist())
    pdat_dx2dr = ffi.new('double[]', dx2dr.ravel().tolist())

    if stress:
        # Handle stress calculation
        pdat_dx1dr = ffi.new('double['+str(m1p*d*9)+']', list(dx1dr.ravel()))
        pout = ffi.new('double['+str(m1*9*m2*3)+']')
        lib.rbf_kff_many_stress(m1p, m2p, 0, m2p, d, m2, zeta, sigma2, l2, tol,
                               pdat_x1, pdat_dx1dr, pdat_ele1, pdat_x1_inds,
                               pdat_x2, pdat_dx2dr, pdat_ele2, pdat_x2_inds,
                               pout)
        out = np.frombuffer(ffi.buffer(pout, m1*9*m2*3*8), dtype=np.float64)
        out.shape = (m1, 9, m2*3)
        C = out[:, :3, :].reshape([m1*3, m2*3])
        Cs = out[:, 3:, :].reshape([m1*6, m2*3])

    elif grad:
        # Handle gradient calculation
        #print(f"[Debug] Size info - m1: {m1}, m2: {m2}, m1p: {m1p}, m2p: {m2p}, d: {d}")
        # Allocate memory for dx1dr array with explicit size check
        dx1dr_size = m1p * d * 3
        pdat_dx1dr = ffi.new(f'double[{dx1dr_size}]')
        
        # Copy data in chunks if needed
        chunk_size = 1000000  # Adjust based on available memory
        for i in range(0, len(dx1dr.ravel()), chunk_size):
            end = min(i + chunk_size, len(dx1dr.ravel()))
            pdat_dx1dr[i:end] = dx1dr.ravel()[i:end].tolist()

        # Allocate output arrays with size verification
        out_size = m1 * 3 * m2 * 3 * 2
        pout = ffi.new(f'double[{out_size}]')
        dpout_dl = ffi.new(f'double[{out_size}]')
        #print(f"[Debug] Allocated arrays - dx1dr: {dx1dr_size}, out: {out_size}")

        # Call C function with explicit error checking
        lib.rbf_kff_many_with_grad(
            m1p, m2p, 0, m2p, d, m2, zeta, sigma2, l,
            pdat_x1, pdat_dx1dr, pdat_ele1, pdat_x1_inds,
            pdat_x2, pdat_dx2dr, pdat_ele2, pdat_x2_inds,
            pout, dpout_dl
        )
        
        # Convert output to numpy arrays safely
        out = np.frombuffer(ffi.buffer(pout, m1*3*m2*3*8), dtype=np.float64)
        out = out.reshape((m1, 3, m2*3))
        C = out[:, :3, :].reshape([m1*3, m2*3])
        dout_dl = np.frombuffer(ffi.buffer(dpout_dl, m1*3*m2*3*8), dtype=np.float64)
        dout_dl = dout_dl.reshape((m1, 3, m2*3))
        C_l = dout_dl[:, :3, :].reshape([m1*3, m2*3])
        C_s = (2/sigma)*C
        #print(f"[Debug] Successfully processed arrays")
    elif diag:
        # Handle diagonal calculation
        pdat_dx1dr = ffi.new('double['+str(m1p*d*3)+']', dx1dr.ravel().tolist())
        pout = ffi.new('double['+str(m1*3*m2*3)+']')
        lib.rbf_kff_many_diag(m1p, d, m2, zeta, sigma2, l2, tol,
                        pdat_x1, pdat_dx1dr, pdat_ele1, pdat_x1_inds,
                        pdat_x2, pdat_dx2dr, pdat_ele2, pdat_x2_inds,
                        pout)
        out = np.frombuffer(ffi.buffer(pout, m1*3*m2*3*8), dtype=np.float64)
        out.shape = (m1, 3, m2*3)
        C = out[:, :3, :].reshape([m1*3, m2*3])
    else:
        # Handle standard calculation
        pdat_dx1dr = ffi.new('double['+str(m1p*d*3)+']', dx1dr.ravel().tolist())
        pout = ffi.new('double['+str(m1*3*m2*3)+']')
        lib.rbf_kff_many(m1p, m2p, 0, m2p, d, m2, zeta, sigma2, l2, tol,
                        pdat_x1, pdat_dx1dr, pdat_ele1, pdat_x1_inds,
                        pdat_x2, pdat_dx2dr, pdat_ele2, pdat_x2_inds,
                        pout)
        out = np.frombuffer(ffi.buffer(pout, m1*3*m2*3*8), dtype=np.float64)
        out.shape = (m1, 3, m2*3)
        C = out[:, :3, :].reshape([m1*3, m2*3])

    # Clean up CFFI resources
    for p in [pdat_x1, pdat_dx1dr, pdat_ele1, pdat_x1_inds,
              pdat_x2, pdat_dx2dr, pdat_ele2, pdat_x2_inds, pout]:
        ffi.release(p)

    if grad:
        ffi.release(dpout_dl)
        return C, C_s, C_l
    elif stress:
        return C, Cs
    else:
        return C