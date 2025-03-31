import numpy as np
from .base import build_covariance, get_mask
from .dot_kernel import kee_C, kff_C, kef_C

class Dot_mb():
    r"""
    .. math::
        k(x_i, x_j) = \sigma ^2 * (\sigma_0 ^ 2 + x_i \cdot x_j)
    """
    def __init__(self, para=[1., 1.], bounds=[[1e-2, 5e+1], [1e-2, 1e+1]], zeta=3, device="cpu"):
        self.name = 'Dot'
        self.bounds = bounds
        self.update(para)
        self.zeta = zeta
        self.device = device

    def __str__(self):
        return "{:.3f}**2 *Dot(length={:.3f})".format(self.sigma, self.sigma0)

    def load_from_dict(self, dict0):
        self.sigma = dict0["sigma"]
        self.sigma0 = dict0["sigma0"]
        self.zeta = dict0["zeta"]
        self.bounds = dict0["bounds"]
        self.name = dict0["name"]
        
    def save_dict(self):
        """
        save the model as a dictionary in json
        """
        dict = {"name": self.name,
                "sigma": self.sigma,
                "sigma0": self.sigma0,
                "zeta": self.zeta,
                "bounds": self.bounds
               }
        return dict

    def parameters(self):
        return [self.sigma, self.sigma0]
 
    def update(self, para):
        self.sigma, self.sigma0 = para[0], para[1]

    def diag(self, data):
        """
        Returns the diagonal of the kernel k(X, X)
        """
        sigma2, sigma02, zeta = self.sigma**2, self.sigma0**2, self.zeta
        C_ee, C_ff = None, None
               
        if "energy" in data:
            try:
                NE = len(data["energy"])
                C_ee = np.zeros(NE)
                for i in range(NE):
                    (x1, ele1) = data["energy"][i]
                    mask = get_mask(ele1, ele1)
                    C_ee[i] = K_ee(x1, x1, sigma2, sigma02, zeta, mask) 

            except:
                NE = data['energy'][-1]
                C_ee = np.zeros(len(NE))
                count = 0
                for i, ne in enumerate(NE):
                    x1, ele1 = data['energy'][0][count:count+ne], data['energy'][1][count:count+ne]
                    mask = get_mask(ele1, ele1)
                    C_ee[i] = K_ee(x1, x1, sigma2, sigma02, zeta, mask)
                    count += ne

        if "force" in data:
            NF = len(data["force"])
            C_ff = np.zeros(3*NF)
            for i in range(NF):
                (x1, dx1dr, ele1) = data["force"][i]
                mask = get_mask(ele1, ele1)
                tmp = K_ff(x1, x1, dx1dr[:,:,:3], dx1dr[:,:,:3], sigma2, sigma02, zeta, mask)
                C_ff[i*3:(i+1)*3] = np.diag(tmp)

        if C_ff is None:
            return C_ee
        elif C_ee is None:
            return C_ff
        else:
            return np.hstack((C_ee, C_ff))

    def k_total(self, data1, data2=None, tol=1e-12):
        """
        Compute the covairance for train data
        Used for energy/force prediction
        # tol is not used
        """
        sigma, sigma0, zeta = self.sigma, self.sigma0, self.zeta

        if data2 is None:
            data2 = data1
            same = True
        else:
            same = False

        C_ee, C_ef, C_fe, C_ff = None, None, None, None
        for key1 in data1.keys():
            d1 = data1[key1]
            for key2 in data2.keys():
                d2 = data2[key2]
                if len(d1)>0 and len(d2)>0:
                    if key1 == 'energy' and key2 == 'energy':
                        C_ee = kee_C(d1, d2, sigma, sigma0, zeta)
                    elif key1 == 'energy' and key2 == 'force':
                        C_ef = kef_C(d1, d2, sigma, zeta)
                    elif key1 == 'force' and key2 == 'energy':
                        if not same:
                            C_fe = kef_C(d2, d1, sigma, zeta, transpose=True)
                        else:
                            C_fe = C_ef.T 
                    elif key1 == 'force' and key2 == 'force':
                        C_ff = kff_C(d1, d2, sigma, zeta)

        return build_covariance(C_ee, C_ef, C_fe, C_ff)
        
    def k_total_with_grad(self, data1):
        """
        Compute the covairance for train data
        Both C and its gradient will be returned
        """
        sigma, sigma0, zeta = self.sigma, self.sigma0, self.zeta

        data2 = data1   
        C_ee, C_ef, C_fe, C_ff = None, None, None, None
        C_ee1, C_ef1, C_fe1, C_ff1 = None, None, None, None
        C_ee2, C_ef2, C_fe2, C_ff2 = None, None, None, None
        for key1 in data1.keys():
            d1 = data1[key1]
            for key2 in data2.keys():
                d2 = data2[key2]
                if len(d1)>0 and len(d2)>0:
                    if key1 == 'energy' and key2 == 'energy':
                        C_ee, C_ee1, C_ee2 = kee_C(d1, d2, sigma, sigma0, zeta, grad=True)
                    elif key1 == 'energy' and key2 == 'force':
                        C_ef, C_ef1, C_ef2 = kef_C(d1, d2, sigma, sigma0, zeta, grad=True)
                        C_fe, C_fe1, C_fe2 = C_ef.T, C_ef1.T, C_ef2.T
                    elif key1 == 'force' and key2 == 'force':
                        C_ff, C_ff1, C_ff2 = kff_C(d1, d2, sigma, sigma0, zeta, grad=True)
        C = build_covariance(C_ee, C_ef, C_fe, C_ff, None, None)
        C1 = build_covariance(C_ee1, C_ef1, C_fe1, C_ff1, None, None)
        C2 = build_covariance(C_ee2, C_ef2, C_fe2, C_ff2, None, None)
        #print('\nC', C, '\nC1', C1, '\nC2', C2)
        return C, np.dstack((C1, C2))

    def k_total_with_stress(self, data1, data2, tol=1e-10):
        """
        Compute the covairance
        Used for energy/force/stress prediction
        # tol is not used
        """
        sigma, sigma0, zeta = self.sigma, self.sigma0, self.zeta
        C_ee, C_ef, C_fe, C_ff = None, None, None, None
        for key1 in data1.keys():
            d1 = data1[key1]
            for key2 in data2.keys():
                d2 = data2[key2]
                if len(d1)>0 and len(d2)>0:
                    if key1 == 'energy' and key2 == 'energy':
                        C_ee = kee_C(d1, d2, sigma, sigma0, zeta)
                    elif key1 == 'energy' and key2 == 'force':
                        C_ef = kef_C(d1, d2, sigma, zeta)
                    elif key1 == 'force' and key2 == 'energy':
                        C_fe, C_se = kef_C(d2, d1, sigma, zeta, stress=True, transpose=True)
                    elif key1 == 'force' and key2 == 'force':
                        C_ff, C_sf = kff_C(d1, d2, sigma, zeta, stress=True) 
        C = build_covariance(C_ee, C_ef, C_fe, C_ff)
        C1 = build_covariance(None, None, C_se, C_sf)
        return C, C1 
        
# ===================== Standalone functions to compute K_ee, K_ff

def K_ee(x1, x2, sigma2, sigma02, zeta=2, mask=None, eps=1e-8):
    """
    Compute the Kee between two structures
    Args:
        x1: [M, D] 2d array
        x2: [N, D] 2d array
        sigma2: float
        sigma02: float
        zeta: power term, float
        mask: to set the kernel zero if the chemical species are different
    """
    x1_norm = np.linalg.norm(x1, axis=1) + eps
    x2_norm = np.linalg.norm(x2, axis=1) + eps
    x1x2_dot = x1@x2.T
    d = x1x2_dot/(eps+x1_norm[:,None]*x2_norm[None,:])
    D = d**zeta

    dk_dD = sigma2*np.ones([len(x1), len(x2)])
    if mask is not None:
        dk_dD[mask] = 0

    Kee0 = dk_dD*(D+sigma02) # [m, n] * [m, n]
    Kee = Kee0.sum(axis=0)
    m = len(x1)
    n = len(x2)
    return Kee.sum()/(m*n)

def K_ff(x1, x2, dx1dr, dx2dr, sigma2, sigma02, zeta=2, mask=None, eps=1e-8):

    x1_norm = np.linalg.norm(x1, axis=1) + eps
    x1_norm2 = x1_norm**2       
    x1_norm3 = x1_norm**3        
    x1_x1_norm3 = x1/x1_norm3[:,None]
    x1x2_dot = x1@x2.T

    x2_norm = np.linalg.norm(x2, axis=1) + eps
    x2_norm2 = x2_norm**2      
    tmp30 = np.ones(x2.shape)/x2_norm[:,None]
    tmp33 = np.eye(x2.shape[1])[None,:,:] - x2[:,:,None] * (x2/x2_norm2[:,None])[:,None,:]

    x2_norm3 = x2_norm**3    
    x1x2_norm = x1_norm[:,None]*x2_norm[None,:]    
    x2_x2_norm3 = x2/x2_norm3[:,None]

    d = x1x2_dot/(x1_norm[:,None]*x2_norm[None,:])
    D2 = d**(zeta-2)
    D1 = d*D2
    D = d*D1

    dk_dD = sigma2*np.ones([len(x1), len(x2)])

    if mask is not None:
        dk_dD[mask] = 0

    tmp11 = x2[None, :, :] * x1_norm[:, None, None]
    tmp12 = x1x2_dot[:,:,None] * (x1/x1_norm[:, None])[:,None,:] 
    tmp13 = x1_norm2[:, None, None] * x2_norm[None, :, None] 
    dd_dx1 = (tmp11-tmp12)/tmp13 

    tmp21 = x1[:, None, :] * x2_norm[None,:,None]
    tmp22 = x1x2_dot[:,:,None] * (x2/x2_norm[:, None])[None,:,:]
    tmp23 = x1_norm[:, None, None] * x2_norm2[None, :, None] 
    dd_dx2 = (tmp21-tmp22)/tmp23 

    tmp31 = x1[:,None,:] * tmp30[None,:,:]
    tmp31 = tmp31[:,:,None,:] * x1_x1_norm3[:,None,:,None]
    tmp32 = x1_x1_norm3[:,None,:,None] * x2_x2_norm3[None,:,None,:] * x1x2_dot[:,:,None,None]
    out1 = tmp31-tmp32
    out2 = tmp33[None,:,:,:]/x1x2_norm[:,:,None,None]
    d2d_dx1dx2 = out2 - out1

    dd_dx1_dd_dx2 = dd_dx1[:,:,:,None] * dd_dx2[:,:,None,:]
    d2D_dx1dx2 = dd_dx1_dd_dx2 * D2[:,:,None,None] * (zeta-1)
    d2D_dx1dx2 += D1[:,:,None,None]*d2d_dx1dx2
    d2D_dx1dx2 *= zeta
    d2k_dx1dx2 = d2D_dx1dx2

    tmp0 = d2k_dx1dx2 * dk_dD[:,:,None,None] #n1, n2, d, d
    _kff1 = (dx1dr[:,None,:,None,:] * tmp0[:,:,:,:,None]).sum(axis=(0,2)) # n1,n2,3
    kff = (_kff1[:,:,:,None] * dx2dr[:,:,None,:]).sum(axis=1)  # n2, 3, 3
    kff = kff.sum(axis=0)
    return kff

