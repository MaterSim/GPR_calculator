import numpy as np
from .base import build_covariance, get_mask
from .rbf_kernel import kee_C, kff_C, kef_C

class RBF_mb():
    r"""
    .. math::
        k(x_i, x_j) = \sigma ^2 * \\exp\\left(- \\frac{d(x_i, x_j)^2}{2l^2} \\right)
    """
    def __init__(self, para=[1., 1.], bounds=[[1e-2, 5e+1], [1e-1, 1e+1]], zeta=2, device='cpu'):
        self.name = 'RBF_mb'
        self.bounds = bounds
        self.update(para)
        self.zeta = zeta
        self.device = device
            
    def __str__(self):
        return "{:.3f}**2 *RBF(length={:.3f})".format(self.sigma, self.l)

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
               
        if "energy" in data:
            try:
                NE = len(data["energy"])
                C_ee = np.zeros(NE)
                for i in range(NE):
                    (x1, ele1) = data["energy"][i]
                    mask = get_mask(ele1, ele1)
                    C_ee[i] = K_ee(x1, x1, sigma2, l2, zeta, mask=mask) 
            except:
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
            C_ff = np.zeros(3*NF)
            for i in range(NF):
                (x1, dx1dr, ele1) = data["force"][i]
                mask = get_mask(ele1, ele1)
                tmp = K_ff(x1, x1, dx1dr, dx1dr, sigma2, l2, zeta, mask)
                #tmp = K_ff(x1, x1, dx1dr[:,:,:3], dx1dr[:,:,:3], sigma2, l2, zeta, mask=mask)
                C_ff[i*3:(i+1)*3] = np.diag(tmp)

        if C_ff is None:
            return C_ee
        elif C_ee is None:
            return C_ff
        else:
            return np.hstack((C_ee, C_ff))

    def k_total(self, data1, data2=None, tol=1e-10):
        """
        Compute the covairance for train data
        Used for energy/force prediction
        """
        sigma, l, zeta = self.sigma, self.l, self.zeta

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
                        C_ee = kee_C(d1, d2, sigma, l, zeta)
                    elif key1 == 'energy' and key2 == 'force':
                        C_ef = kef_C(d1, d2, sigma, l, zeta)
                    elif key1 == 'force' and key2 == 'energy':
                        if not same:
                            C_fe = kef_C(d2, d1, sigma, l, zeta, transpose=True)
                        else:
                            C_fe = C_ef.T 
                    elif key1 == 'force' and key2 == 'force':
                        C_ff = kff_C(d1, d2, sigma, l, zeta, tol=tol)
        # print("C_ee", C_ee)               
        # print("C_ef", C_ef)               
        # import sys; sys.exit()
        return build_covariance(C_ee, C_ef, C_fe, C_ff)
        
    def k_total_with_grad(self, data1):
        """
        Compute the covairance for train data
        Used for energy/force training
        """
        sigma, l, zeta = self.sigma, self.l, self.zeta

        data2 = data1   
        C_ee, C_ef, C_fe, C_ff = None, None, None, None
        C_ee_s, C_ef_s, C_fe_s, C_ff_s = None, None, None, None
        C_ee_l, C_ef_l, C_fe_l, C_ff_l = None, None, None, None
        for key1 in data1.keys():
            d1 = data1[key1]
            for key2 in data2.keys():
                d2 = data2[key2]
                if len(d1)>0 and len(d2)>0:
                    if key1 == 'energy' and key2 == 'energy':
                        C_ee, C_ee_s, C_ee_l = kee_C(d1, d2, sigma, l, zeta, grad=True)
                    elif key1 == 'energy' and key2 == 'force':
                        C_ef, C_ef_s, C_ef_l = kef_C(d1, d2, sigma, l, zeta, grad=True)
                        C_fe, C_fe_s, C_fe_l = C_ef.T, C_ef_s.T, C_ef_l.T
                    elif key1 == 'force' and key2 == 'force':
                        C_ff, C_ff_s, C_ff_l = kff_C(d1, d2, sigma, l, zeta, grad=True)
        C = build_covariance(C_ee, C_ef, C_fe, C_ff, None, None)
        C_s = build_covariance(C_ee_s, C_ef_s, C_fe_s, C_ff_s, None, None)
        C_l = build_covariance(C_ee_l, C_ef_l, C_fe_l, C_ff_l, None, None)
        return C, np.dstack((C_s, C_l))

    def k_total_with_stress(self, data1, data2, tol=1e-10):
        """
        Compute the covairance
        Used for energy/force/stress prediction
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

def K_ee(x1, x2, sigma2, l2, zeta=2, grad=False, mask=None, eps=1e-8):
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
    if mask is not None:
        k[mask] = 0
    dk_dD = (-0.5/l2)*k

    Kee = k.sum(axis=0)
    m = len(x1)
    n = len(x2)
    return Kee.sum()/(m*n)

def K_ff(x1, x2, dx1dr, dx2dr, sigma2, l2, zeta=2, mask=None, eps=1e-8):
    """
    Compute the Kff between one and many configurations
    x2, dx1dr, dx2dr will be called from the cuda device in the GPU mode
    """
    l = np.sqrt(l2)
    l3 = l*l2

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

